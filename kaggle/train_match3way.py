import os, sys, json, glob, urllib.request, traceback
os.environ["UNSLOTH_FORCE_FLOAT32"] = "1"
os.system("pip install --pre --upgrade --no-deps unsloth > /dev/null 2>&1")
os.system("pip install --no-deps --force-reinstall git+https://github.com/unslothai/unsloth-zoo.git > /dev/null 2>&1")
os.system("pip install --no-deps trl peft accelerate bitsandbytes datasets > /dev/null 2>&1")

PK_TOKEN = os.environ.get("PUSHOVER_TOKEN", "")
PK_USER = os.environ.get("PUSHOVER_USER", "")

def pushover(msg):
    try:
        body = json.dumps({"token": PK_TOKEN, "user": PK_USER, "message": "[match3way] " + msg}).encode()
        req = urllib.request.Request("https://api.pushover.net/1/messages.json", data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=20)
    except Exception as e:
        print("pushover fail:", e, flush=True)

try:
    import torch
    from unsloth import FastLanguageModel

    train_files = glob.glob("/kaggle/input/**/match3way_train_v1_80.jsonl", recursive=True)
    test_files = glob.glob("/kaggle/input/**/match3way_test_v1_20.jsonl", recursive=True)
    print("TRAIN FILE:", train_files, flush=True)
    print("TEST FILE:", test_files, flush=True)
    if not train_files or not test_files:
        print("DATASET NOT MOUNTED - re-push needed", flush=True)
        pushover("dataset not mounted - re-push needed")
        sys.exit(1)

    def load_jsonl(path):
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    train_pairs = load_jsonl(train_files[0])
    test_pairs = load_jsonl(test_files[0])
    print("train pairs:", len(train_pairs), "test pairs:", len(test_pairs), flush=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-4-E4B-it",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        device_map={"": 0},
    )

    probe = tokenizer.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False)
    print("TEMPLATE PROBE:", repr(probe), flush=True)

    model = FastLanguageModel.get_peft_model(
        model,
        r=16, alpha=16, dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )

    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from unsloth.chat_templates import train_on_responses_only

    train_ds = Dataset.from_list([{"messages": p["messages"]} for p in train_pairs])

    trainer = train_on_responses_only(
        SFTTrainer(
            model=model,
            train_dataset=train_ds,
            args=SFTConfig(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=8,
                num_train_epochs=3,
                learning_rate=2e-4,
                warmup_ratio=0.03,
                logging_steps=5,
                output_dir="/kaggle/working/out",
                report_to="none",
            ),
            tokenizer=tokenizer,
        ),
        instruction_part="<|turn>user",
        response_part="<|turn>model",
    )

    trainer.train()
    print("TRAIN HISTORY LAST:", trainer.state.log_history[-1], flush=True)
    pushover("train done: " + json.dumps(trainer.state.log_history[-1])[:200])

    model.save_pretrained("/kaggle/working/match3way_e4b_lora")
    tokenizer.save_pretrained("/kaggle/working/match3way_e4b_lora")
    print("LORA SAVED", flush=True)

    TOP = ["vendor","po_number","grn_number","invoice_number","invoice_date","currency","items","total_amount"]
    ITEMK = ["material","po_qty","grn_qty","inv_qty","po_unit_price","inv_unit_price"]

    def extract_json(text):
        s = text.find("{")
        if s == -1:
            return None
        depth = 0
        for i in range(s, len(text)):
            ch = text[i]
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try: return json.loads(text[s:i+1])
                    except Exception: return None
        return None

    def num_eq(a, b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
            return abs(float(a) - float(b)) < 0.01
        return a == b

    def compute_verdict(g):
        if g.get("po_number") is None: return "MISSING_PO"
        if g.get("grn_number") is None: return "MISSING_GRN"
        for it in g.get("items") or []:
            if it.get("po_unit_price") is None or it.get("inv_unit_price") is None: continue
            if abs(float(it["inv_unit_price"]) - float(it["po_unit_price"])) > 0.005: return "PRICE_MISMATCH"
        for it in g.get("items") or []:
            if it.get("grn_qty") is None or it.get("po_qty") is None: continue
            if it["grn_qty"] != it["po_qty"]: return "PARTIAL_DELIVERY"
        for it in g.get("items") or []:
            if it.get("inv_qty") is None or it.get("po_qty") is None: continue
            if it["inv_qty"] != it["po_qty"]: return "QTY_MISMATCH"
        return "OK"

    def flatten(g):
        flat = {}
        for k in TOP:
            if k == "items": continue
            flat[k] = g.get(k)
        for n, it in enumerate(g.get("items") or []):
            for k in ITEMK:
                flat["items.%d.%s" % (n, k)] = it.get(k) if isinstance(it, dict) else None
        return flat

    FastLanguageModel.for_inference(model)
    inner = getattr(tokenizer, "tokenizer", tokenizer)

    field_ok = {}
    field_total = {}
    json_valid = 0; all_ok = 0; invented_po = 0; invented_grn = 0
    verdict_ok = 0; verdict_total = 0
    results = []

    for n, c in enumerate(test_pairs):
        user = c["messages"][0]["content"]
        gold = json.loads(c["messages"][1]["content"])
        gold_verdict = c["meta"]["verdict"]
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        ids = inner(prompt, return_tensors="pt").input_ids.to("cuda")
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        try:
            out = model.generate(**inputs, max_new_tokens=700, do_sample=False)
            reply = inner.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        except Exception as e:
            reply = "ERROR: " + str(e)
        pred = extract_json(reply)
        row = {"case": c["meta"]["case"], "edges": c["meta"]["edges"], "json_valid": pred is not None, "reply_head": reply[:200]}
        if pred is not None:
            json_valid += 1
            gf = flatten(gold)
            pf_ = flatten(pred)
            oks = 0
            for k, gv in gf.items():
                base = k.split(".")[-1]
                field_total[base] = field_total.get(base, 0) + 1
                if k in pf_ and num_eq(pf_[k], gv):
                    oks += 1
                    field_ok[base] = field_ok.get(base, 0) + 1
            if oks == len(gf):
                all_ok += 1
            if gold.get("po_number") is None and pred.get("po_number") is not None: invented_po += 1
            if gold.get("grn_number") is None and pred.get("grn_number") is not None: invented_grn += 1
            try:
                verdict_total += 1
                if compute_verdict(pred) == gold_verdict: verdict_ok += 1
            except Exception: pass
            row["ok_fields"] = oks
            row["total_fields"] = len(gf)
        row["gold"] = gold
        results.append(row)
        print("case " + str(row["case"]) + ": json_valid=" + str(row["json_valid"]) + " ok=" + str(row.get("ok_fields", "-")), flush=True)

    total = len(test_pairs)
    summary = {
        "model": "gemma-4-E4B-it + LoRA r16 (fine-tuned on 80 3-way-match pairs)",
        "total": total,
        "json_valid": json_valid,
        "all_fields_exact": all_ok,
        "invented_po_when_missing": invented_po,
        "invented_grn_when_missing": invented_grn,
        "verdict_accuracy": round(verdict_ok / verdict_total, 3) if verdict_total else None,
        "field_accuracy": {k: round(field_ok.get(k, 0) / field_total[k], 3) for k in field_total},
    }
    with open("/kaggle/working/eval_finetuned_results.json", "w") as f:
        json.dump({"summary": summary, "cases": results}, f, indent=2)

    print("====SUMMARY====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("EVAL_DONE", flush=True)
    pushover("eval done: exact " + str(all_ok) + "/" + str(total) + ", verdict " + str(summary["verdict_accuracy"]))

except SystemExit:
    raise
except Exception:
    print(traceback.format_exc(), flush=True)
    pushover("KERNEL ERROR: " + traceback.format_exc()[-250:])
    raise
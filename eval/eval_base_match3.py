import json, os, subprocess, time, sys

DATA = "/root/workspace/projects/match3way_dataset"
TEST = os.path.join(DATA, "match3way_test_v1_20.jsonl")
OUT = os.path.join(DATA, "eval_base_results.json")
LLAMA = "/root/llama.cpp/build/bin/llama-cli"
MODEL = "/root/models/gguf/gemma-4-E4B-it-Q4_K_M.gguf"
PF = os.path.join(DATA, "_prompt.txt")
TOP = ["vendor","po_number","grn_number","invoice_number","invoice_date","currency","items","total_amount"]
ITEMK = ["material","po_qty","grn_qty","inv_qty","po_unit_price","inv_unit_price"]

def run_llama(user, chat_mode):
    if chat_mode:
        payload = user
        extra = ["-st"]
    else:
        payload = "<|turn>user" + chr(10) + user + "<turn|>" + chr(10) + "<|turn>model" + chr(10)
        extra = []
    with open(PF, "w") as f:
        f.write(payload)
    cmd = [LLAMA, "-m", MODEL, "-f", PF, "-n", "900", "--temp", "0", "-t", "4", "--no-display-prompt", "-co", "off"] + extra
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    return p.stdout, time.time() - t0

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

cases = []
with open(TEST) as f:
    for line in f:
        line = line.strip()
        if line: cases.append(json.loads(line))
print("cases:", len(cases), flush=True)

chat_mode = True
out0, dt0 = run_llama(cases[0]["messages"][0]["content"], True)
print("MODE PROBE len:", len(out0.strip()), flush=True)
if len(out0.strip()) < 20:
    chat_mode = False
    out0, dt0 = run_llama(cases[0]["messages"][0]["content"], False)
    print("FALLBACK RAW MODE, len:", len(out0.strip()), flush=True)
print("CHAT MODE:", chat_mode, flush=True)

results = []
field_ok = {}
field_total = {}
json_valid = 0; all_ok = 0; invented_po = 0; invented_grn = 0
verdict_ok = 0; verdict_total = 0

def score(n, c, out, dt):
    global json_valid, all_ok, invented_po, invented_grn, verdict_ok, verdict_total
    gold = json.loads(c["messages"][1]["content"])
    gold_verdict = c["meta"]["verdict"]
    pred = extract_json(out)
    row = {"case": c["meta"]["case"], "edges": c["meta"]["edges"], "seconds": round(dt, 1), "json_valid": pred is not None, "reply_head": out[:200]}
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
    results.append(row)
    print("case %s %s json_valid=%s ok=%s (%.1fs)" % (row["case"], row["edges"], row["json_valid"], row.get("ok_fields", "-"), dt), flush=True)

score(0, cases[0], out0, dt0)
for n in range(1, len(cases)):
    try:
        out, dt = run_llama(cases[n]["messages"][0]["content"], chat_mode)
    except Exception as e:
        out, dt = "ERROR: " + str(e), 0.0
    score(n, cases[n], out, dt)

total = len(cases)
summary = {
    "model": "gemma-4-E4B-it Q4_K_M (base, zero-shot, temp 0)",
    "total": total,
    "json_valid": json_valid,
    "all_fields_exact": all_ok,
    "invented_po_when_missing": invented_po,
    "invented_grn_when_missing": invented_grn,
    "verdict_accuracy": round(verdict_ok / verdict_total, 3) if verdict_total else None,
    "field_accuracy": {k: round(field_ok.get(k, 0) / field_total[k], 3) for k in field_total},
}
with open(OUT, "w") as f:
    json.dump({"summary": summary, "cases": results}, f, indent=2)
print("---- SUMMARY ----", flush=True)
print(json.dumps(summary, indent=2), flush=True)
print("=== BASELINE DONE ===", flush=True)
with open(os.path.join(DATA, "eval_base.done"), "w") as f:
    f.write("ok")
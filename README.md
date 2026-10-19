# 3-Way Match Bot — Fine-Tuning Worksheet (PO + GRN + Invoice → JSON)

Teaching a 4-bit open model the Accounts Payable **3-way match**: read messy procurement messages, extract PO + GRN + Invoice data into strict JSON, and return the match verdict.

Sequel to [invoice-e4b-finetuning-worksheet](https://github.com/romisap/invoice-e4b-finetuning-worksheet) — same methodology (dataset-as-code → zero-shot baseline → QLoRA → held-out eval), harder task.

## Task

- **Input:** one unstructured message describing three documents — purchase order, goods receipt, invoice. Formats: email, short text, tabular, WhatsApp-style. Sometimes a document is missing.
- **Output:** strict JSON: `vendor`, `po_number`, `grn_number`, `invoice_number`, `invoice_date`, `currency`, `items[{material, po_qty, grn_qty, inv_qty, po_unit_price, inv_unit_price}]`, `total_amount`.
- **Verdict** (computed deterministically from the JSON): `MISSING_PO > MISSING_GRN > PRICE_MISMATCH > PARTIAL_DELIVERY > QTY_MISMATCH > OK`.

## Dataset (as code, seed 42 — fully reproducible)

- `dataset/make_match3way_dataset.py` generates 100 cases: 4 formats, 6 edge types (clean 50%, qty_mismatch / price_mismatch / missing_grn / missing_po / partial delivery at 10% each)
- 80 training pairs, 20 held-out test cases — every edge type appears in the test set
- `match3way_golden_v1_100.jsonl` — the full 100-case golden set (all formats + edge types, reproducible from seed 42)

## Results (20 held-out cases)

| Metric | Base (zero-shot Q4_K_M, temp 0) | Fine-tuned (QLoRA r16) |
| --- | --- | --- |
| Valid JSON | 0/20 | **20/20** |
| All 13 fields exact | 0/20 | **20/20** |
| Verdict accuracy | n/a | **100%** |
| Invented PO / GRN when document missing | n/a | **0** (correctly returns null) |

Training: 80 pairs, QLoRA r16 a16 (q/k/v/o/gate/up/down), 3 epochs, **15 optimizer steps, ~6 min on a free Kaggle T4**. Loss 0.042 → 0.006 — but loss is not the quality metric; the held-out eval is.

> **Honest baseline note:** the base model READS the data fine — it extracts every field correctly in prose — but never follows the JSON output contract and hits the 900-token budget mid-analysis. 0/20 valid JSON is a formatting-contract failure, not a comprehension failure. Exactly what the LoRA teaches: not new knowledge, contract-following.

## Reproduce on Kaggle (free)

1. Dataset: https://www.kaggle.com/datasets/romibagga/match3way-dataset
2. Training kernel: https://www.kaggle.com/code/romibagga/match3way-e4b-train — fork it, enable GPU (T4), Run All. The kernel trains AND evaluates the 20 held-out cases in-kernel.
3. The full dataset is also in this repo (`dataset/`) — 100 golden cases + 80/20 train/test split `python dataset/make_match3way_dataset.py` (seed 42)

## Files

```
dataset/  make_match3way_dataset.py, match3way_golden_v1_100.jsonl, match3way_train_v1_80.jsonl, match3way_test_v1_20.jsonl
eval/    eval_base_match3.py (llama.cpp baseline harness), eval_base_results.json
kaggle/  train_match3way.py (unsloth QLoRA kernel), kernel-metadata.json
results/ eval_finetuned_results.json (20 held-out cases, per-field detail)
```

## License

MIT

4. Or regenerate the dataset locally:
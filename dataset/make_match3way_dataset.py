import json, random, os, datetime, hashlib
SEED = 42
OUT = "/root/workspace/projects/match3way_dataset"
os.makedirs(OUT, exist_ok=True)
random.seed(SEED)
NL = chr(10)
VENDORS = ["Sharma Traders Pvt Ltd", "Gupta Enterprises", "Verma Supplies Co", "Mehta and Sons", "Kiran Wholesale Mart", "Deepak Materials Pvt Ltd", "Bansal Packaging Industries", "Nanda Electricals", "Prakash Hardware Store", "Sundaram Textiles Ltd", "Kapoor Imports", "Reddy Agro Supplies", "Chawla Chemicals Pvt Ltd", "Iyer Logistics Solutions", "Grover Steel Traders"]
MATERIALS = ["LAPTOP-14I5", "PRINTER-A4L", "MONITOR-24F", "ROUTER-AX6", "SSD-1TB", "RAM-16GB", "KEYBOARD-WL", "MOUSE-USB", "CABLE-HDMI2", "CHAIR-ERGO", "DESK-L120", "UPS-1KVA"]
CURRS = [("INR", "Rs.", 18), ("AED", "AED ", 5), ("EUR", "EUR ", None)]
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
EDGES = ["clean","qty_mismatch","clean","price_mismatch","clean","missing_grn","clean","missing_po","clean","partial"]
STYLES = ["email","short","tabular","whatsapp"]

def money(x, sym):
    v = float(x)
    if abs(v - round(v)) < 0.005:
        return sym + format(int(round(v)), ",")
    return sym + format(v, ",.2f")

def rand_iso():
    return "2026-%02d-%02d" % (random.randint(1,12), random.randint(1,28))

def human_date(iso):
    y, m, d = iso.split("-")
    return "%s %s %s" % (d, MONTHS[int(m)-1], y)

def shift(iso, days):
    d = datetime.date(int(iso[:4]), int(iso[5:7]), int(iso[8:10])) - datetime.timedelta(days=days)
    return "%04d-%02d-%02d" % (d.year, d.month, d.day)

def compute_verdict(g):
    if g["po_number"] is None: return "MISSING_PO"
    if g["grn_number"] is None: return "MISSING_GRN"
    for it in g["items"]:
        if abs(float(it["inv_unit_price"]) - float(it["po_unit_price"])) > 0.005: return "PRICE_MISMATCH"
    for it in g["items"]:
        if it["grn_qty"] != it["po_qty"]: return "PARTIAL_DELIVERY"
    for it in g["items"]:
        if it["inv_qty"] != it["po_qty"]: return "QTY_MISMATCH"
    return "OK"

def render_email(g, c):
    sym = c["sym"]; gst = c["gst"]
    docs = []
    if g["po_number"] is not None:
        L = ["Dear Team,", "We have placed purchase order " + g["po_number"] + " with " + g["vendor"] + " on " + human_date(c["po_date"]) + ".", "Items ordered:"]
        for it in g["items"]:
            L.append("- " + it["material"] + ", quantity " + str(it["po_qty"]) + ", rate " + money(it["po_unit_price"], sym) + " each.")
        L.append("Regards, Purchase Department.")
        docs.append(NL.join(L))
    if g["grn_number"] is not None:
        L = ["Goods receipt note " + g["grn_number"] + " against PO " + g["po_number"] + " dated " + human_date(c["grn_date"]) + ".", "Items received:"]
        for it in g["items"]:
            L.append("- " + it["material"] + ", quantity " + str(it["grn_qty"]) + ".")
        L.append("Store Keeper.")
        docs.append(NL.join(L))
    L = ["Dear Sir, please find attached invoice " + g["invoice_number"] + " from " + g["vendor"] + " dated " + human_date(g["invoice_date"]) + ".", "Line items:"]
    for it in g["items"]:
        L.append("- " + it["material"] + ", quantity " + str(it["inv_qty"]) + ", rate " + money(it["inv_unit_price"], sym) + " each.")
    if gst:
        L.append("Total payable " + money(g["total_amount"], sym) + " (GST " + str(gst) + " percent included).")
    else:
        L.append("Total payable " + money(g["total_amount"], sym) + ".")
    docs.append(NL.join(L))
    return NL.join(docs)

def render_short(g, c):
    sym = c["sym"]; gst = c["gst"]
    docs = []
    if g["po_number"] is not None:
        s = "PO " + g["po_number"] + " | " + g["vendor"] + " | " + human_date(c["po_date"]) + " | "
        s += ", ".join("%s x%d @ %s" % (it["material"], it["po_qty"], money(it["po_unit_price"], sym)) for it in g["items"])
        docs.append(s)
    if g["grn_number"] is not None:
        s = "GRN " + g["grn_number"] + " | " + human_date(c["grn_date"]) + " | received " + ", ".join("%s x%d" % (it["material"], it["grn_qty"]) for it in g["items"])
        docs.append(s)
    s = "INV " + g["invoice_number"] + " | " + g["vendor"] + " | " + human_date(g["invoice_date"]) + " | "
    s += ", ".join("%s x%d @ %s" % (it["material"], it["inv_qty"], money(it["inv_unit_price"], sym)) for it in g["items"])
    if gst: s += " | Total " + money(g["total_amount"], sym) + " (GST incl)"
    else: s += " | Total " + money(g["total_amount"], sym)
    docs.append(s)
    return NL.join(docs)

def render_tabular(g, c):
    sym = c["sym"]; gst = c["gst"]
    docs = []
    if g["po_number"] is not None:
        L = ["-- PURCHASE ORDER --", "PO No: " + g["po_number"], "Vendor: " + g["vendor"], "Date: " + human_date(c["po_date"])]
        for n, it in enumerate(g["items"], 1):
            L.append("Item " + str(n) + ": " + it["material"] + " | Qty " + str(it["po_qty"]) + " | Rate " + money(it["po_unit_price"], sym))
        docs.append(NL.join(L))
    if g["grn_number"] is not None:
        L = ["-- GOODS RECEIPT --", "GRN No: " + g["grn_number"], "Date: " + human_date(c["grn_date"]), "Against PO: " + g["po_number"]]
        for n, it in enumerate(g["items"], 1):
            L.append("Item " + str(n) + ": " + it["material"] + " | Received Qty " + str(it["grn_qty"]))
        docs.append(NL.join(L))
    L = ["-- INVOICE --", "Invoice No: " + g["invoice_number"], "Vendor: " + g["vendor"], "Date: " + human_date(g["invoice_date"])]
    for n, it in enumerate(g["items"], 1):
        L.append("Item " + str(n) + ": " + it["material"] + " | Qty " + str(it["inv_qty"]) + " | Rate " + money(it["inv_unit_price"], sym))
    if gst: L.append("Total: " + money(g["total_amount"], sym) + " (GST " + str(gst) + " percent included)")
    else: L.append("Total: " + money(g["total_amount"], sym))
    docs.append(NL.join(L))
    return NL.join(docs)

def render_whatsapp(g, c):
    sym = c["sym"]; gst = c["gst"]
    docs = []
    if g["po_number"] is not None:
        docs.append("bhai PO bana " + g["vendor"] + " nu - " + g["po_number"] + ", " + human_date(c["po_date"]) + " - " + ", ".join("%s %d number @ %s" % (it["material"], it["po_qty"], money(it["po_unit_price"], sym)) for it in g["items"]))
    if g["grn_number"] is not None:
        docs.append("store wangu GRN " + g["grn_number"] + " ho gya " + human_date(c["grn_date"]) + " da, " + ", ".join("%s %d aa gaye" % (it["material"], it["grn_qty"]) for it in g["items"]))
    s = "vendor da invoice aa gaya - " + g["invoice_number"] + ", " + human_date(g["invoice_date"]) + " - " + ", ".join("%s %d @ %s" % (it["material"], it["inv_qty"], money(it["inv_unit_price"], sym)) for it in g["items"])
    if gst: s += ", total " + money(g["total_amount"], sym) + " (GST shamil hai)"
    else: s += ", total " + money(g["total_amount"], sym)
    docs.append(s)
    return NL.join(docs)

INSTR = "Three-way match documents (purchase order, goods receipt, invoice) are given below. Extract one JSON object with exactly these fields: vendor, po_number, grn_number, invoice_number, invoice_date (as YYYY-MM-DD), currency (INR, AED or EUR), items, total_amount. Each element of items must have: material, po_qty, grn_qty, inv_qty, po_unit_price, inv_unit_price. Use null for any document or value that is absent."

RENDER = {"email": render_email, "short": render_short, "tabular": render_tabular, "whatsapp": render_whatsapp}

def make_case(i):
    edge = EDGES[i % len(EDGES)]
    style = STYLES[i % len(STYLES)]
    vendor = VENDORS[i % len(VENDORS)]
    cur = CURRS[i % 3]
    sym = cur[1]
    gst = cur[2]
    inv_no = "INV-2026-%d" % random.randint(10000, 99999)
    po_no = "4500%d" % random.randint(100000, 999999)
    grn_no = "5000%d" % random.randint(100000, 999999)
    inv_date = rand_iso()
    c = {"sym": sym, "gst": gst, "po_date": shift(inv_date, random.randint(10, 30)), "grn_date": shift(inv_date, random.randint(1, 5))}
    n_items = random.choice([1, 1, 2, 2, 3])
    mats = random.sample(MATERIALS, n_items)
    items = []
    for mat in mats:
        po_qty = random.randint(1, 15)
        po_price = float(random.randrange(500, 90001, 50))
        inv_qty = po_qty
        inv_price = po_price
        grn_qty = po_qty
        if edge == "qty_mismatch":
            inv_qty = max(1, po_qty + random.choice([-3, -1, 1, 2, 4]))
        elif edge == "price_mismatch":
            delta = random.choice([250, 500, 1000, 2000]) if po_price >= 3000 else 100
            inv_price = float(int(po_price) + delta)
        elif edge == "partial":
            grn_qty = max(1, po_qty - random.randint(1, max(1, po_qty - 1)))
            inv_qty = grn_qty
        items.append({"material": mat, "po_qty": po_qty, "grn_qty": grn_qty, "inv_qty": inv_qty, "po_unit_price": po_price, "inv_unit_price": inv_price})
    has_po = edge != "missing_po"
    has_grn = has_po and edge != "missing_grn"
    total = sum(it["inv_qty"] * it["inv_unit_price"] for it in items)
    gold = {"vendor": vendor, "po_number": po_no if has_po else None, "grn_number": grn_no if has_grn else None, "invoice_number": inv_no, "invoice_date": inv_date, "currency": cur[0], "items": items, "total_amount": round(total, 2)}
    for it in gold["items"]:
        if not has_po:
            it["po_qty"] = None
            it["po_unit_price"] = None
        if not has_grn:
            it["grn_qty"] = None
    verdict = compute_verdict(gold)
    bundle = RENDER[style](gold, c)
    user = INSTR + NL + NL + bundle
    assistant = json.dumps(gold, indent=2)
    meta = {"case": i, "format": style, "edges": edge, "verdict": verdict}
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}], "meta": meta}

cases = [make_case(i) for i in range(100)]

# validation
TOP = {"vendor", "po_number", "grn_number", "invoice_number", "invoice_date", "currency", "items", "total_amount"}
ITEMK = {"material", "po_qty", "grn_qty", "inv_qty", "po_unit_price", "inv_unit_price"}
seen = set()
errs = []
edge_dist = {}
for cs in cases:
    g = json.loads(cs["messages"][1]["content"])
    if set(g.keys()) != TOP: errs.append(("top keys", cs["meta"]["case"]))
    for it in g["items"]:
        if set(it.keys()) != ITEMK: errs.append(("item keys", cs["meta"]["case"]))
    t = sum(it["inv_qty"] * it["inv_unit_price"] for it in g["items"] if it["inv_qty"] is not None and it["inv_unit_price"] is not None)
    if abs(g["total_amount"] - t) > 0.01: errs.append(("total", cs["meta"]["case"]))
    if compute_verdict(g) != cs["meta"]["verdict"]: errs.append(("verdict", cs["meta"]["case"]))
    h = hashlib.md5(cs["messages"][0]["content"].encode()).hexdigest()
    if h in seen: errs.append(("dup prompt", cs["meta"]["case"]))
    seen.add(h)
    edge_dist[cs["meta"]["edges"]] = edge_dist.get(cs["meta"]["edges"], 0) + 1
print("VALIDATION ERRORS:", errs)
print("EDGE DISTRIBUTION:", edge_dist)

rng2 = random.Random(SEED)
idx = list(range(100))
rng2.shuffle(idx)
test_idx = sorted(idx[:20])
train_idx = sorted(idx[20:])
test_edges = [cases[i]["meta"]["edges"] for i in test_idx]
print("TEST EDGES:", sorted(set(test_edges)), "count", len(set(test_edges)))
assert len(set(test_edges)) == 6, "test split missing an edge type"

def dump(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + chr(10))

dump(os.path.join(OUT, "match3way_golden_v1_100.jsonl"), cases)
dump(os.path.join(OUT, "match3way_train_v1_80.jsonl"), [cases[i] for i in train_idx])
dump(os.path.join(OUT, "match3way_test_v1_20.jsonl"), [cases[i] for i in test_idx])
print("WROTE 100/80/20 OK")
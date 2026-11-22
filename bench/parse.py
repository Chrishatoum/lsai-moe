import csv, glob, re, statistics as st
from pathlib import Path

def header(text):
    m = re.search(r"^BENCH (.*)$", text, re.M)
    return dict(kv.split("=", 1) for kv in m.group(1).split()) if m else None

def train(h, text):
    ms = [float(x) for x in re.findall(r"elapsed time per iteration \(ms\): ([\d.]+)", text)]
    tf = [float(x) for x in re.findall(r"throughput per GPU \(TFLOP/s/GPU\): ([\d.]+)", text)]
    if len(ms) < 15:
        return {**h, "status": "failed" if "Error" in text else "incomplete"}
    it = st.mean(ms[-5:]) / 1000
    tok = int(h["gbs"]) * int(h["seq"]) / it / int(h["gpus"])
    return {**h, "status": "ok", "iter_s": round(it, 4), "tok_s_gpu": round(tok), "tflops_gpu": round(st.mean(tf[-5:]), 1) if tf else ""}

def serve(h, text):
    grab = lambda k: (m := re.search(rf"{k}.*?:\s+([\d.]+)", text)) and float(m.group(1))
    out = grab(r"Output token throughput \(tok/s\)")
    if not out:
        return {**h, "status": "failed"}
    r = {**h, "status": "ok", "out_tok_s_gpu": round(out / int(h["gpus"]))}
    for m in ["TTFT", "TPOT", "ITL"]:
        for p in [50, 90, 99]:
            r[f"p{p}_{m.lower()}_ms"] = grab(rf"P{p} {m} \(ms\)")
    return r

rows = {"train": [], "serve": []}
for f in sorted(glob.glob("results/logs/*-*.out")):
    text = Path(f).read_text(errors="ignore")
    h = header(text)
    if h:
        rows[h["kind"]].append({"job": Path(f).stem, **(train if h["kind"] == "train" else serve)(h, text)})

for kind, rs in rows.items():
    if rs:
        keys = list(dict.fromkeys(k for r in rs for k in r))
        with open(f"results/{kind}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rs)
        print(f"results/{kind}.csv: {len(rs)} runs")

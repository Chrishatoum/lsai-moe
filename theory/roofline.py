import glob, re, sys

C = float(sys.argv[1]) * 1e12 if len(sys.argv) > 1 else 300e12  # achieved FLOP/s per GPU
NIC = 25e9                                                       # Slingshot per GPU, bytes/s
MODELS = {"g1": 8192, "g2": 4096, "g4": 2048, "g8": 1024, "g16": 512}

measured = {}
for f in glob.glob("results/logs/a2a-*.out"):
    for g, mb, bw in re.findall(r"gpus=\s*(\d+) size=\s*(\d+)MB busbw=\s*([\d.]+)GB/s", open(f).read()):
        if mb == "256":
            measured.setdefault(int(g), []).append(float(bw) * 1e9)

def bandwidth(gpus):
    if gpus in measured:
        return max(measured[gpus]), "measured"
    remote = (gpus - 4) / (gpus - 1)   # share of off-GPU traffic that leaves the node
    return NIC / remote, "estimated"

print(f"C = {C/1e12:.0f} TFLOP/s per GPU; hidden if expert width d >= d* = C / (2B)\n")
print(f"{'nodes':>5} {'GPUs':>5} {'B GB/s':>8} {'':9} {'d*':>6}   " + "  ".join(f"{m:>5}" for m in MODELS))
for n in [1, 2, 4, 8]:
    B, src = bandwidth(4 * n)
    d = C / (2 * B)
    marks = "  ".join(f"{'ok' if w >= d else 'COMM':>5}" for w in MODELS.values())
    print(f"{n:>5} {4*n:>5} {B/1e9:>8.1f} {src:9} {d:>6.0f}   {marks}")

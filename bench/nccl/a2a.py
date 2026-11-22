import os, time, torch, torch.distributed as dist

rank, world = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
torch.cuda.set_device(int(os.environ["SLURM_LOCALID"]))
dist.init_process_group("nccl", rank=rank, world_size=world)

for mb in [1, 4, 16, 64, 256]:
    x = torch.empty(mb * 2**19, dtype=torch.bfloat16, device="cuda")
    y = torch.empty_like(x)
    for _ in range(5):
        dist.all_to_all_single(y, x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        dist.all_to_all_single(y, x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / 20
    if rank == 0:
        bw = mb * 2**20 * (world - 1) / world / dt / 1e9
        print(f"gpus={world:3d} size={mb:4d}MB busbw={bw:7.1f}GB/s", flush=True)

dist.destroy_process_group()

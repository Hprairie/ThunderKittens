import mla_decode
import torch
import numpy as np
import math
import heapq
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from scheduler_v2 import backward_schedule
from scheduler import sample_schedule_generator, priority_schedule_tasks, visualize_schedule, create_arguments_from_task_schedule
from timings import save_gantt_chart
from graphviz import Digraph
from scheduler_regression import estimate_schedule_length


torch.manual_seed(0)

D_Main, D_Rot = 512, 64
PAGE_SIZE = 256
H = 16                  # H heads
NUM_PAGES = 1000        # number of pages in cache
NUM_PROCESSORS = 132    # number of processors
MAX_NUM_PAGES = 65536 // PAGE_SIZE

ENABLE_TIMINGS = True

def init_arguments(seq_lengths: List[int], NEW_TOKENS: int):

    B = len(seq_lengths)

    # Need to initialize QRot, QV, K_cache, V_cache, Lengths, Table    
    QRot    = torch.randn(B, NEW_TOKENS, H, D_Rot, dtype=torch.bfloat16, device='cuda')
    QV      = torch.randn(B, NEW_TOKENS, H, D_Main, dtype=torch.bfloat16, device='cuda')
    K_cache = torch.randn(NUM_PAGES, PAGE_SIZE, D_Rot, dtype=torch.bfloat16, device='cuda')
    V_cache = torch.randn(NUM_PAGES, PAGE_SIZE, D_Main, dtype=torch.bfloat16, device='cuda')
    Lengths = torch.tensor(seq_lengths, dtype=torch.int32, device='cuda')
    Table = torch.randint(0, NUM_PAGES, (B, MAX_NUM_PAGES), dtype=torch.int32, device='cuda')

    return QRot, QV, K_cache, V_cache, Lengths, Table

def create_thundermla_arguments(seq_lengths, NEW_TOKENS):
    # Processor assignment heuristic: assign processors proportionally to sequence lengths.
    t0 = time.time()
    processor_assignments = [math.floor(s / sum(seq_lengths) * NUM_PROCESSORS) for s in seq_lengths]
    while sum(processor_assignments) < NUM_PROCESSORS:
        min_idx = processor_assignments.index(max(processor_assignments))
        processor_assignments[min_idx] += 1
    processor_assignments = sorted([(estimate_schedule_length(p, NEW_TOKENS, s), p, s, i) for i, (p, s) in enumerate(zip(processor_assignments, seq_lengths))])
    while len(seq_lengths) > 1:
        best, worst = processor_assignments[0], processor_assignments[-1]
        new_t0, new_tn1 = estimate_schedule_length(best[1]-1, NEW_TOKENS, best[2]), estimate_schedule_length(worst[1]+1, NEW_TOKENS, worst[2])
        new_time = max(new_t0, new_tn1)
        if new_time < worst[0]:
            processor_assignments[0] = (new_t0, best[1]-1, best[2], best[-1])
            processor_assignments[-1] = (new_tn1, worst[1]+1, worst[2], worst[-1])
            processor_assignments = sorted(processor_assignments)
        else:
            break
    num_processors = [None for _ in seq_lengths]
    for _, p, s, i in processor_assignments:
        num_processors[i] = min(p, s//128)
    # Create schedule
    start_processors = [sum(num_processors[:i]) for i in range(len(num_processors))]
    scheduled_tasks = []
    partial_uid, reduction_uid = 0, NUM_PROCESSORS
    for batch_id, (seq_l, start_p, num_p) in enumerate(zip(seq_lengths, start_processors, num_processors)):
        new_tasks, partial_uid, reduction_uid = backward_schedule(
            list(range(start_p, start_p + num_p)), batch_id, seq_l, list(range(NEW_TOKENS)), partial_uid, reduction_uid
        )
        scheduled_tasks.extend(new_tasks)
    t1 = time.time()
    print(f'Time taken to create schedule: {(t1-t0)*1000} ms')
    Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = create_arguments_from_task_schedule(
        scheduled_tasks, NEW_TOKENS, num_processors=NUM_PROCESSORS, enable_timings=ENABLE_TIMINGS
    )
    # visualize_schedule(scheduled_tasks, NUM_PROCESSORS)
    return Instructions, O_scratch, Lvec_scratch, Semaphore, Timings

def run_thundermla(QRot, QV, K_cache, V_cache, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings, tic=None):
    if tic is None:
        Semaphore.zero_()
        tic = 1
    O = torch.zeros_like(QV)
    Q_all = torch.concat([QV, QRot], dim=-1).contiguous()
    KV_all = torch.cat([V_cache, K_cache], dim=-1).contiguous()
    softmax_scale = 1.0 / math.sqrt(D_Main+D_Rot)
    torch.cuda.synchronize()
    if Timings is not None:
        mla_decode.mla_decode(Instructions, QRot, QV, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, tic, Timings)
        mla_decode.mla_decode(Instructions, QRot, QV, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, 1-tic, Timings)
    else:
        mla_decode.mla_decode(Instructions, QRot, QV, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, tic)
        mla_decode.mla_decode(Instructions, QRot, QV, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, 1-tic)
    torch.cuda.synchronize()
    return O, Timings

def profile_thundermla(QRot, QV, K_cache, V_cache, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings, ITERS=1000):
    Semaphore.zero_()
    O = torch.zeros_like(QV)
    softmax_scale = 1.0 / math.sqrt(D_Main+D_Rot)
    torch.cuda.synchronize()
    t0 = time.time()
    for it in range(1,ITERS+1):
        if Timings is not None:
            mla_decode.mla_decode(Instructions, QRot, QV, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, it%2, Timings)
        else:
            mla_decode.mla_decode(Instructions, QRot, QV, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, it%2)
    torch.cuda.synchronize()
    t1 = time.time()
    return (t1-t0) / ITERS

def run_mla_torch(QRot, QV, K_cache, V_cache, Lengths, Table):
    Q = torch.concat([QRot, QV], dim=-1)
    full_K = torch.cat([K_cache, V_cache], dim=-1)[Table].reshape(Q.shape[0], -1, Q.shape[-1])
    full_V = V_cache[Table].reshape(Q.shape[0], -1, QV.shape[-1])
    softmax_scale = 1.0 / math.sqrt(D_Main+D_Rot)
    O = torch.zeros_like(QV)
    for b, l in enumerate(Lengths):
        # assert Q.shape[1] == 1, "Q must have shape (B, 1, H, D) for the time being."
        mask = torch.ones(Q.shape[1], l, dtype=torch.bool).tril(diagonal=l-Q.shape[1]).to(Q.device)
        O[b:b+1] = torch.nn.functional.scaled_dot_product_attention(
            Q[b:b+1].transpose(1, 2),
            full_K[b:b+1, :l].unsqueeze(-2).repeat((1,1,H,1)).transpose(1, 2),
            full_V[b:b+1, :l].unsqueeze(-2).repeat((1,1,H,1)).transpose(1, 2),
            is_causal=False,
            attn_mask=mask,
            scale=softmax_scale
        ).transpose(1, 2)
    return O

def main():
    seq_lengths=sorted([4641,45118,1730,1696])
    NEW_TOKENS = 4
    QRot, QV, K_cache, V_cache, Lengths, Table = init_arguments(seq_lengths, NEW_TOKENS)
    ref = run_mla_torch(QRot, QV, K_cache, V_cache, Lengths, Table)
    Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = create_thundermla_arguments(seq_lengths, NEW_TOKENS)
    O, Timings = run_thundermla(QRot, QV, K_cache, V_cache, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings)
    print("ref mean:", torch.mean(ref.abs()))
    print("Kernel output mean", torch.mean(O.abs()))
    print("Max absolute diff", torch.max(torch.abs(O - ref)))
    print("Avg absolute diff", torch.mean(torch.abs(O - ref)))

    time_per_iter = profile_thundermla(QRot, QV, K_cache, V_cache, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings)
    print(f"Time per iter: {time_per_iter*1000} ms")

    save_gantt_chart(Timings, Instructions, name='new')

if __name__ == "__main__":
    main()
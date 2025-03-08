import mha_decode
import torch
import numpy as np
import math
import heapq
import time
import random

from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
from scheduler_v2 import backward_schedule
from scheduler import sample_schedule_generator, priority_schedule_tasks, visualize_schedule, create_arguments_from_task_schedule
from timings import save_gantt_chart
from pprint import pprint
from scheduler_regression import estimate_schedule_length

torch.manual_seed(0)

# Configuration parameters
HEAD_DIM  = 128
PAGE_SIZE = 256
H = 16                # Number of heads
NUM_PAGES      = 1000 # Number of pages in cache
NUM_PROCESSORS = 132  # Number of processors
MAX_NUM_PAGES = 65536 // PAGE_SIZE
ENABLE_TIMINGS = True

def init_arguments(seq_lengths: List[int], NEW_TOKENS: int):
    """Initialize Q, K_cache, V_cache, Lengths and Table."""
    B = len(seq_lengths)
    Q       = torch.randn(B, NEW_TOKENS, H, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    K_cache = torch.randn(NUM_PAGES, PAGE_SIZE, H, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    V_cache = torch.randn(NUM_PAGES, PAGE_SIZE, H, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    
    Lengths = torch.tensor(seq_lengths, dtype=torch.int32, device='cuda')
    Table   = torch.randint(0, NUM_PAGES, (B, MAX_NUM_PAGES), dtype=torch.int32, device='cuda')
    return Q, K_cache, V_cache, Lengths, Table

def create_thundermha_arguments(seq_lengths, new_tokens, num_heads):
    """Creates a schedule and associated arguments for the thunder mha decode."""
    seq_head_lengths = sorted([(s, h, b) for b, s in enumerate(seq_lengths) for h in range(num_heads)])
    print("Number of (seq, head) pairs:", len(seq_head_lengths))
    t0 = time.time()
    
    # Initially assign processors per (seq, head)
    processor_assignments = [
        max(math.floor(s / (sum(seq_lengths)*num_heads) * NUM_PROCESSORS), 1)
        for s in seq_lengths for _ in range(num_heads)
    ]
    
    # Adjust so that the sum equals NUM_PROCESSORS
    while sum(processor_assignments) < NUM_PROCESSORS:
        min_idx = processor_assignments.index(max(processor_assignments))
        processor_assignments[min_idx] += 1
    while sum(processor_assignments) > NUM_PROCESSORS:
        min_idx = processor_assignments.index(max(processor_assignments))
        processor_assignments[min_idx] -= 1

    # Convert to tuple: (estimated schedule length, processor count, (seq_length, head, batch), index)
    processor_assignments = sorted([
        (estimate_schedule_length(p, new_tokens, shb[0]), p, shb, i)
        for i, (p, shb) in enumerate(zip(processor_assignments, seq_head_lengths))
    ])

    # Balance processor assignments further using the schedule-length estimator
    while len(seq_head_lengths) > 1:
        best, worst = processor_assignments[0], processor_assignments[-1]
        if best[1]-1 == 0:
            break
        new_t0 = estimate_schedule_length(best[1]-1, new_tokens, best[2][0])
        new_tn1 = estimate_schedule_length(worst[1]+1, new_tokens, worst[2][0])
        new_time = max(new_t0, new_tn1)
        if new_time < worst[0]:
            processor_assignments[0]  = (new_t0, best[1]-1, best[2], best[-1])
            processor_assignments[-1] = (new_tn1, worst[1]+1, worst[2], worst[-1])
            processor_assignments = sorted(processor_assignments)
        else:
            break

    # Determine number of processors per (seq, head) for scheduling
    num_processors_list = [None for _ in seq_head_lengths]
    for _, p, (s, h, b), i in processor_assignments:
        # print(f"Processor assignment for (seq_length={s}, head={h}, batch={b}): {p} (s//128 = {s//128})")
        num_processors_list[i] = max(min(p, s//128), 1)
    
    # Create schedule using backward_schedule for each (seq, head)
    start_processors = [sum(num_processors_list[:i]) for i in range(len(num_processors_list))]
    print(start_processors, num_processors_list)
    scheduled_tasks = []
    partial_uid, reduction_uid = 0, NUM_PROCESSORS
    for (seq_l, h, b), start_p, num_p in zip(seq_head_lengths, start_processors, num_processors_list):
        new_tasks, partial_uid, reduction_uid = backward_schedule(
            list(range(start_p, start_p + num_p)), b, h, seq_l, list(range(new_tokens)), partial_uid, reduction_uid
        )
        scheduled_tasks.extend(new_tasks)
    t1 = time.time()
    print(f"Time taken to create schedule: {(t1-t0)*1000:.3f} ms")
    
    Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = create_arguments_from_task_schedule(
        scheduled_tasks, new_tokens, num_processors=NUM_PROCESSORS, num_heads=num_heads, enable_timings=ENABLE_TIMINGS
    )
    # Optionally, visualize the schedule:
    # visualize_schedule(scheduled_tasks, NUM_PROCESSORS)
    return Instructions, O_scratch, Lvec_scratch, Semaphore, Timings

def run_thundermha(Q, K_cache, V_cache, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings, tic=None):
    """Runs a single call to the thunder mha decode."""
    if tic is None:
        Semaphore.zero_()
        tic = 1
    O = torch.zeros_like(Q)
    softmax_scale = 1.0 / math.sqrt(HEAD_DIM)
    
    torch.cuda.synchronize()
    mha_decode.mha_decode(Instructions, Q, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, tic, Timings)
    torch.cuda.synchronize()
    return O

def profile_thundermha(Q, K_cache, V_cache, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings, ITERS=10):
    """Profiles the thunder mha decode kernel over a number of iterations."""
    Semaphore.zero_()
    O = torch.zeros_like(Q)
    softmax_scale = 1.0 / math.sqrt(HEAD_DIM)
    
    # Warm-up call
    mha_decode.mha_decode(Instructions, Q, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, 1, Timings)
    torch.cuda.synchronize()
    t0 = time.time()
    for it in range(ITERS):
        mha_decode.mha_decode(Instructions, Q, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, it % 2, Timings)
    torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) / ITERS

def run_benchmark_tk(seq_lengths: List[int], new_tokens: int, iterations: int = 10):
    """
    Runs a benchmark on tk mha decoding with the provided sequence lengths (for the cache) and new_tokens (query length).
    This mirrors the flash-attn benchmark configuration.
    """
    print(f"\n----------- starting seq_lengths: {seq_lengths} new_tokens: {new_tokens} -----------")
    torch.manual_seed(0)
    
    # Initialize arguments
    Q, K_cache, V_cache, Lengths, Table = init_arguments(seq_lengths, new_tokens)
    Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = create_thundermha_arguments(seq_lengths, new_tokens, H)
    
    # Profile the thunder mha decode kernel
    avg_time = profile_thundermha(Q, K_cache, V_cache, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings, ITERS=iterations)
    print(f"Profiling: Average time per iteration = {avg_time*1e6:.1f} µs over {iterations} iterations")
    
    # save_gantt_chart(Timings, Instructions)
    
    # Compute memory I/O and FLOPS (using similar estimates as in flash-attn)
    total_length = sum(seq_lengths)
    # Memory I/O in bytes: for each token in the cache, H heads, each with key and value (each HEAD_DIM elements) at 2 bytes per element.
    mem_io = total_length * H * (HEAD_DIM + HEAD_DIM) * 2
    # FLOPS: for each query token and each cached token, H heads, (HEAD_DIM + 2*HEAD_DIM) operations per element, times 2 (fused multiply-add)
    flops = new_tokens * total_length * H * (HEAD_DIM + 2*HEAD_DIM) * 2
    print(f"Time: {avg_time*1e6:.1f} µs, {(mem_io/1e9) / avg_time:.0f} GB/s, {(flops/1e12) / avg_time:.0f} TFLOPS/s")

if __name__ == "__main__":
    # Run benchmarks with the same configurations as the flash-attn benchmark:
    # run_benchmark_tk([4641, 45118, 1730, 1696], 4)
    # run_benchmark_tk([4641, 45118, 1730, 1696], 2)
    # run_benchmark_tk([4641, 45118, 1730, 1696], 1)
    run_benchmark_tk([65536], 1)
    run_benchmark_tk([65536], 2)
    run_benchmark_tk([65536], 4)
    run_benchmark_tk([2048, 2048, 2048, 2048], 1)
    run_benchmark_tk([2048, 2048, 2048, 2048], 2)
    run_benchmark_tk([2048, 2048, 2048, 2048], 4)

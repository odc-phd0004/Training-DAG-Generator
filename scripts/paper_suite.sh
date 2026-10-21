#!/usr/bin/env bash
# A suite of realistic job configurations for a multi-tenant cluster study.
# Each line = one training job captured on THIS single GPU as one logical
# rank per pipeline stage of an N-GPU job.  Comment out what you don't need.
#
# Memory: in fake mode a process holds exactly what one rank of the real job
# holds.  If a job would fit on an 80 GB H100, it fits on a 96 GB RTX PRO 6000.
# On a 48 GB card, add --layers K (truncate depth) or lower local_batch_size.
#
#   bash scripts/paper_suite.sh            # runs everything below
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
J() { echo; echo "######## $1"; bash "$HERE/scripts/capture_job.sh" "$@" || echo "!!! $1 failed (see traces/$1/capture.log)"; }

# --- Llama-3.1 8B ------------------------------------------------------------
# pure FSDP (ZeRO-3) on 8 GPUs
J llama3_8b_fsdp8            8 -- --module llama3 --config llama3_8b
# FSDP on 32 GPUs (4 nodes)
J llama3_8b_fsdp32          32 -- --module llama3 --config llama3_8b
# 2D: TP=2 (intra-node) x FSDP=8
J llama3_8b_tp2_fsdp8       16 -- --module llama3 --config llama3_8b \
    --parallelism.tensor_parallel_degree 2
# HSDP: replicate 2 x shard 8
J llama3_8b_hsdp2x8         16 -- --module llama3 --config llama3_8b \
    --parallelism.data_parallel_replicate_degree 2 --parallelism.data_parallel_shard_degree 8
# 3D: PP=2 (1F1B, 8 microbatches) x TP=2 x FSDP=4
J llama3_8b_pp2_tp2_fsdp4   16 -- --module llama3 --config llama3_8b \
    --training.local_batch_size 8 \
    --parallelism.pipeline_parallel_degree 2 --parallelism.pipeline_parallel_schedule 1F1B \
    --parallelism.pipeline_parallel_microbatch_size 1 --parallelism.tensor_parallel_degree 2
# 3D with interleaved schedule (2 virtual stages per rank)
J llama3_8b_pp4i_tp2_fsdp2  16 -- --module llama3 --config llama3_8b \
    --training.local_batch_size 8 \
    --parallelism.pipeline_parallel_degree 4 --parallelism.pipeline_parallel_schedule Interleaved1F1B \
    --parallelism.pipeline_parallel_microbatch_size 1 --parallelism.tensor_parallel_degree 2

# --- Llama-3.1 70B -------------------------------------------------------------
# TP=8 x FSDP=8 on 64 GPUs (TorchTitan's reference config); smaller batch for 48/96 GB
J llama3_70b_tp8_fsdp8      64 -- --module llama3 --config llama3_70b \
    --training.local_batch_size 2
# PP=4 x TP=8 x FSDP=2 on 64 GPUs
J llama3_70b_pp4_tp8_fsdp2  64 -- --module llama3 --config llama3_70b \
    --training.local_batch_size 8 \
    --parallelism.pipeline_parallel_degree 4 --parallelism.pipeline_parallel_schedule 1F1B \
    --parallelism.pipeline_parallel_microbatch_size 1

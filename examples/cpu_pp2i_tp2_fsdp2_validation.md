# dagtrace validation: /home/claude/runs/e2e_cpu

**14 passed, 0 failed, 4 informational**

| check | rank | result | detail |
|---|---|---|---|
| stability | 0 | PASS | structure hash per step: {2: '2281b7bc30d5', 3: '2281b7bc30d5'} |
| coverage.groups | 0 | PASS | 0 collectives without process group; 0 unmatched API tags |
| coverage.kernels | 0 | info | time source cpu_ops; unattributed GPU time 0.000% |
| pp_p2p.size | 0 | PASS | 24 P2P ops, expected size 65536 B each (mbs 1 x seq 256/tp x dim 256 x 2 B); mismatching sizes: [] |
| pp_p2p.count | 0 | PASS | sends 12, recvs 12, expected 12 each (4 microbatches, stages [0, 2] of 4) |
| tp_msg.size | 0 | PASS | 152/152 TP all-gather/reduce-scatter ops (100.0% of TP volume) have the SP activation size 131072 B = 1x256x256x2; 3.17 per layer-microbatch (4 fwd+bwd per layer without recompute; more with activation checkpointing) |
| fsdp_volume.reduce_scatter | 0 | PASS | RS volume 6166528 B = 1.0000 x (local params 770816 x dp_shard 2 x 4 B); exactly one gradient sync per step |
| fsdp_volume.all_gather | 0 | info | AG volume 3083264 B = 1.000 x one parameter pass (770816 x 2 x 2 B); ~2 = forward + backward re-gather (reshard_after_forward) |
| stability | 4 | PASS | structure hash per step: {2: '9f071fd3f182', 3: '9f071fd3f182'} |
| coverage.groups | 4 | PASS | 0 collectives without process group; 0 unmatched API tags |
| coverage.kernels | 4 | info | time source cpu_ops; unattributed GPU time 0.000% |
| pp_p2p.size | 4 | PASS | 24 P2P ops, expected size 65536 B each (mbs 1 x seq 256/tp x dim 256 x 2 B); mismatching sizes: [] |
| pp_p2p.count | 4 | PASS | sends 12, recvs 12, expected 12 each (4 microbatches, stages [1, 3] of 4) |
| tp_msg.size | 4 | PASS | 152/152 TP all-gather/reduce-scatter ops (100.0% of TP volume) have the SP activation size 131072 B = 1x256x256x2; 3.17 per layer-microbatch (4 fwd+bwd per layer without recompute; more with activation checkpointing) |
| fsdp_volume.reduce_scatter | 4 | PASS | RS volume 9316352 B = 1.5105 x (local params 770944 x dp_shard 2 x 4 B); every gradient is synced at least once; >1 means some FSDP groups reduce-scatter per microbatch (framework behaviour, e.g. PP grad accumulation) |
| fsdp_volume.all_gather | 4 | info | AG volume 7282176 B = 2.361 x one parameter pass (770944 x 2 x 2 B); ~2 = forward + backward re-gather (reshard_after_forward) |
| params |  | PASS | sum of captured stages' parameters 6,163,712 vs analytical 6,163,712 |
| global.consistency |  | PASS | 916 collective instances, 96 P2P pairs across 8 ranks; 0 problems  |

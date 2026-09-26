# HM-router: factored model-harness routing

An agent's performance depends on both its model and its harness (the
scaffold that manages tool use and execution), and the best model often changes
from one harness to another, so routing should pick the pair rather than the
model alone. HM-router routes each query jointly over (model, harness) pairs:
it learns model and harness representations that are shared across routes. 
Because the representations are shared, evidence from observed
pairs carries over to combinations that have little or no execution history.

This repo contains the HM-router implementation, the **HM-router-bench**
dataset (12 public agent benchmarks, 293 routes), two baselines (Avengers,
Random) and a script that reproduces the main results table.

## Project structure

```
data/
  hm_router_bench.csv        the benchmark: raw acc per (benchmark, query, route)
  queries.csv                query texts per (benchmark, query_id)
src/
  data.py                    benchmark loading, query embeddings, train/test splits
  hm_router.py               HM-router: scorer, training loop, harness summaries
  baselines.py               Random and Avengers
  run.py                     trains + evaluates all methods, prints the results table
```

### The benchmark

`data/hm_router_bench.csv` has 12 agentic benchmarks, 1,648 queries and
293 routes. A route is a (model, harness) pair, and every route covers every
query of its benchmark.

| column | meaning |
|---|---|
| `benchmark` | benchmark name |
| `query_id` | task id (text in `data/queries.csv`) |
| `model` | model name |
| `harness` | harness (agent scaffold) name as shown on the source leaderboard |
| `scaffold` | canonical harness key shared across benchmarks (e.g. `claudecode`) |
| `acc` | raw accuracy of the route on the query, in [0, 1]: the success rate over trials when a route was run more than once, and the rubric score on workspace_bench_lite |

```python
import pandas as pd
df = pd.read_csv("data/hm_router_bench.csv", dtype={"query_id": str})
acc = df[df.benchmark == "gaia"].pivot(index="query_id", columns=["model", "harness"], values="acc")
```

## How to run

```bash
pip install -r requirements.txt
python -m src.run                            # all 12 benchmarks x 5 seeds
python -m src.run --benchmarks gaia scicode  # a subset
python -m src.run --raw                      # raw accuracy instead of oracle-normalized
```


## Results (oracle-normalized, mean over 5 seeds)

| benchmark | Random | Avengers | HM-router |
|---|---:|---:|---:|
| terminal_bench_2.1 | 0.7729 | 0.8332 | **0.8689** |
| taubench_airline | 0.3566 | 0.5152 | **0.5752** |
| corebench_hard | 0.2684 | 0.6693 | **0.9133** |
| scienceagentbench | 0.3028 | 0.4156 | **0.5647** |
| gaia | 0.4229 | 0.6999 | **0.7387** |
| online_mind2web | 0.3971 | 0.4695 | **0.5126** |
| swebench_verified | 0.6165 | 0.8612 | **0.8822** |
| harness_bench | 0.7971 | 0.8295 | **0.8933** |
| corebench_v1.1 | 0.8660 | 0.8833 | **0.9333** |
| scicode | 0.2364 | 0.5962 | **0.6503** |
| skillsbench | 0.4010 | 0.5600 | **0.6623** |
| workspace_bench_lite | 0.5124 | 0.6601 | **0.6769** |
| **mean** | 0.4958 | 0.6661 | **0.7393** |

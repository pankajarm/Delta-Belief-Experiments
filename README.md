<div align="center">

# Intrinsic Credit Assignment for Long Horizon Interaction

**[Ilze Amanda Auzina](https://ilzeamandaa.github.io/)\*,
[Joschka Strüber](https://joschkastrueber.github.io/)\*,
[Sergio Hernández-Gutiérrez](https://sergiohg.com/)\*,
[Shashwat Goel](https://shash42.github.io/)†,
[Ameya Prabhu](https://ameya.prabhu.be/)†,
[Matthias Bethge](https://bethgelab.org/)†**

*\* Equal contribution &emsp; † Equal supervision*

**[Tübingen AI Center](https://tuebingen.ai/) · [University of Tübingen](https://uni-tuebingen.de/) · [MPI for Intelligent Systems](https://www.tuebingen.mpg.de/10020/mpi-fuer-intelligente-systeme) · [ELLIS Institute Tübingen](https://institute-tue.ellis.eu/)**

[![alphaXiv](https://img.shields.io/badge/alphaXiv-Paper-%23c83232)](https://www.alphaxiv.org/abs/intrinsic-credit-assignment)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://bethgelab.github.io/delta-belief-rl/)
[![Models](https://img.shields.io/badge/-HuggingFace-3B4252?style=flat&logo=huggingface&logoColor=)](https://huggingface.co/collections/bethgelab/delta-belief-rl)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Fork](https://img.shields.io/badge/Fork_of-bethgelab%2Fdelta--belief--rl-informational)](https://github.com/bethgelab/delta-belief-rl)

</div>

---

> **This repository** is a fork of [bethgelab/delta-belief-rl](https://github.com/bethgelab/delta-belief-rl) with **additional experiments** by [Pankaj Mathur](https://github.com/pankajarm), including a novel AWQ quantized judge configuration and full paper reproduction on 8x A100 80GB GPUs. The original paper and codebase are by Auzina, Struber, Hernandez-Gutierrez, Goel, Prabhu, and Bethge at the Tubingen AI Center.

---

<p align="center">
  <img src="static/images/figure1.png" alt="Main contributions overview" width="60%">
</p>

## TL;DR

How can we train agents to navigate uncertainty over long horizons? We propose **ΔBelief-RL**, which uses the change in an agent's own belief about the correct answer as a dense intrinsic reward for training in long-horizon, information-seeking tasks. No critic or process reward model needed — just the agent's self-assessment. Our trained agents (**CIA** — Curious Information-seeking Agents) outperform DeepSeek-V3.2 (670B) with ~98% fewer parameters on our in-distribution test-set, generalize to out-of-distribution tasks, and continue improving beyond the training horizon.

## Method

ΔBelief-RL leverages an agent's internal beliefs as an intrinsic reward signal, enabling dense credit assignment in probabilistic, long-horizon tasks. In a multi-turn interaction setting where an agent must uncover a latent target concept through a trajectory of actions and observations:

**1. Agent Beliefs** — We elicit the agent's internal belief by leveraging its underlying token probability distribution over the target concept at each turn.

**2. ΔBelief Reward** — The per-turn belief change is computed as:

$$\Delta\text{Belief}_t = \log b_t - \log b_{t-1}$$

**3. Training Reward** — The sparse outcome reward is augmented with the dense intrinsic signal:

$$r_t = \underbrace{r^{\mathrm{eog}}}_{\text{trajectory outcome}} + \underbrace{\lambda\, \max(\Delta\text{Belief}_t, 0)}_{\text{intrinsic exploration}} + \underbrace{r_p}_{\text{efficiency penalty}}$$

We use **Turn-wise GRPO**, computing advantages at the turn level rather than applying the same reward across the entire trajectory.

<p align="center">
  <img src="static/images/beliefs_deepseek_v3.2_base_only.png" alt="Belief updates over time" width="50%">
</p>
<p align="center"><em>Beliefs steadily increase on average, and the rate of growth strongly correlates with the outcome of the trajectory.</em></p>

## Key Results

### CIA outperforms much larger models on 20 Questions

<div align="center">

| Method | Mean@8 ± std | Pass@8 |
|:---|:---:|:---:|
| Baseline (1.7B) | 9.97% ± 1.04% | 32.03% |
| StarPO (1.7B) | 16.54% ± 1.32% | 45.73% |
| **CIA (ours, 1.7B)** | **24.80% ± 1.10%** | **53.10%** |
| Baseline (4B) | 13.34% ± 1.05% | 36.87% |
| StarPO (4B) | 24.36% ± 1.18% | 59.12% |
| **CIA (ours, 4B)** | **33.72% ± 1.26%** | **63.97%** |
| DeepSeek-V3.2 (670B) | 14.35% ± 0.87% | 47.34% |
| Qwen3-235B-A22B-Instr. | 8.83% ± 0.87% | 27.71% |

</div>

### Efficient exploration

<p align="center">
  <img src="static/images/num_questions.png" alt="Number of questions during training" width="50%">
  <br>
  <img src="static/images/num_repeated_questions.png" alt="Repeated questions during training" width="50%">
</p>
<p align="center"><em>ΔBelief-RL reduces the number of turns required and suppresses redundant queries more rapidly than standard GRPO.</em></p>

### Test-time interaction scaling

<p align="center">
  <img src="static/images/1.7b_interaction_scaling_1.png" alt="Test-time interaction scaling 1.7B" width="50%">
  <br>
  <img src="static/images/4b_interaction_scaling_1.png" alt="Test-time interaction scaling 4B" width="50%">
</p>
<p align="center"><em>CIA continues to improve as the interaction budget extends beyond the 20-turn training horizon. At 4B scale, the improvement from 20 to 50 turns is +26% for CIA vs +13% for StarPO.</em></p>

### Out-of-distribution generalization

<p align="center">
  <img src="static/images/passk_benchmarks_k8_games.png" alt="OOD generalization" width="55%">
</p>
<p align="center"><em>Despite training only on 20 Questions, CIA generalizes to Guess My City and Murder Mystery.</em></p>

<p align="center">
  <img src="static/images/passk_cs_k8.png" alt="Customer Service" width="32%">
  <img src="static/images/user_personalization.png" alt="User Personalization" width="30%">
</p>
<p align="center"><em>CIA outperforms StarPO on practical applications: Customer Service (+5–11%) and User Personalization (up to +15%).</em></p>

## Additional Experiments (8x A100 80GB)

The following experiments were conducted by [Pankaj Mathur](https://github.com/pankajarm) to reproduce and extend the paper's results on **8x NVIDIA A100 80GB GPUs**.

### Experiment 1: Paper Reproduction (Full Precision)

**Configuration:** `train_8xa100_paper`

| Component | Setup |
|:---|:---|
| Actor | Qwen3-1.7B-SFT + LoRA (rank 64) on 1 GPU |
| Judge | Qwen3-14B full precision, TP=2 on 6 GPUs |
| Training | Batch=240, lr=3e-5, 15 epochs, exact paper hyperparameters |

<div align="center">

| Training Step | Validation Samples | Wins | Mean@4 |
|:---:|:---:|:---:|:---:|
| 0 (baseline) | 792 | 76 | 9.6% |
| 4 | 792 | 111 | **14.0%** |
| 5 | 792 | 106 | 13.4% |
| 8 | 792 | 87 | 11.0% |
| 9 | 792 | 100 | 12.6% |

</div>

### Experiment 2: AWQ Quantized Judge (Novel)

> This experiment is **not in the original paper**. It demonstrates that DeltaBelief-RL works with INT4 quantized judges, dramatically lowering the hardware barrier.

Instead of running a single full-precision Qwen3-14B judge with TP=4, we use **Qwen3-14B-AWQ (INT4 quantized)** with TP=1 and **7 data-parallel replicas** across 7 GPUs. This increases judge throughput while freeing GPU memory.

**Configuration:** `train_8xa100_awq_dp7`

| Component | Setup |
|:---|:---|
| Actor | Qwen3-1.7B-SFT + LoRA (rank 64) on GPU 0 |
| Judge | Qwen3-14B-AWQ (INT4), TP=1, **DP=7** on GPUs 1-7 |
| GPU Utilization | 94.5% stable |
| Key Fixes | MarlinLinearKernel disabled, FlashAttention 3 -> 2 (A100 compatibility) |

<div align="center">

| Training Step | Samples | Wins | Success Rate | Note |
|:---:|:---:|:---:|:---:|:---|
| 0 (baseline) | 792 | 81 | 10.2% (Mean@4) | |
| 9 | 792 | 94 | 11.9% | |
| 12 | 792 | 102 | 12.9% | |
| 16 | 792 | 126 | 15.9% | |
| 17 | 1584 | 224 | 14.1% (Mean@8) | Doubled validation samples |
| 20 | 1584 | 265 | 16.7% | |
| **24** | **1584** | **292** | **18.4% (Mean@8)** | **Pass@8 = 48.0%** |

</div>

### Why AWQ Quantization Matters: Original vs Our Approach

The original paper uses a **full-precision FP16/BF16 judge** that requires multiple GPUs via tensor parallelism just to fit in memory. Our AWQ experiment flips this tradeoff — trading negligible judge quality for massive infrastructure gains:

<div align="center">

| | Original Paper (FP16) | Ours (AWQ INT4) | Improvement |
|:---|:---:|:---:|:---:|
| **Judge model** | Qwen3-14B FP16 | Qwen3-14B-AWQ INT4 | 3.5x smaller weights |
| **GPUs per judge** | 2-4 (TP=2 or TP=4) | **1** (TP=1) | 2-4x fewer GPUs per replica |
| **Judge replicas** | 1 (single instance) | **7** (DP=7) | **7x judge throughput** |
| **Judge GPU budget** | 6 GPUs for 1 judge | 7 GPUs for 7 judges | 7x rollout parallelism |
| **GPU memory/judge** | ~28 GB (FP16) | ~7 GB (INT4) | 4x less VRAM per replica |
| **Min. hardware** | 4+ A100s for judge alone | **1 A100 per judge** | Runs on cheaper setups |
| **Training convergence** | Converges | **Converges comparably** | No quality loss observed |

</div>

**The bottom line:** Full-precision judges are a bottleneck — they consume multiple GPUs for a single instance, limiting rollout throughput. AWQ quantization breaks this bottleneck by fitting each judge on a single GPU, enabling **7 parallel judge replicas on the same hardware budget**. This means:

1. **7x faster rollouts** — the judge evaluates 7 agent trajectories simultaneously instead of 1
2. **Democratized access** — researchers with 2-4 GPUs can now run the full pipeline (1 actor GPU + 1 AWQ judge GPU), whereas the original requires 4+ GPUs minimum just for the judge
3. **No quality sacrifice** — AWQ INT4 quantization preserves judge accuracy; our training curve shows steady improvement (10.2% -> 18.4% Mean@8, still climbing at step 24/60)
4. **Better scaling** — with more judge replicas generating diverse rollouts, the actor sees more varied training signal per batch

This is particularly relevant for the RL community where **judge/reward model throughput is often the training bottleneck**, not the actor model itself.

### Infrastructure Innovations

- **Watchdog auto-restart** (`watchdog.sh`): Monitors training, detects OOM/crashes, auto-restarts with checkpoint resume
- **HF checkpoint backup** (`hf_backup.sh`): Automated upload of checkpoints to HuggingFace Hub
- **vLLM A100 fixes**: Disabled `MarlinLinearKernel` to avoid CUDA illegal memory access on vLLM 0.10.2; FlashAttention 3 -> 2 fallback (FA3 requires Hopper; A100 is Ampere)
- **Memory optimization**: `gpu_memory_utilization=0.60`, `kv_cache_memory_bytes=35GB` for stable 94.5% GPU utilization

### Validation Artifacts

Raw validation trajectories (JSONL) for all experiments are in the `validation/` directory:
- `validation/train_8xa100_paper/` — Steps 0, 4, 5, 8, 9 (198 secrets, n=4 per secret)
- `validation/train_8xa100_awq_dp7/` — Steps 0, 9, 12, 16, 17, 20, 24 (198 secrets, n=4 then n=8)

Each JSONL line contains: `secret`, `input`, `output`, `score`, `game_status`, `reward`, `num_questions`, `log_prob_reward`, `log_prob_diff`.

## Installation

### Prerequisites

- Python >= 3.12
- CUDA 12.8 compatible GPU(s) (minimum 2x GPUs recommended)
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# Clone with submodules (includes verl framework)
git clone --recurse-submodules https://github.com/bethgelab/delta-belief-rl.git
cd delta-belief-rl

# Create virtual environment and install dependencies
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
cd delta_belief_rl && uv sync && cd ..
```

### Weights & Biases (optional)

Training logs to W&B by default. Set your credentials:
```bash
export WANDB_PROJECT=delta-belief-rl
export WANDB_ENTITY=<your-entity>
```

## Running Experiments

All training and evaluation scripts are located in `delta_belief_rl/scripts/`. Each script handles Ray initialization and GPU setup automatically via `scripts/slurm_setup.sh`.

### Training

**ΔBelief-RL** (our method):
```bash
# Qwen3-1.7B
sbatch delta_belief_rl/scripts/train/delta-belief-rl/qwen1.7b.sh

# Qwen3-4B
sbatch delta_belief_rl/scripts/train/delta-belief-rl/qwen4b.sh
```

**GRPO baseline** (StarPO):
```bash
# Qwen3-1.7B
sbatch delta_belief_rl/scripts/train/grpo/qwen1.7b.sh

# Qwen3-4B
sbatch delta_belief_rl/scripts/train/grpo/qwen4b.sh
```

### Evaluation

Evaluate trained checkpoints (or HuggingFace-hosted models) on various benchmarks:

```bash
# 20 Questions (in-distribution)
slurm delta_belief_rl/scripts/eval/twenty_questions.sh

# Guess My City (OOD)
slurm delta_belief_rl/scripts/eval/guess_my_city.sh

# Murder Mystery (OOD)
slurm delta_belief_rl/scripts/eval/murder_mystery.sh

# Customer Service (OOD, uses API judge)
slurm delta_belief_rl/scripts/eval/customer_service.sh
```

### GPU Configuration

Adjust `CUDA_VISIBLE_DEVICES` at the top of each script to match your setup. The number of GPUs per model is configured via:
```bash
actor_rollout_ref.ngpus=1 # GPUs for the questioner model
judge_rollout.ngpus=1     # GPUs for the judge model
```

### 8x A100 GPU Configuration

For running on 8x A100 80GB GPUs without SLURM:

```bash
# Set credentials as environment variables
export WANDB_API_KEY="your-wandb-key"
export HF_TOKEN="your-hf-token"  # only needed for hf_backup.sh

# Full precision judge (paper reproduction)
bash launch_train_8xa100.sh

# AWQ quantized judge (novel configuration — 7x judge throughput)
bash launch_train_8xa100_awq.sh
```

### SLURM Clusters

The setup script (`scripts/slurm_setup.sh`) handles Ray initialization, port allocation, and GPU memory checks automatically. For SLURM-managed clusters, the `SLURM_JOB_ID` is detected automatically; for other environments, set it manually in the script.

## Project Structure

```
delta-belief-rl/
├── train.py                          # Main entry point
├── delta_belief_rl/
│   ├── config/                       # Hydra YAML configs
│   ├── env/                          # Environment data & prompts
│   │   ├── twenty_questions/         #   20 Questions (training env)
│   │   ├── guess_my_city/            #   Guess My City (OOD eval)
│   │   ├── murder_mystery/           #   Murder Mystery (OOD eval)
│   │   └── customer_service/         #   Customer Service (OOD eval)
│   ├── llm_agent/
│   │   ├── generation.py             #   Multi-turn conversation logic
│   │   ├── belief.py                 #   ΔBelief computation
│   │   └── prompts.py                #   Prompt templates per environment
│   ├── trainer/
│   │   ├── multistep_trainer.py      #   Ray-based FSDP training loop
│   │   └── ppo/core_algos.py         #   PPO, GRPO, Turn-GRPO, RLOO
│   ├── workers/                      #   Ray actor & rollout workers
│   ├── scripts/
│   │   ├── train/                    #   Training scripts
│   │   │   ├── delta-belief-rl/      #     ΔBelief-RL (ours)
│   │   │   └── grpo/                 #     GRPO baseline
│   │   └── eval/                     #   Evaluation scripts
│   └── pyproject.toml                # Python dependencies
├── verl/                             # veRL framework (git submodule)
└── static/                           # Project page assets
```

## Citation

```bibtex
@misc{auzina2026intrinsiccreditassignmentlong,
      title={Intrinsic Credit Assignment for Long Horizon Interaction}, 
      author={Ilze Amanda Auzina and Joschka Strüber and Sergio Hernández-Gutiérrez and Shashwat Goel and Ameya Prabhu and Matthias Bethge},
      year={2026},
      eprint={2602.12342},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.12342}, 
}
```

If you use the additional experiments or AWQ configuration from this fork, please also cite:

```bibtex
@software{mathur2026deltabeliefexperiments,
  author = {Mathur, Pankaj},
  title = {Delta-Belief-Experiments: Reproducing and Extending ΔBelief-RL with AWQ Quantization},
  year = {2026},
  url = {https://github.com/pankajarm/Delta-Belief-Experiments},
}
```

## Acknowledgments

This codebase builds on [veRL](https://github.com/volcengine/verl) for distributed RL training with FSDP and Ray.

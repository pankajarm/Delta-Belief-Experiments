# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeltaBelief-RL implements **intrinsic credit assignment for long-horizon interaction** (arXiv 2602.12342). It trains agents ("CIA" — Curious Information-seeking Agents) that use changes in their internal belief about the correct answer as a dense intrinsic reward for multi-turn, information-seeking tasks (primarily 20 Questions). Built on top of the [veRL](https://github.com/volcengine/verl) distributed RL framework (included as a git submodule).

## Setup & Installation

Requires Python >= 3.12, CUDA 12.8 GPUs (minimum 2), and `uv` package manager.

```bash
git clone --recurse-submodules https://github.com/bethgelab/delta-belief-rl.git
cd delta-belief-rl
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
cd delta_belief_rl && uv sync && cd ..
```

The `verl/` submodule must be initialized (`git submodule update --init --recursive`).

## Running Experiments

Single entry point: `python3 -m train <hydra overrides>`. Requires `RAY_CLIENT_ADDRESS` env var.

Training and eval scripts are in `delta_belief_rl/scripts/`. They are launched via SLURM (`sbatch`) and handle Ray initialization via `scripts/slurm_setup.sh`:

```bash
# Training (DeltaBelief-RL)
sbatch delta_belief_rl/scripts/train/delta-belief-rl/qwen1.7b.sh
sbatch delta_belief_rl/scripts/train/delta-belief-rl/qwen4b.sh

# Training (GRPO baseline)
sbatch delta_belief_rl/scripts/train/grpo/qwen1.7b.sh

# Evaluation
sbatch delta_belief_rl/scripts/eval/twenty_questions.sh
sbatch delta_belief_rl/scripts/eval/guess_my_city.sh
```

GPU allocation is configured per script via `actor_rollout_ref.ngpus` and `judge_rollout.ngpus` Hydra overrides.

## Architecture

### Config System

Hydra + OmegaConf. Primary config: `delta_belief_rl/config/base_multiturn.yaml`. All training parameters are overridden via Hydra CLI args in the shell scripts.

### Core Modules

- **`train.py`** — Entry point. Sets up `RewardManager`, Ray workers, `ResourcePoolManager`, and `RayMultiStepTrainer`. Routes reward computation to environment-specific functions based on `data_source`.

- **`delta_belief_rl/llm_agent/`** — Agent logic:
  - `generation.py` — `LLMGenerationManager`: multi-turn conversation loop, input formatting, logprob reward computation, judge verification. Largest file (~2400 lines).
  - `belief.py` — `BeliefManager`: extends `LLMGenerationManager` for belief estimation (logprob delta computation).
  - `prompts.py` — All prompt templates for all environments and the judge.
  - `tensor_helper.py` — `TensorHelper`: padding, masking, position ID utilities.

- **`delta_belief_rl/trainer/`** — Training infrastructure:
  - `multistep_trainer.py` — `RayMultiStepTrainer`: Ray-based FSDP training loop, validation, checkpointing, metrics.
  - `ppo/core_algos.py` — Advantage estimators: `compute_grpo_turn_advantage` (Turn-GRPO) and `compute_multi_turn_reinforce`. Uses a registry pattern (`@register_advantage_estimator`).
  - `metrics_utils.py` — Metric computation (pass@k, success rates).

- **`delta_belief_rl/workers/`** — Distributed GPU workers:
  - `fsdp_workers.py` — `ActorRolloutRefWorker`, `JudgeRolloutWorker`: FSDP-based training/inference with LoRA adapter support.
  - `api_workers.py` — `APIJudgeRolloutWorker`, `APIActorRolloutWorker`: external API models (e.g., OpenRouter) for eval.
  - `rollout/vllm_rollout/` — vLLM rollout engine integration.

- **`delta_belief_rl/env/`** — Environments with reward functions and data:
  - `twenty_questions/` — Training environment. `reward.py` defines `TQRewardSignal` enum and `traj_reward_fn`.
  - `guess_my_city/`, `murder_mystery/`, `customer_service/` — OOD evaluation environments, each with their own `reward.py`.

- **`delta_belief_rl/utils/`** — Utilities:
  - `format.py` — DataProto padding/unpadding, `MTAlgoConfig` dataclass.
  - `syntax.py` — Text normalization, judge response parsing, answer extraction.
  - `lora_adapters.py` — LoRA adapter management with FSDP context managers.
  - `dataset/rl_dataset.py` — `RLHFDataset` for loading parquet data.

### Key Algorithms

- **DeltaBelief reward**: At each turn, compute agent's log-probability of the ground truth. The delta between consecutive turns is the intrinsic reward signal.
- **Turn-wise GRPO** (`grpo_turn`): Advantages computed and normalized per-turn rather than per-trajectory. Defined in `core_algos.py`.
- **Supported estimators**: `grpo`, `grpo_turn`, `reinforce_plus_plus`, `multi_turn_reinforce`, `reinforce_plus_plus_baseline`.

### Key Technologies

| Component | Technology |
|---|---|
| Distributed training | FSDP (via veRL) + Ray |
| Inference | vLLM 0.10.2 |
| Config | Hydra + OmegaConf |
| Fine-tuning | LoRA (PEFT) |
| Logging | Weights & Biases |
| Job scheduling | SLURM |
| Base models | Qwen3 family (1.7B/4B actor, 14B judge) |

## Conventions

- No test suite, linter, or CI/CD currently configured.
- Git commit messages follow `[Tag] description` pattern (e.g., `[Add]`, `[Ref]`, `[Fix]`).
- The `verl/` submodule is upstream code — avoid modifying it directly.

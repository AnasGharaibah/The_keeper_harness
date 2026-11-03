# ARC-AGI-3: Daniel Franzen's Milestone 2 solution

This repository contains my solution for the **ARC-AGI-3 Progress Prize #2**. **For a detailed explanation of the approach, changes, and experiments, see the [write-up](WRITEUP.md).**

My approach is based on [Tufa Labs' Duck harness](https://github.com/Tufalabs/duck-harness). Full credit to **Jeroen Cottaar and Tufa Labs** for the original solver, harness, and notebook.

My changes focus on getting more useful work from the model within the competition's compute budget: faster inference with a larger context, better prefix-cache retention, adaptive compute allocation across games, and improvements to the information and tools available to the agent.

## Reproducing the solution

The [competition notebook](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution) is the intended reproduction path. It includes the harness patch and sets up the model and offline serving runtime on Kaggle. When copying it, select the **RTX Pro 6000** GPU manually.

The fork is based on upstream commit [`7652836`](https://github.com/Tufalabs/duck-harness/tree/7652836056c59e044f093e3c13ed7438c814169e). The source here includes both the features used in the submission and optional experiments. The published notebook's settings identify the competition configuration; enabling every option is not the intended setup.

For local use and development, see the [harness README](ARC3-Inference/README.md). The checked-in local configuration and Tufa deployment commands are separate from the competition notebook's serving setup.

For building the offline SGLang runtime and details of the serving patches, see the [serving README](serving/README.md).

## Main changes

- **Model and serving:** Qwen3.8-Flash-Next with an optimized SGLang runtime. Serving work makes long contexts practical while maintaining throughput, including changes to Mamba prefix-cache retention and checkpoint prefetching.
- **Context management:** calibrated text-token estimates, image-size-aware token accounting, and blockwise history trimming. Evicting a larger block at once lets the retained prefix be reused for more requests before the next trim.
- **Compute allocation:** bounded concurrent admission reduces cache pressure. Priority scheduling allocates compute using current progress, actions and tokens spent, and continuation value. Cached continuations receive priority; tail fade shifts emphasis toward immediate level completions near the deadline.
- **Perception and feedback:** larger board images, board-difference and death images, animation frames and timelines accessible in Python, and clearer action-result history.
- **Agent tools and prompts:** executable undo, optional reset, guidance for transferring knowledge across levels, and retention of eligible Python functions and supported imports.
- **Execution safeguards:** guards for stale-state actions and no-op batches, protection at level boundaries, and clearer tool output with action echo and optional middle truncation.

- **Makefile compatibility:** fixes for newer GNU Make versions, including how configuration-derived variables are evaluated.
- **Viewer fixes and improvements:** correct per-turn transcript display, visibility of retried attempts, clearer handling of pending transcripts and the latest board state, and per-step token-usage information.

The detailed option reference is available in the [configuration guide](ARC3-Inference/CONFIGURATION.md). The [write-up](WRITEUP.md) distinguishes features used in the competition from experiments with negative or inconclusive results.

## Repository layout

| Path | Contents |
| --- | --- |
| [ARC3-Inference/](ARC3-Inference/) | Modified solver, prompts, tools, scheduler, diagnostics, and viewer. |
| [serving/](serving/) | Offline SGLang wheelhouse builder, custom patches, and build instructions. |
| [tufa-arc-agi-framework/](tufa-arc-agi-framework/) | Tufa's bundled game execution framework. |
| [taaf-duck-harness-kaggle-share.ipynb](taaf-duck-harness-kaggle-share.ipynb) | Tufa's original bundled notebook; see the competition notebook above for this solution. |

## Inspect a saved run

The viewer does not require a model server or GPU. From the repository root, with `uv` installed:

```bash
cd ARC3-Inference
uv sync --locked
make view VIEW_RUN_DIR=/path/to/your/run VIEW_PORT=8011
```

Open `http://127.0.0.1:8011` to inspect the saved run's boards, actions, and model transcripts.

## Acknowledgements

This work builds on Tufa's [Duck harness](https://github.com/Tufalabs/duck-harness) and [technical write-up](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133), the Qwen models, and the SGLang serving framework.

The serving setup also builds on work by [John Pezzulli (Pennyroyal)](https://github.com/jpezzulli/sglang-rtxpro6000), [Mamy Ratsimbazafy](https://github.com/mratsim/sglang-qwen38fn-sm120-turbo), and [Gabriel Olympie](https://github.com/gabrielolympie/sglang-flashnext-sm120), with model quantizations from RadixArk and Intel. Exact checkpoints, runtime versions, and patches are recorded in the competition notebook and its build scripts.

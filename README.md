# Keeper: falsification-gated memory for ARC-AGI-3 agents

**Team PSUT MindX**: Anas Gharaibah, Tuleen Bakir, Omar Abdel Rahem
Princess Sumaya University for Technology (PSUT), Amman, Jordan

Our entry for the Kaggle competition [ARC Prize 2026: ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3) and the [ARC Prize 2026 Paper Track](https://www.kaggle.com/competitions/arc-prize-2026-paper-track). It was our first Kaggle competition.

**Best public LB score: 32.56** (submission 57716708, notebook [`f3-s9-keeper-v6`](https://www.kaggle.com/code/anasgharaibah/f3-s9-keeper-v6)). Every notebook in the results table below is public on Kaggle.

> **Branches.** This is `main`: our earlier Keeper v1/v2 work on Tufa Labs' Duck harness (its old README is [docs/KEEPER_V2_README.md](docs/KEEPER_V2_README.md)). The code behind the 32.56 score, and every file linked below, is on the [`keeper-v6`](https://github.com/AnasGharaibah/The_keeper_harness/tree/keeper-v6) branch.

---

## What Keeper is

ARC-AGI-3 games come with no instructions: the agent has to work out the rules and the goal by playing. A language-model agent forms beliefs about the game ("the blue block pushes the red one", "the goal is to fill the frame"), and wrong beliefs stay in its context and mislead it level after level.

**Keeper lets the agent keep a belief only after the belief has survived a test against the moves it actually played.** It is a set of optional layers on top of [Daniel Franzen's Milestone 2 solution](https://github.com/da-fr/arc-agi-3-solution), which is built on [Tufa Labs' Duck harness](https://github.com/Tufalabs/duck-harness). Every layer is an environment switch, and with all of them off the code behaves like Franzen's baseline.

| Switch | File | What it does |
| --- | --- | --- | --- |
| `KEEPER_SURPRISE` | `keeper_auto.py` | Tells the model when a move did something it did not expect. |
| `KEEPER_PERCEPTION` | `keeper_perception.py` | Reports object roles, rotations and shape matches that a plain frame diff misses. |
| `KEEPER_REVIEW` | `keeper_review.py`, `keeper_check_sandbox.py` | After each level (and after a game over), one model call proposes rules and goals as small Python checks. A sandbox runs each check on the recorded moves. Only checks that pass become pinned facts, shown to the model as `KEEPER NOTES`. |
| fail-safe | `keeper_guard.py`, `keeper_resources.py` | If a Keeper hook crashes, times out or uses too much memory, Keeper switches itself off for that game. The run itself never dies. |
| run deadline | notebook | A soft deadline at 8h40 makes sure the hidden rerun always finishes inside Kaggle's time limit. |

All Keeper code is in [`ARC3-Inference/inference/agent/keeper_*.py`](https://github.com/AnasGharaibah/The_keeper_harness/blob/keeper-v6/ARC3-Inference/inference/agent/), and its tests are in `ARC3-Inference/tests/test_keeper_*.py`.

```
 game frame ──► agent (Qwen, Python tool) ──► action ──► game
                    ▲                            │
       KEEPER NOTES │                            ▼
   (passed checks)  │                   recorded moves
                    │                            │
               fact store ◄── sandbox test ◄── review call proposes rules
                              (fail = dropped)
```

## Results

The public leaderboard scores runs on hidden games. Each team can submit once per day.

| Submission | What | Public LB | Notebook |
| --- | --- | --- | --- |
| 57716708 | **F3**: Keeper v6 + S9 layers | **32.56** | [f3-s9-keeper-v6](https://www.kaggle.com/code/anasgharaibah/f3-s9-keeper-v6) v1 |
| 57892002 | F3 rerun 2 (same notebook) | 31.39 | [f3-s9-keeper-v6](https://www.kaggle.com/code/anasgharaibah/f3-s9-keeper-v6) v2 |
| 57565176 | F2: Franzen + D' scheduler, Keeper off | 29.04 | [f2-franzen-gateway-dprime](https://www.kaggle.com/code/tuleenbakir/f2-franzen-gateway-dprime) |
| 57460716 | Keeper v3 notebook, Keeper off | 28.74 | [keeper-v3-a-baseline](https://www.kaggle.com/code/anasgharaibah/keeper-v3-a-baseline) |
| 57786009 | F3.1: F3 on Keeper v6.1 | 26.93 | [f3-1-keeper-v6-1](https://www.kaggle.com/code/anasgharaibah/f3-1-keeper-v6-1) |
| 57676562 | F2K2-D: Keeper v5 + run deadline | 26.32 | [f2k2-d-rerun-deadline](https://www.kaggle.com/code/anasgharaibah/f2k2-d-rerun-deadline) |
| 57751962 | **F3-off**: the F3 notebook with every `KEEPER_*` switch off | 25.63 | [f3-off-s9-no-keeper](https://www.kaggle.com/code/anasgharaibah/f3-off-s9-no-keeper) |
| 57854361 | F3 rerun 1 (same notebook) | 24.91 | [f3-s9-keeper-v6](https://www.kaggle.com/code/anasgharaibah/f3-s9-keeper-v6) v2 |
| 57823468 | F3-guard: F3 + no-op repeat guard | 24.17 | [f3-guard](https://www.kaggle.com/code/anasgharaibah/f3-guard) |
| 57692322 | F2K3: Keeper v6 without S9 | 20.91 | [f2k3-keeper-v6](https://www.kaggle.com/code/anasgharaibah/f2k3-keeper-v6) |
| 57415945 | Keeper v2 (Duck harness) | 1.11 | [keeper-v2-20261020](https://www.kaggle.com/code/anasgharaibah/keeper-v2-20261020) |
| 56853156 | Omar's hybrid council search | 0.11 | Omar's account |
| 57501713, 57579040, 57610610 | Keeper v3 (d), v4, F2K2: overran the hidden time limit | 0.00 | [keeper-v3-d-all](https://www.kaggle.com/code/anasgharaibah/keeper-v3-d-all), [f2k-franzen-gateway-dprime-keeper](https://www.kaggle.com/code/tuleenbakir/f2k-franzen-gateway-dprime-keeper), [f2k2-franzen-gateway-dprime-keeper](https://www.kaggle.com/code/anasgharaibah/f2k2-franzen-gateway-dprime-keeper) |

**What we learned from these numbers:**

- **Ablation.** The same notebook scored 32.56 with Keeper on and 25.63 with Keeper off. That is a one-run comparison, so it is a hint, not proof.
- **Noise is large.** The F3 notebook scored 32.56, 24.91 and 31.39 on three identical submissions (mean 29.6). On Franzen's own notebook, others see about 27.6-31.5 from run to run. One submission cannot show a 1-3 point difference.
- **Our 0.00 runs** came from Keeper slowing the agent until the hidden rerun ran out of time. The 8h40 soft deadline fixed this, and every later Keeper run finished.
- **Short checks do not predict LB scores.** F3's 20-minute check scored 1.62.

## Reproduce

The intended path is the Kaggle notebook **[f3-s9-keeper-v6](https://www.kaggle.com/code/anasgharaibah/f3-s9-keeper-v6)**. It installs the keeper-v6 harness patch, adds the S9 layers, and sets the Keeper switches. Run it on an **RTX Pro 6000** GPU.

To run the Keeper unit tests locally:

```bash
cd ARC3-Inference
python -m pytest tests/test_keeper_*.py
```

For local runs of the harness, see the [harness README](https://github.com/AnasGharaibah/The_keeper_harness/blob/keeper-v6/ARC3-Inference/README.md) and the [configuration guide](https://github.com/AnasGharaibah/The_keeper_harness/blob/keeper-v6/ARC3-Inference/CONFIGURATION.md). Franzen's original README is kept as [docs/FRANZEN_README.md](https://github.com/AnasGharaibah/The_keeper_harness/blob/keeper-v6/docs/FRANZEN_README.md).

## Paper

Our Paper Track write-up, *Keeper: falsification-gated memory for ARC-AGI-3 agents*, will be linked here when it is published (due 9 November 2026). [docs/KEEPER_REPORT.md](https://github.com/AnasGharaibah/The_keeper_harness/blob/keeper-v6/docs/KEEPER_REPORT.md) is our longer internal report (written 22 Oct, at Keeper v3).

## Credits

Keeper (the `keeper_*` modules, their tests and the notebook wiring) is our work. Everything else belongs to its original authors:

- **Daniel Franzen**: the [Milestone 2 solution](https://github.com/da-fr/arc-agi-3-solution) this repository forks ([write-up](https://github.com/AnasGharaibah/The_keeper_harness/blob/keeper-v6/WRITEUP.md)).
- **Jeroen Cottaar and Tufa Labs**: the [Duck harness](https://github.com/Tufalabs/duck-harness) and TAAF.
- **ondrewsha**: the HF fixes, RESETC, H3v2.1 audit and H1 solved-level memory from the public notebook [ARC-AGI-3 stack S9](https://www.kaggle.com/code/ondrewsha/arc-agi-3-stack-s9), used verbatim in F3.
- **shiiin9** ([Affectify](https://www.kaggle.com/code/shiiin9/affectify-arc-31-54-in-a-single-sub)) for the D' scheduler formula and **tantan0327** ([arc3-franzen-dprime](https://www.kaggle.com/code/tantan0327/arc3-franzen-dprime)) for its implementation.
- **Qwen team**: the language model.

## License

[Apache License 2.0](LICENSE), the same as the upstream code. Third-party code and model weights keep their own licenses.

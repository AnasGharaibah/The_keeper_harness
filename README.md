# The Keeper Harness

The Keeper is a fork of **The Duck** 🦆, Tufa Labs' tool-using solver for
[ARC-AGI-3](https://arcprize.org) games, which runs on TAAF (the Tufa ARC-AGI
Framework). It keeps the Duck's behavior unchanged and adds two optional layers
that help a small model learn what is actually true about a game:

1. **Prediction check**: the model says what it expects a move to do, and the
   harness tells it mechanically whether it was right.
2. **Level review**: the model records rules and goals it believes; after each
   level the harness tests them against the recorded moves. Only claims that pass
   become **verified facts**, which stay pinned in the model's context across
   levels.

Both layers are switched from the config file and are **off by default**, so the
original Duck can be run as a baseline for A/B comparison.

> **Status:** the layer code and its unit tests are written, but they have not
> been run yet. See [Verify on a new machine](#verify-on-a-new-machine).

## Why

Two problems motivated the layers:

- The Duck sees `board_changed` and before/after frames, but nothing makes it
  commit to a prediction, so it rarely notices when its idea of a move is wrong.
- Every Python tool call starts with a fresh sandbox, and old messages are
  dropped when the context fills up. What the model learned on level 1 is often
  gone by level 3.

The prediction check addresses the first. The level review keeps learning
across levels, but only what the harness has checked, never the model's opinion.

## Repository layout

| Path | What it is |
| --- | --- |
| `ARC3-Inference/` | The harness: solver, prompts, Python sandbox, run artifacts, scoring, viewer, and the two new layers. |
| `ARC3-Inference/inference/agent/prediction.py` | Layer 1: compares an expectation with the real move. |
| `ARC3-Inference/inference/agent/level_memory.py` | Layer 2: hypotheses, review, verified facts. |
| `ARC3-Inference/inference/agent/layer_config.py` | On/off switches and limits for both layers. |
| `ARC3-Inference/configs/` | `inference.json` (baseline), `inference.layer1.json` (layer 1 only), `inference.layers.json` (both layers). |
| `ARC3-Inference/tests/` | Unit tests for the layers. |
| `tufa-arc-agi-framework/` | TAAF: the `Benchmark` / `GameAPI` framework the harness runs on. Unchanged. |
| `taaf-duck-harness-kaggle-share.ipynb` | The original Kaggle notebook. The layers are not wired into it. |

The upstream repository also ships a 2.8 GB `example-run/`. It is not included
here.

## Branches

- `main`: the untouched Duck snapshot this fork starts from.
- `duck-layers`: the two layers, configs, and tests.

## Quick start

You need Python 3.12 and [`uv`](https://docs.astral.sh/uv/), plus a running
vLLM server (OpenAI-compatible) serving the model, e.g. `Qwen3.6-27B-FP8`.

```bash
git clone -b duck-layers https://github.com/AnasGharaibah/The_keeper_harness.git
cd The_keeper_harness/ARC3-Inference
make install
```

Play one game with both layers on, against a vLLM server on another machine:

```bash
LOCAL_ANALYZER_BASE_URL=http://<VLLM_HOST>:<PORT>/v1 \
LOCAL_ANALYZER_API_KEY=<key, if the server needs one> \
EXPERIMENTS_DIR=./runs \
make interactive GAME=<game> N_PASSES=1 MAX_RUNTIME_MINUTES=10 \
  CONFIG_PATH=configs/inference.layers.json
```

If the server's model name differs from the config, also pass
`MODEL=<served name> LOCAL_ANALYZER_MODEL_ID=<served name>`.

Open the viewer on your runs:

```bash
make view VIEW_RUN_DIR=runs/<run-name>
```

## A/B comparison

Run the same games and passes with each config, then compare the scores with the
existing `make score_run` and `make significance` tools
(see [`ARC3-Inference/README.md`](ARC3-Inference/README.md)).

| Arm | Config |
| --- | --- |
| Baseline (original Duck) | `configs/inference.json` |
| Layer 1 only | `configs/inference.layer1.json` |
| Layers 1 + 2 | `configs/inference.layers.json` |

With both layers off, the prompts, tool description, and tool results are the
same as the original Duck's.

## Layer 1: prediction check

Inside its Python tool, the model can attach an expectation to any action:

```python
action(['RIGHT'], expect={'moves': {3: [0, 1]}, 'board_changed': True})
action('MOUSE', row=12, col=40, expect={'cells': {(12, 40): 'R'}})
```

The old call forms (`action(['UP'])`, `action([{'action': 'MOUSE', 'row': 4, 'col': 7}])`)
work as before.

| `expect` key | Checked against |
| --- | --- |
| `board_changed`, `level_completed`, `game_over` | The real result flags. |
| `moves` | `{object_id_or_hash: [d_row, d_col]}`: where that object is after the move. Ids and hashes are the ones in `current_frame.segmentation`. |
| `appears` | A color character or object hash that should gain a new object. Free-text items are stored, not checked. |
| `disappears` | An object id or hash that should lose an object. |
| `cells` | `{(row, col): color_char}` on the board after the move. |
| `note` | Free text; stored only. |

After the move runs, the harness (not the model) compares the expectation with
the real before/after boards. The tool result gets a `prediction_check` block:

```
#12 RIGHT -> confirmed (moves,board_changed)
#13 UP -> WRONG: moves[id 3]: moved [0,0], expected [-1,0]
```

Long loops are summarized as `N confirmed / M wrong / K unchecked` plus the first
few wrong diffs. Inside the sandbox the same results are available as
`last_prediction` and `prediction_log`. Wrong moves are marked in the viewer.

## Layer 2: level review and verified facts

The model records claims with a short, testable check:

```python
hypothesis(
    "RIGHT moves the blue cell one column right",
    kind="rule",              # or "goal"
    check_code="""
def check_transition(t):
    if parse_action(t.action)[0] != 'RIGHT':
        return None           # does not apply to this move
    old = [c for r, c, a, b in changed_cells(t.before_frame, t.after_frame) if a == 'b']
    new = [c for r, c, a, b in changed_cells(t.before_frame, t.after_frame) if b == 'b']
    return len(old) == 1 and len(new) == 1 and new[0] == old[0] + 1
""",
)
```

`check_transition(t)` returns `True` (held), `False` (violated), or `None` (not
applicable). Check code can use `cell(frame, row, col)`, `changed_cells(a, b)`
and `parse_action(text)`. A quick check runs right away on the current level's
moves, with its own short time limit.

When a level is completed, on game over, and at the end of the run, the harness
reviews the claims:

1. **Re-check facts** from earlier levels on the new moves. A fact that fails is
   moved to REFUTED, and the model is told.
2. **Run every open check** in the same restricted sandbox (same import
   allowlist and time limit) against all recorded moves of the level. A claim
   becomes a **fact** only if all of these hold:
   - it never returns `False` on a real move;
   - it applied (`True`) to at least `min_support` moves;
   - it returns a clean `False` on at least one move the harness altered (board
     frozen, action swapped, or result flags flipped). A check that passes on
     everything is marked *unverified (unfalsifiable)*. A crash on an altered
     move does not count.

   A model-written check that raises an error on real moves refutes the claim; a broken check written by the review LLM is discarded instead.
3. **One optional LLM call** per level, in a separate short conversation. It
   writes check code for claims that have none (the harness then runs that
   code) and lists claims the evidence contradicts. It can refute a claim that
   has no passing check, but it can never promote a fact or override a passing
   check. If it times out, only the mechanical results are kept.

Facts and the most recent refuted claims are pinned in the system message, which
is never evicted when the context fills up:

```
VERIFIED FACTS (harness-checked against recorded transitions of this game; ...):
- H1 [rule] RIGHT moves the blue cell one column right (learned L1, support 14)
REFUTED (do not rely on these):
- H3 [goal] reach the red square -- check failed on level 2 at step 41 (UP)
```

Everything is also saved to `artifacts/<game>_p<N>_facts.json` in the run
directory.

## Configuration

Both layers live under `analyzer.*` in the strict-JSON config (no comments).
The Makefile turns each key into a `LOCAL_ANALYZER_*` environment variable.

```json
"prediction_check": {"enabled": true, "max_wrong_diffs_shown": 5, "sandbox_log_size": 20},
"level_review": {
  "enabled": true, "llm_review": true, "max_review_tokens": 2048,
  "max_llm_calls_per_level": 1, "review_thinking": false,
  "llm_timeout_seconds": 60, "review_time_budget_seconds": 90,
  "max_facts_in_context": 15, "max_refuted_in_context": 5, "max_block_chars": 2500,
  "min_support": 2, "perturbation_test": true,
  "check_timeout_seconds": 10, "quick_check_seconds": 3,
  "max_review_transitions": 300, "max_hypotheses": 60
}
```

| Key | Meaning |
| --- | --- |
| `max_wrong_diffs_shown` | Wrong diffs shown in a long-loop summary. |
| `sandbox_log_size` | Entries of `prediction_log` exposed in the sandbox. |
| `llm_review` | Allow the one review LLM call per level. |
| `max_review_tokens`, `review_thinking`, `llm_timeout_seconds` | Limits for that call. |
| `review_time_budget_seconds` | Total time for one review; also capped by the game's remaining time. |
| `max_facts_in_context`, `max_refuted_in_context`, `max_block_chars` | Size of the pinned block; highest-support facts are kept. |
| `min_support` | Moves a check must apply to before it can become a fact. |
| `perturbation_test` | Require a check to reject at least one altered move. |
| `check_timeout_seconds` | Time limit for each check during a review. |
| `quick_check_seconds` | Total time for quick checks inside one tool call. |
| `max_review_transitions` | Most recent moves of a level that a review looks at. |
| `max_hypotheses` | Cap on recorded claims per game run. |

## Verify on a new machine

Nothing in the layers has been executed yet. Before trusting a run:

```bash
cd ARC3-Inference
make install
uv run --no-sync pytest tests/test_sandbox_layers.py -x -q   # most fragile part first
make prepare-ci                                              # ruff + all tests
```

Then run the one-game smoke test from [Quick start](#quick-start) and check:

- the transcript shows `prediction_check` lines and a `[LEVEL REVIEW]` section;
- `runs/<run>/artifacts/*_facts.json` exists;
- the viewer shows a "Prediction" badge on steps where `expect=` was used.

## Known limits

- The layers are not wired into the Kaggle notebook, so they are off on Kaggle.
- `moves` matches objects by shape and color. An object that changes shape when
  it moves shows up as "not found after the move".
- On games with a timer bar, `board_changed` is almost always `True`.
- A review runs at the start of the first turn on the next level, which adds a
  little latency there. Its tokens count toward the game's token total.

## Credits

The Duck harness and TAAF are by [Tufa Labs](https://tufalabs.ai). See their
[technical write-up](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133)
and [blog post](https://tufalabs.ai/research/duck-harness/). The harness's own
documentation is in [`ARC3-Inference/README.md`](ARC3-Inference/README.md).

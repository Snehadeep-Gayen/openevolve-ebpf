# Idea-First Evolution Loop

This mini-framework evolves **ideas** and then generates/evaluates programs for each idea. It stays additive to OpenEvolve—no MAP-Elites/islands needed—and reuses:

- `rag_codebase/eval_agent/idea_agent.py` for idea refinement
- `openevolve/evaluator.py` for program evaluation
- Simple embedding similarity checks (like OpenEvolve) to avoid duplicate ideas

## Quick Start

```bash
export OPENAI_API_KEY=...        # required for idea/gen/reasoning calls
python -m new_evolve_framework.run \
  examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py \
  --config examples/function_minimization/config.yaml \
  --iterations 2 \
  --programs-per-idea 2 \
  --model gpt-5.1 \
  --temperature 0.8 \
  --results tmp_idea_loop
```

The `rag_codebase/eval_agent/evaluation_agent.py` Q&A loop always runs before ideation. The resulting `eval_output.json` is mirrored into each `iteration_<n>/` directory.

Artifacts land under `--results` as `iteration_<n>/` plus a top-level `index.yaml` containing all nodes/programs and a `nodes_clean.yaml` with just the ideas/eval context/best programs. Each iteration also captures the exact program-generation prompt/response under `program_generations/gen_<k>.yaml`.

Each iteration also stores the dynamically generated LLM payloads (`eval_prompt.yaml`, `idea_prompt.yaml`) that embed the current best program metrics, reasoning, and evaluation summary/hypothesis used to drive the agents.

## How It Works (hacky overview)

1. **Bootstrap**: evaluate the initial program once; store as the first `ProgramRecord` and create a bootstrap `IdeaNode` with no design.
2. **Idea generation**: pick the best node (highest score), run `idea_agent.run_idea_refinement(...)` seeded with that node’s best program metrics/reasoning to get a new idea payload.
3. **Similarity check**: embed the idea summary; if it looks too close to an existing node, log it and continue anyway (threshold in config).
4. **Program generation**: for the new idea, generate N programs sequentially. Each generation prompt includes the idea payload and previous attempts so candidates learn from recent failures/successes.
5. **Evaluation + reasoning**: each program is evaluated with `Evaluator`; on success, an LLM reasoning pass summarizes metrics/runs. Failed runs get a couple of compile-fix retries via a simple LLM prompt.
6. **Selection**: choose the best program by `combined_score` (or average numeric metrics), update the node, and save snapshots plus YAML indexes.

Artifacts: we normalize evaluator artifacts similar to OpenEvolve’s worker flow—pull `summary`, `runs`, and create an `evolve_debug` directory under any returned `debug_path`.

## Prompts

Default prompts live in `new_evolve_framework/prompts/`:

- `eval_agent_prompt_base.txt`: base system/user text for evaluation Q&A
- `idea_agent_prompt_base.txt`: base system/user text for idea refinement
- `program_generation.txt`: builds full programs from an idea + prior attempts
- `compile_fix.txt`: repairs non-compiling/running code

You can edit/replace them and point the engine at custom paths if needed.

## Notes & Limitations

- Deterministic best-node selection for now (no sampling); tweak later if desired.
- Similarity uses OpenAI embeddings (`text-embedding-3-small` by default) on idea summaries.
- Evaluation retries for compile/run are minimal; adjust `compile_fix_attempts` inside `IdeaProgramEngine` if you want more.
- If evaluator emits a `debug_path`, we create an `evolve_debug` subdir there and copy any parent `exp-*` directory into the iteration folder for easier inspection.
- Failed attempts are now recorded (with error text and artifacts) and included in subsequent prompts so the model can learn from build/run issues.
- `idea_output.json` and `eval_output.json` (written by the RAG agents) are copied into each iteration directory when present.

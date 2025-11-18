# OpenEvolve Controller Deep Dive

This guide dissects the `OpenEvolve` orchestrator at `openevolve/controller.py` so you can debug, extend, or replace any part of the runtime lifecycle.

---

## Class Overview

- **File:** `openevolve/controller.py`
- **Primary class:** `OpenEvolve` (`openevolve/controller.py:60`)
- **Key collaborators:** `Config`, `ProgramDatabase`, `Evaluator`, `LLMEnsemble`, `PromptSampler`, `ProcessParallelController`, `EvolutionTracer`

The controller is the glue that binds configuration, LLM interaction, program evaluation, MAP-Elites state management, and checkpointing into a coherent run.

---

## Initialization Sequence (`OpenEvolve.__init__`, `openevolve/controller.py:74-197`)

1. **Config resolution**
   - Accepts either a `Config` instance or a YAML path (`config_path`). When neither is provided, defaults are loaded via `load_config`.
   - Environment hooks in `load_config` inject `OPENAI_API_KEY` and `OPENAI_API_BASE` into the LLM config if present.

2. **Output directory**
   - Defaults to `<initial_program_dir>/openevolve_output/`.
   - Call `OpenEvolve(..., output_dir=...)` to redirect all run artifacts.

3. **Logging setup (`_setup_logging`, `openevolve/controller.py:198-220`)**
   - Creates `<output_dir>/logs/openevolve_<timestamp>.log`.
   - Configures root logger level from `Config.log_level`.
   - Adds both file and console handlers; the controller shares this root logger so you can capture module logs without extra configuration.

4. **Random seeding**
   - When `Config.random_seed` is set the controller seeds Python, NumPy, and derives an MD5-based LLM seed (`openevolve/controller.py:99-123`).
   - Propagates the same seed into every evolution and evaluator model config if the model-level seed is unset. This keeps sampling order deterministic across restarts.

5. **Baseline program ingestion**
   - Loads source text (`_load_initial_program`, `openevolve/controller.py:222-225`).
   - Infers language with `extract_code_language` if `Config.language` is unspecified (`openevolve/utils/code_utils.py:118-173`).
   - Records the file suffix so downstream modules know how to name temp files. You can override `Config.file_suffix` in YAML if you need a different extension than the initial program.

6. **Component wiring**
   - Evolution LLM ensemble (`LLMEnsemble`) is created from `config.llm.models`.
   - Evaluator ensemble is created from `config.llm.evaluator_models`.
   - Two `PromptSampler` instances are kept: one for evolution prompts, another configured with the evaluator system template (`PromptSampler.set_templates`, `openevolve/prompt/sampler.py:37-50`).
   - The controller injects the evolution LLM ensemble into `config.database.novelty_llm` so novelty checks can reuse it (override if you want a dedicated judge).
   - `ProgramDatabase` receives the fully populated `DatabaseConfig`.
   - `Evaluator` is instantiated with the evaluator config, evaluation script path, evaluator LLM ensemble, and prompt sampler. The suffix ensures the temporary file matches the source language.
   - Optional `EvolutionTracer` is constructed when `Config.evolution_trace.enabled` is true, with default paths under `<output_dir>/`.

7. **Parallel controller placeholder**
   - `self.parallel_controller` is initialised to `None` and created lazily in `run()`. This allows the same `OpenEvolve` object to be reused after a run with a fresh worker pool.

---

## Run Loop (`OpenEvolve.run`, `openevolve/controller.py:227-520`)

### 1. Iteration bookkeeping
Determines `max_iterations` from the argument or `Config.max_iterations`. When resuming from `checkpoint_path`, `_load_checkpoint` hydrates the database, with `start_iteration` set to the last recorded iteration plus one.

### 2. Initial program handling
If the database is empty (fresh run) the controller:
- Evaluates the initial program (`Evaluator.evaluate_program`).
- Adds it to the database with `ProgramDatabase.add`.
- Warns if the evaluator does not produce a `combined_score` metric so you can adjust the evaluator for better guidance.

### 3. Parallel execution setup
Within a `try/finally` to guarantee cleanup:
- Constructs `ProcessParallelController` with the config, evaluator path, database, optional evolution tracer, and detected file suffix.
- Registers SIGINT/SIGTERM handlers to request graceful shutdown and to force-exit on the second Ctrl+C event.
- Starts the process pool (`ProcessParallelController.start`).
- Calculates the actual iteration range. A first-time run sets `evolution_start = 1` so the initial evaluation counts as iteration 0.

### 4. Evolution loop
`_run_evolution_with_checkpoints` delegates to `ProcessParallelController.run_evolution`. That coroutine:
- Streams new iterations into the worker pool.
- Calls back into `_save_checkpoint` at the configured interval (`Config.checkpoint_interval`).
- Monitors early stopping thresholds (`Config.early_stopping_patience`, `Config.early_stopping_metric`, `Config.convergence_threshold`).
- Stops on graceful shutdown or target score hit.

### 5. Cleanup and finalization
Regardless of success or interruption:
- The process pool is stopped (`ProcessParallelController.stop`).
- The evolution tracer buffer is flushed and closed.
- The controller retrieves the best known program: first the tracked `database.best_program_id`, then recomputes using `ProgramDatabase.get_best_program`.
- If a different program has a significantly better `combined_score`, it replaces the tracked one to guard against stale references.
- `_save_best_program` writes the winning source and metadata under `<output_dir>/best/`.
- The best `Program` object is returned for downstream use.

---

## Checkpointing

- **Saving (`_save_checkpoint`, `openevolve/controller.py:437-505`)**
  - Creates `<output_dir>/checkpoints/checkpoint_<iteration>/`.
  - Persists the database state (`ProgramDatabase.save`), including prompts/artifacts when logging is enabled.
  - Dumps the current best program source and supplementary info (iteration found, metrics, timestamps).

- **Loading (`_load_checkpoint`, `openevolve/controller.py:494-505`)**
  - Reads database metadata and program JSON from the checkpoint directory.
  - Restores island allocations, feature maps, archive, iteration counters, and cached feature stats.

**Tip:** When scripting restarts, always point `--checkpoint` (CLI) or the `checkpoint_path` argument to the directory containing `metadata.json`, not the database root.

---

## Logging Helpers

- `_log_iteration` (`openevolve/controller.py:413-435`) formats metrics and improvements using `format_metrics_safe` and `format_improvement_safe` so logs remain stable even when metrics mix numeric and categorical values.
- `_format_metrics` and `_format_improvement` (top-level helpers) provide the same functionality for other modules to reuse.

---

## Customization Recipes

### Swap in a custom LLM ensemble
```python
config.llm.models = [
    LLMModelConfig(name="gpt-4o", api_key="...", temperature=0.3, weight=0.5),
    LLMModelConfig(name="claude-3-opus", init_client=build_claude_client, weight=0.5),
]
controller = OpenEvolve(..., config=config)
```
The controller will seed both models and make them available to the database novelty judge automatically.

### Disable logging to disk
Set `config.log_dir = "/dev/null"` or override `_setup_logging` in a subclass to avoid file handlers when running in ephemeral environments.

### Override checkpoint naming
Subclass `OpenEvolve` and override `_save_checkpoint` / `_load_checkpoint` if you need a different checkpoint layout. The rest of the controller refers to those hooks, so no other changes are necessary.

### Run without process parallelism
If your environment prohibits multiprocessing, replace the `ProcessParallelController` with a single-process loop:
```python
from openevolve.iteration import run_iteration_with_shared_db

class SerialOpenEvolve(OpenEvolve):
    async def _run_evolution_with_checkpoints(...):
        for iteration in range(start_iteration, start_iteration + max_iterations):
            result = await run_iteration_with_shared_db(
                iteration,
                self.config,
                self.database,
                self.evaluator,
                self.llm_ensemble,
                self.prompt_sampler,
            )
            if result is None:
                continue
            self.database.add(result.child_program, iteration=iteration)
            ...
```
Reuse the checkpoint hooks to keep persistence intact.

---

## Debugging Pointers

- Enable DEBUG logging to inspect seed values, LLM sampling choices, and iteration summaries.
- Use `OpenEvolve.database.log_island_status()` (called automatically at checkpoints) to track per-island population health.
- To inspect prompts, set `config.database.log_prompts = True` before constructing the controller. Prompt JSON files will be saved under the database directory and included in checkpoints.
- If you see stalled progress, check for novelty rejection in `_is_novel` logs; you might need to adjust `DatabaseConfig.similarity_threshold`.

This controller is intentionally modular: swap in custom components or override protected methods on a subclass whenever you need to change orchestration behavior.


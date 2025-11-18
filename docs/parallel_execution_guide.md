# Parallel Execution & Worker Guide

This guide explains how OpenEvolve distributes work across processes, how database snapshots travel to workers, and how iteration results flow back to the controller.

---

## Components

- **ProcessParallelController** (`openevolve/process_parallel.py:275-660`)
- **Worker initializer & runner** (`openevolve/process_parallel.py:37-272`)
- **SerializableResult** (`openevolve/process_parallel.py:23-34`)
- **Fallback single-process path** (`openevolve/iteration.py`)

---

## High-Level Flow

1. Controller instantiates `ProcessParallelController` with the global config, evaluator file, database, optional tracer, and file suffix.
2. `ProcessParallelController.start()` creates a `ProcessPoolExecutor` with `max_workers = EvaluatorConfig.parallel_evaluations`.
3. Each worker runs `_worker_init`, reconstructing config objects and deferring expensive component creation until needed.
4. `run_evolution` schedules iterations across islands, each invoking `_run_iteration_worker` in a separate process.
5. Workers return `SerializableResult` objects containing new programs, prompts, artifacts, and timing data.
6. The controller integrates results, updates the database, logs traces, and checks for early stopping or checkpoint triggers.

---

## Worker Initialization

`_worker_init(config_dict, evaluation_file, parent_env)` (`openevolve/process_parallel.py:37-94`):
- Propagates environment variables from the parent process (ensures API keys and other settings are available).
- Reconstructs dataclass configs (`Config`, `LLMConfig`, `PromptConfig`, etc.) from the serialized dictionary produced by `_serialize_config`.
- Leaves `_worker_evaluator`, `_worker_llm_ensemble`, and `_worker_prompt_sampler` uninitialised until first use to reduce process spawn cost.

`_lazy_init_worker_components` (`openevolve/process_parallel.py:96-130`) builds:
- Evolution LLM ensemble for code generation.
- Prompt sampler for constructing iteration prompts.
- Evaluator-specific LLM ensemble and sampler for evaluation-time feedback.

Each worker maintains its own evaluator instance pointing to the same evaluation script path. Because workers are separate processes, evaluator modules must be pickle-safe and avoid global mutable state.

---

## Database Snapshotting

Before dispatching a job, `ProcessParallelController._create_database_snapshot` (`openevolve/process_parallel.py:363-386`) creates a lightweight snapshot:
- `programs`: dict of program IDs to `Program.to_dict()` output.
- `islands`: list of island membership sets (converted to lists).
- `current_island`: current island index at the time of sampling (for logging/context).
- `feature_dimensions`: needed to render prompts.
- `artifacts`: cached artifacts for up to 100 programs (to avoid shipping huge payloads).

Workers reconstruct `Program` objects from this snapshot (`_run_iteration_worker`, `openevolve/process_parallel.py:132-145`).

---

## Iteration Execution (`_run_iteration_worker`, `openevolve/process_parallel.py:132-272`)

1. **Sampling**
   - Receives `parent_id` and `inspiration_ids` chosen on the main process via `ProgramDatabase.sample_from_island`.
   - Retrieves parent and inspiration programs from the snapshot.
   - Collects parent artifacts for additional context.

2. **Prompt building**
   - Determines island-specific top programs (sorted by `combined_score` or safe numeric average).
   - Builds prompts using `_worker_prompt_sampler.build_prompt`, feeding in metrics, inspirations, artifacts, feature dimensions, and language.

3. **LLM generation**
   - Calls `LLMEnsemble.generate_with_context`. Wrapped in `asyncio.run` because workers operate in synchronous processes.
   - Handles response parsing:
     - Diff mode: `extract_diffs`, `apply_diff`, `format_diff_summary`.
     - Full rewrite mode: `parse_full_rewrite`.
   - Rejects responses with no valid diff/code or exceeding `Config.max_code_length`, returning an error.

4. **Evaluation**
   - Generates a UUID for the child program and runs `Evaluator.evaluate_program`.
   - Retrieves pending artifacts via `Evaluator.get_pending_artifacts`.

5. **Program serialization**
   - Constructs a new `Program` dataclass populating metrics, lineage, iteration index, and metadata (`changes`, `parent_metrics`, `island`).
   - Packages everything into `SerializableResult`, including prompt and raw LLM response for tracing.

6. **Error handling**
   - Catches all exceptions, logs them, and returns `SerializableResult(error=str(e))`. The controller logs the failure and continues.

---

## Scheduling & Coordination (`ProcessParallelController.run_evolution`, `openevolve/process_parallel.py:388-660`)

- **Initial submission**
  - Calculates `batch_size = min(num_workers * 2, max_iterations)` and distributes evenly across islands (`batch_per_island`).
  - Submits jobs via `_submit_iteration(iteration, island_id)`, which handles sampling and serialization (`openevolve/process_parallel.py:604-660`).

- **Pending futures**
  - Maintains `pending_futures` dict keyed by iteration number and `island_pending` lists to track outstanding work per island.
  - Polls for completed futures; when none are ready, sleeps briefly (`await asyncio.sleep(0.01)`), keeping the event loop responsive.

- **Result processing**
  - On success: adds program to database, stores artifacts, logs to evolution tracer, records prompts, updates island counters, triggers migration if needed.
  - On failure: logs warnings and continues.
  - On timeout: cancels the future, logs timeout details, and moves on.

- **Checkpointing**
  - When `completed_iteration % Config.checkpoint_interval == 0`, logs island status and invokes the controller’s checkpoint callback.

- **Target score & early stopping**
  - Checks `target_score` against `child_program.metrics["combined_score"]`.
  - Tracks best score across iterations; if no improvement for `Config.early_stopping_patience`, sets `self.early_stopping_triggered` and exits.

- **Island rotation**
  - After each completion, if the island has produced `programs_per_island` programs, rotates to the next island (`ProgramDatabase.next_island`).

- **Graceful shutdown**
  - `request_shutdown` sets `shutdown_event`, causing the loop to break and cancel remaining futures.

---

## SerializableResult (`openevolve/process_parallel.py:23-34`)

The return payload contains:
- `child_program_dict`: serialized Program (or `None` on failure).
- `parent_id`: to retrieve parent metadata for tracing.
- `iteration_time`: wall-clock seconds consumed in the worker.
- `prompt`, `llm_response`: original prompt dictionary and raw response (optional).
- `artifacts`: evaluation artifacts.
- `iteration`: iteration index (useful for trace alignment).
- `error`: string describing failure (if any).

The controller reconstructs `Program` objects via `Program(**child_program_dict)`.

---

## Tracing Integration

When `EvolutionTracer` is enabled, `run_evolution` logs each successful iteration:
- Includes prompt, raw response, artifacts, iteration time, and metadata (changes summary).
- Associates each entry with island ID so offline analysis can compare island performance.

---

## Single-Process Alternative

`openevolve/iteration.py:22-119` provides `run_iteration_with_shared_db` which:
- Samples directly from the shared database.
- Uses asynchronous `LLMEnsemble` and `Evaluator` without spawning new processes.
- Returns a `Result` dataclass similar to `SerializableResult`.

To switch:
```python
from openevolve.iteration import run_iteration_with_shared_db

result = await run_iteration_with_shared_db(
    iteration,
    config,
    database,
    evaluator,
    llm_ensemble,
    prompt_sampler,
)
if result:
    database.add(result.child_program, iteration=iteration)
```
Useful for debugging or running inside environments that disable process forking.

---

## Customization Recipes

### Increase worker count
Set `config.evaluator.parallel_evaluations` to the desired number. Ensure your evaluator script can handle running concurrently (no shared global state, thread-safe file access).

### Attach custom metadata
Modify `_run_iteration_worker` to append additional info to `child_program.metadata` (e.g. patch size, diff stats). The database persists metadata automatically.

### Add alternative scheduling logic
Subclass `ProcessParallelController` and override `_submit_iteration` or `run_evolution`:
```python
class PriorityController(ProcessParallelController):
    def _submit_iteration(self, iteration, island_id=None):
        parent, inspirations = self.database.sample(priority_mode=True)
        ...
```
Remember to update the controller to instantiate your subclass.

### Integrate tracing hooks
Wrap `future.result()` calls to record metrics like average iteration duration or failure rates. Because results include `iteration_time`, you can accumulate statistics without instrumenting workers further.

---

## Debugging Tips

- If workers crash immediately, confirm your evaluator module and dependencies are importable in isolated processes.
- Use DEBUG logging in `ProcessParallelController` to see island sampling modes and migration triggers.
- When prompts or artifacts seem mismatched, inspect the serialized snapshot to ensure artifacts are included (`ProgramDatabase.get_artifacts` only returns the most recent artifacts per program).
- On platforms where `fork` is restricted (Windows, sandboxed environments), consider setting `multiprocessing.set_start_method("spawn")` at process entry or rely on the single-process path.

Mastery of the parallel execution layer lets you scale evaluations across cores, plug in custom schedulers, and keep evolution responsive even with expensive evaluators.

# OpenEvolve Expert Guide

This document is meant to make you productive as an OpenEvolve core contributor. It walks through every moving part, shows how the agent, evaluator, and LLMs talk to each other, and points you to the exact code you will need to touch when extending the system.

---

## Quick Index
- [Deep Dive Companion Guides](#deep-dive-companion-guides)
- [Architectural Overview](#architectural-overview)
- [Configuration Bootstrapping](#configuration-bootstrapping)
- [Runtime Orchestrator](#runtime-orchestrator)
- [Program Database & Search](#program-database--search)
- [Prompting & Language Models](#prompting--language-models)
- [Evolution Iteration & Parallelization](#evolution-iteration--parallelization)
- [Evaluation Pipeline](#evaluation-pipeline)
- [LLM ↔ Agent Communication Flow](#llm--agent-communication-flow)
- [Outputs, Logging, and Tracing](#outputs-logging-and-tracing)
- [Interfaces & Entry Points](#interfaces--entry-points)
- [Extending OpenEvolve](#extending-openevolve)
- [Repository Map](#repository-map)
- [Troubleshooting & Testing Tips](#troubleshooting--testing-tips)

---

## Deep Dive Companion Guides

Use these subsystem walkthroughs when you need deeper implementation detail:

- [Controller Deep Dive](docs/controller_guide.md) — wiring, initialization, and lifecycle of `OpenEvolve`.
- [Program Database & Islands Guide](docs/database_islands_guide.md) — MAP-Elites archive, novelty filters, and migration tuning.
- [Prompting System Guide](docs/prompting_guide.md) — template resolution, history injection, and artifact rendering.
- [LLM Integration Guide](docs/llm_guide.md) — ensemble mechanics and provider customization.
- [Evaluator & Cascade Engine Guide](docs/evaluator_guide.md) — evaluation retries, cascade stages, and LLM feedback.
- [Parallel Execution & Worker Guide](docs/parallel_execution_guide.md) — process pool scheduling, worker snapshots, and result handling.

---

## Architectural Overview

**Control surfaces.** OpenEvolve can be driven in three ways:
- CLI wrapper (`openevolve/cli.py#L14-L156`) for command-line runs.
- Library API (`openevolve/api.py#L17-L198`) for embedding in Python code.
- Direct controller usage via the `OpenEvolve` class (`openevolve/controller.py#L60-L520`).

**Evolution pipeline.** Conceptually every iteration passes through:
1. Sample parent(s) and inspiration programs from the MAP-Elites database.
2. Build a task-specific prompt (diff or full rewrite) and send it to an LLM ensemble.
3. Parse the LLM response into either diffs or a fresh program.
4. Evaluate the candidate (potentially via a cascade of tests).
5. Score, archive, and migrate within the MAP-Elites grid.
6. Log metrics, artifacts, prompts, and trace the evolution.

Parallelism is achieved with a process pool so evaluation workloads can run truly concurrently.

---

## Configuration Bootstrapping

OpenEvolve is configured through a hierarchy of dataclasses defined in `openevolve/config.py`.

- **Model configuration** (`LLMModelConfig`, `LLMConfig`) controls API endpoints, sampling parameters, and weighting inside ensembles (`openevolve/config.py#L16-L188`). `LLMConfig.__post_init__` back-fills defaults and supports legacy `primary_model` settings.
- **Prompt configuration** (`PromptConfig`, `openevolve/config.py#L191-L233`) toggles template selection, number of exemplars, artifact embedding, and template stochasticity.
- **Database configuration** (`DatabaseConfig`, `openevolve/config.py#L236-L301`) sets population sizes, island counts, MAP-Elites dimensions, novelty detection parameters, and logging behavior.
- **Evaluator configuration** (`EvaluatorConfig`, `openevolve/config.py#L304-L344`) sets timeout limits, retries, cascade options, parallel evaluation count, and LLM feedback settings.
- **Trace configuration** (`EvolutionTraceConfig`, `openevolve/config.py#L347-L363`) toggles structured evolution logging.
- **Master config** (`Config`, `openevolve/config.py#L366-L475`) aggregates everything and provides convenience defaults.
- **Loading** (`load_config`, `openevolve/config.py#L493-L516`) injects environment-provided API keys and ensures system messages propagate down to model configs.

Typical usage:

```python
from openevolve.config import Config, LLMModelConfig

config = Config()
config.llm.models = [
    LLMModelConfig(name="gpt-4.1-mini", api_key="...", temperature=0.2, weight=0.7),
    LLMModelConfig(name="gpt-4o-mini", api_key="...", temperature=0.4, weight=0.3),
]
config.prompt.template_dir = "my_prompts"
config.database.feature_dimensions = ["combined_score", "diversity"]
```

---

## Runtime Orchestrator

The `OpenEvolve` controller (`openevolve/controller.py#L60-L520`) owns the end-to-end run:

1. **Initialization** (`__init__`, `openevolve/controller.py#L74-L196`)
   - Loads configuration (file or object), output directories, and logging.
   - Seeds RNG across Python, NumPy, and LLM configs to guarantee reproducibility.
   - Loads the baseline program, infers its language, and records the file suffix.
   - Instantiates the evolution LLM ensemble, evaluator LLM ensemble, prompt samplers, database, evaluator, and optional `EvolutionTracer`.

2. **Execution** (`run`, `openevolve/controller.py#L227-L520`)
   - Optionally resumes from checkpoints, rehydrating the database.
   - Evaluates and inserts the initial program if starting fresh.
   - Spins up the `ProcessParallelController` to handle iteration fan-out.
   - Kicks off the evolution loop with checkpoint, target-score, and early-stopping support.
   - On completion it persists the best program and metadata under `<output_dir>/best/`.

3. **Checkpoints** (`_save_checkpoint`, `_load_checkpoint`, `openevolve/controller.py#L437-L505`)
   - Snapshots the database, best program, and metrics at configured intervals.
   - Allows resuming from `<output_dir>/checkpoints/checkpoint_X`.

4. **Best program persistence** (`_save_best_program`, `openevolve/controller.py#L520-L563`)
   - Maintains `best_program.py` plus `best_program_info.json` with lineage and metrics.

Because `OpenEvolve` controls the database and evaluator instances, any customization pipeline typically starts by subclassing or injecting new components through the config before constructing the controller.

---

## Program Database & Search

The MAP-Elites + island population database lives in `openevolve/database.py`.

### Program model
- `Program` dataclass (`openevolve/database.py#L50-L116`) stores code, metrics, lineage, MAP features, metadata, artifacts, embeddings, and prompt history.

### Adding and tracking candidates
- `ProgramDatabase.add` (`openevolve/database.py#L197-L324`) inserts a program, computes feature coordinates, inherits island assignments, runs novelty checks, updates MAP-Elites cells, enforces population limits, and tracks island-best and global-best members.
- `get_best_program` (`openevolve/database.py#L457-L544`) selects the highest-fitness or metric-specific program.
- `sample` / `sample_from_island` (`openevolve/database.py#L364-L455`) provide thread-safe parent + inspiration selection with configurable exploration/exploitation.

### MAP-Elites mechanics
- Feature coordinate calculation, hashing, and coverage logging happen in helper methods used by `add`.
- Migration controls (`should_migrate`, `migrate_programs`, etc., `openevolve/database.py#L1572-L1825`) periodically exchange programs between islands.

### Novelty detection
- Optional embedding-based similarity filter plus LLM arbitration is implemented in `_is_novel`, `_cosine_similarity`, and `_llm_judge_novelty` (`openevolve/database.py#L949-L1020`). By default the evolution LLM ensemble is reused as the novelty judge; you can point `DatabaseConfig.novelty_llm` or `embedding_model` at alternatives.

### Artifact, prompt, and persistence infrastructure
- Artifacts are split into inline JSON vs. filesystem blobs (`store_artifacts`, `get_artifacts`, `openevolve/database.py#L2289-L2487`).
- Prompts/responses are optionally stored per program (`log_prompt`, `openevolve/database.py#L2489-L2520`).
- Database persistence is in `save` and `load` (`openevolve/database.py#L575-L705`), which write per-program JSON and `metadata.json` so runs can be paused/resumed or analyzed later.

If you add new metrics, remember:
- Return raw continuous values in the evaluator so MAP-Elites can scale and bin them.
- Update `DatabaseConfig.feature_dimensions` and, if needed, `feature_bins`.

---

## Prompting & Language Models

### PromptSampler
`PromptSampler` (`openevolve/prompt/sampler.py#L21-L220`) assembles both evolution and evaluation prompts:
- Chooses templates (`diff_user` vs `full_rewrite_user` or overrides) and system messages (`set_templates`).
- Injects metrics, improvement insights, history of top programs, inspiration snippets, and optional artifacts.
- Applies stochastic template variants (`PromptConfig.template_variations`) to diversify instructions.

### TemplateManager
`TemplateManager` (`openevolve/prompt/templates.py#L100-L194`) loads defaults from `openevolve/prompts/defaults` and merges optional custom directories. Fragments in `fragments.json` supply the reusable narrative snippets referenced by the sampler.

To customize prompting:
1. Add `.txt` templates to a custom directory.
2. Point `config.prompt.template_dir` at it.
3. Optionally call `prompt_sampler.set_templates(system_key, user_key)` for bespoke evaluators.

### LLM Ensemble
`LLMEnsemble` (`openevolve/llm/ensemble.py#L17-L91`) is a lightweight router:
- Initializes a list of LLM clients (defaulting to OpenAI-compatible via `OpenAILLM` but you can supply `LLMModelConfig.init_client` builders).
- Normalizes weights and samples models with a seeded RNG for reproducibility (`_sample_model`, `openevolve/llm/ensemble.py#L67-L72`).
- Supports parallel fan-out (`generate_all_with_context`) used by the evaluator’s LLM feedback channel.

### OpenAI-compatible client
`OpenAILLM` (`openevolve/llm/openai.py#L17-L185`) wraps `openai.OpenAI` or Azure clients:
- Builds chat/completions payloads, including special handling for reasoning models that want `max_completion_tokens`.
- Applies retries, timeouts, and optional seeds (`generate_with_context`, `openevolve/llm/openai.py#L64-L174`).
- Runs API calls in a thread pool to keep the async interface responsive.

To add a new provider, create a class implementing `LLMInterface` (`openevolve/llm/base.py#L8-L22`) and set `LLMModelConfig.init_client` to return it.

---

## Evolution Iteration & Parallelization

The heavy lifting happens in `openevolve/process_parallel.py`.

### ProcessParallelController
`ProcessParallelController` (`openevolve/process_parallel.py#L275-L660`) manages a process pool of workers:
- Serializes configuration and environment, stripping non-picklable objects (`_serialize_config`, `openevolve/process_parallel.py#L295-L327`).
- Starts a `ProcessPoolExecutor` with an initializer that rebuilds configs in worker processes (`start`, `openevolve/process_parallel.py#L329-L347`).
- Continuously submits iteration jobs per island, maintains pending futures, and enforces balanced sampling across islands (`run_evolution`, `openevolve/process_parallel.py#L388-L660`).
- Handles early stopping, target score checks, checkpoint triggers, artifact persistence, migrations, and graceful shutdown.

### Worker lifecycle
`_run_iteration_worker` (`openevolve/process_parallel.py#L132-L272`) executes inside each worker:
1. Rehydrate program snapshots and parent/inspiration sets.
2. Generate prompts from island-specific best/diverse programs.
3. Call the evolution LLM ensemble to produce diffs or a rewritten program.
4. Parse diffs (`openevolve/utils/code_utils.py#L34-L96`) or rewrites (`openevolve/utils/code_utils.py#L98-L133`).
5. Run evaluation via an in-process `Evaluator`, retrieving artifacts and metrics.
6. Return everything serialized through `SerializableResult`.

Workers lazily initialize expensive components (`_lazy_init_worker_components`, `openevolve/process_parallel.py#L96-L130`) to keep process startup snappy.

### Alternative single-process path
`openevolve/iteration.py` contains a cooperative coroutine (`run_iteration_with_shared_db`, `openevolve/iteration.py#L22-L119`) used before the process-parallel controller existed. It is still handy for debugging because it runs entirely in-process with shared objects.

---

## Evaluation Pipeline

The evaluator executes and scores candidate programs (`openevolve/evaluator.py`).

### Loading evaluators
- The constructor (`openevolve/evaluator.py#L27-L77`) loads the evaluation module from disk, adding its directory to `sys.path`.
- `_load_evaluation_function` (`openevolve/evaluator.py#L60-L99`) imports the module, validates that an `evaluate` function exists, and checks for cascade stage functions.

### Execution flow
`evaluate_program` (`openevolve/evaluator.py#L132-L288`) handles:
1. Writing the candidate code to a temporary file.
2. Running either `_cascade_evaluate` (`openevolve/evaluator.py#L290-L520`) or `_direct_evaluate`.
3. Wrapping raw dicts into `EvaluationResult` objects (`openevolve/evaluation_result.py#L17-L63`).
4. Aggregating LLM feedback (`_llm_evaluate`, `openevolve/evaluator.py#L550-L620`) when enabled.
5. Capturing artifacts (stderr, traceback, logs) and storing them for the database to persist.
6. Logging metrics via `format_metrics_safe` to keep mixed-value dicts legible.

Cascade evaluation lets you short-circuit expensive tests if a program fails early. Implement `evaluate_stage1`, `evaluate_stage2`, etc., inside the evaluator module and return either numeric metrics or `EvaluationResult` objects with both metrics and artifacts.

### Combining metrics
`get_fitness_score` (`openevolve/utils/metrics_utils.py#L47-L115`) prefers a `combined_score` metric; otherwise it averages non-feature numeric metrics so MAP-Elites can reason about quality even without explicit weights.

### Artifacts
Artifacts accumulate in `_pending_artifacts` and are retrieved via `get_pending_artifacts` (`openevolve/evaluator.py#L210-L242`) immediately after evaluation so the database can store them next to the program entry.

---

## LLM ↔ Agent Communication Flow

Putting the pieces together:
1. **Prompt construction** – `PromptSampler.build_prompt` (`openevolve/prompt/sampler.py#L51-L154`) gathers metrics, history, artifacts, and evolving guidance into a system + user message pair.
2. **Model selection** – `LLMEnsemble.generate_with_context` (`openevolve/llm/ensemble.py#L55-L65`) picks a weighted model and calls `generate_with_context` on it.
3. **Request formatting** – `OpenAILLM.generate_with_context` (`openevolve/llm/openai.py#L64-L174`) injects the system message at the front of the conversation, fills in sampling parameters, and invokes the provider API.
4. **Response parsing** – Workers decode the returned text into diff blocks (`extract_diffs`, `openevolve/utils/code_utils.py#L59-L79`) or complete code blocks (`parse_full_rewrite`, `openevolve/utils/code_utils.py#L98-L133`).
5. **Diff application** – `apply_diff` (`openevolve/utils/code_utils.py#L34-L58`) rewrites the parent code before evaluation.
6. **Logging** – Prompts and responses are stored with the program when `DatabaseConfig.log_prompts` is true (`openevolve/database.py#L2489-L2520`), and `EvolutionTracer` captures the same information if configured.

This same pipeline is reused for novelty adjudication and evaluator-side LLM feedback, so any change to prompt formatting or decoding should consider all three contexts.

---

## Outputs, Logging, and Tracing

- **Logging** – `OpenEvolve._setup_logging` (`openevolve/controller.py#L198-L220`) writes timestamped log files under `<output_dir>/logs/` and mirrors output to the console.
- **Artifacts** – Persisted via `ProgramDatabase.store_artifacts` (`openevolve/database.py#L2289-L2332`) with large artifacts saved under `<db_path>/artifacts/<program_id>/`.
- **Prompts** – Optional prompt/response logs live alongside program JSON files (`programs/<program_id>.json`).
- **Evolution tracing** – `EvolutionTracer` (`openevolve/evolution_trace.py#L61-L199`) streams structured traces (iteration, parent/child IDs, metrics deltas, prompts, artifacts) to JSONL/JSON/HDF5. Configure via `Config.evolution_trace`.
- **Checkpoints** – Created regularly in `<output_dir>/checkpoints/checkpoint_<iteration>/`, containing a database snapshot and the best program at that point.
- **Best outputs** – Final best program and metadata under `<output_dir>/best/`.

---

## Interfaces & Entry Points

### Library API
`run_evolution` (`openevolve/api.py#L17-L94`) accepts inline code strings, file paths, or callables as evaluators and returns an `EvolutionResult` dataclass with best program, score, metrics, and output_dir.

### CLI
`openevolve-run.py` is a thin wrapper around the CLI argument parser defined in `openevolve/cli.py#L14-L156`. It supports:
- Config file selection (`--config`),
- Iteration limits (`--iterations`),
- Target scores (`--target-score`),
- Checkpoint resume (`--checkpoint`),
- API overrides (`--api-base`, `--primary-model`, `--secondary-model`).

### Tests & Examples
- Sample problem definitions live in `examples/`.
- Integration tests and regression checks are under `tests/`.
- Supporting scripts (visualizer, data exports) live in `scripts/`.

---

## Extending OpenEvolve

Here are common extension points with code references:

1. **Add a new LLM provider**
   - Implement `LLMInterface` (`openevolve/llm/base.py#L8-L22`).
   - Set `LLMModelConfig.init_client` to your factory (`openevolve/config.py#L16-L187`).
   - Register it in the config before constructing `OpenEvolve`.

2. **Customize prompts**
   - Create new templates and fragments in a directory.
   - Update `config.prompt.template_dir` and, if necessary, `config.prompt.use_template_stochasticity`.
   - Use `PromptSampler.set_templates` for evaluator-specific instructions (`openevolve/prompt/sampler.py#L37-L50`).

3. **Change evolution mode**
   - Toggle `config.diff_based_evolution` (diff vs full rewrite).
   - Adjust diff parsing logic or formats in `openevolve/utils/code_utils.py`.

4. **Introduce new metrics or MAP dimensions**
   - Modify the evaluator to emit the metrics.
   - Add the names to `config.database.feature_dimensions`.
   - Consider custom bin counts via `config.database.feature_bins`.

5. **Augment evaluator logic**
   - Implement cascade stages (`evaluate_stage1`, etc.) to short-circuit failures.
   - Return `EvaluationResult` with artifacts for richer tracing.
   - Enable LLM quality checks (`config.evaluator.use_llm_feedback`) so `_llm_evaluate` combines subjective scores with objective metrics.

6. **Tune population strategy**
   - Update `DatabaseConfig` ratios (exploration/exploitation), `num_islands`, migration frequency, and novelty thresholds.
   - Override or extend selection helpers in `ProgramDatabase` if you need alternate evolutionary strategies.

7. **Instrument evolution**
   - Enable tracing (`config.evolution_trace.enabled = True`) to feed downstream analytics.
   - Inspect prompts by turning on `DatabaseConfig.log_prompts`.

8. **Swap in a different novelty detector**
   - Provide `DatabaseConfig.embedding_model` (OpenAI or Azure names are supported out of the box, see `openevolve/embedding.py#L10-L83`).
   - Supply a custom `novelty_llm` ensemble if you want a separate model judging novelty.

9. **Run inside your own orchestrator**
   - Drive the `ProcessParallelController` directly, or replace it with a custom scheduler that still produces `SerializableResult`s for compatibility.

Example: customizing for a Rust optimization project.

```python
from openevolve import OpenEvolve
from openevolve.config import Config, LLMModelConfig

config = Config()
config.language = "rust"
config.diff_based_evolution = True
config.llm.models = [
    LLMModelConfig(
        name="gpt-4.1",
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0.3,
        weight=1.0,
    )
]
config.prompt.template_dir = "prompts/rust"
config.database.feature_dimensions = ["combined_score", "latency_ms"]
config.evaluator.use_llm_feedback = False  # rely on benchmark metrics only

controller = OpenEvolve(
    initial_program_path="baseline.rs",
    evaluation_file="bench_evaluator.py",
    config=config,
)
asyncio.run(controller.run(iterations=200))
```

---

## Repository Map

- `openevolve/` – Core package.
  - `controller.py` – Primary orchestrator.
  - `process_parallel.py` – Process pool scheduler and worker logic.
  - `database.py` – MAP-Elites database, islands, artifacts, novelty.
  - `evaluator.py` – Candidate execution, cascade, LLM feedback.
  - `llm/` – Interfaces, ensemble router, OpenAI-compatible client.
  - `prompt/` – Prompt sampler, template manager, defaults.
  - `utils/` – Helpers for async tooling, diffs, metrics, formatting, trace export.
  - `embedding.py`, `novelty_judge.py` – Novelty utilities.
  - `evolution_trace.py` – Structured trace logging.
  - `api.py`, `cli.py` – Library and CLI entry points.
- `configs/` – Ready-made YAML configurations.
- `examples/` – End-to-end problem setups.
- `tests/` – Regression and smoke tests.
- `scripts/` – Visualization and helper scripts.

---

## Troubleshooting & Testing Tips

- **Configuration sanity** – Before a run, pretty-print `config.to_dict()` to ensure model credentials and feature dimensions are populated correctly.
- **Prompt debugging** – Enable `DatabaseConfig.log_prompts` and inspect the generated JSON in the database directory to understand what the LLM saw.
- **Diff failures** – If the LLM keeps returning invalid diffs, temporarily switch to `diff_based_evolution = False` or add stricter instructions inside `diff_user.txt`.
- **Timeouts** – Increase `EvaluatorConfig.timeout` or reduce evaluation load. Remember the process controller adds a 30s buffer.
- **Parallel deadlocks** – Ensure your evaluator is pure-Python or uses subprocesses carefully; worker processes already fork, so nested multiprocessing might need guards.
- **Determinism** – Set `Config.random_seed` to lock in evolution reproducibility (this propagates into the ensemble selection).
- **Tracing** – Turn on `EvolutionTraceConfig` to diagnose regressions; traces are compact and can be replayed offline.

With this map you should be able to navigate any subsystem, know where to add hooks, and understand the data flow from prompt construction to persisted artifacts. Happy evolving!

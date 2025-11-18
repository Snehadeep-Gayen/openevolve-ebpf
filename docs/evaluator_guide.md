# Evaluator & Cascade Engine Guide

This guide explores the evaluator architecture responsible for executing candidate programs, aggregating metrics, capturing artifacts, and integrating optional LLM feedback.

---

## Entry Points

- **Class:** `Evaluator` (`openevolve/evaluator.py:27-523`)
- **Supporting structures:** `EvaluationResult` (`openevolve/evaluation_result.py:13-64`)
- **Async utilities:** `TaskPool`, `run_in_executor` (`openevolve/utils/async_utils.py`)

---

## Initialization (`Evaluator.__init__`, `openevolve/evaluator.py:42-77`)

1. Stores configuration (`EvaluatorConfig`), evaluator script path, LLM ensemble, prompt sampler, and optional database reference for prompt logging.
2. Creates a `TaskPool` limited by `EvaluatorConfig.parallel_evaluations`.
3. Calls `_load_evaluation_function` to import the evaluator module.
4. Prepares a dictionary for pending artifacts keyed by program ID.

---

## Loading Evaluator Modules

`_load_evaluation_function` (`openevolve/evaluator.py:60-99`):
- Verifies the file exists and modifies `sys.path` to allow local imports from the evaluator directory.
- Uses `importlib.util.spec_from_file_location` to load the module under the name `evaluation_module`.
- Ensures an `evaluate(program_path)` function exists.
- Calls `_validate_cascade_configuration` (`openevolve/evaluator.py:101-131`) to detect mismatches between `cascade_evaluation` flag and available `evaluate_stageX` functions. Warnings alert you when the configuration is inconsistent with the module.

**Tip:** Keep evaluator modules deterministic—store any state externally or reset it each invocation to avoid cross-run contamination.

---

## Program Evaluation Flow (`evaluate_program`, `openevolve/evaluator.py:132-288`)

1. **Setup**
   - Records start time and pretty string for logging.
   - Checks `ENABLE_ARTIFACTS` environment variable to determine whether to capture artifacts (defaults to true).

2. **Retry loop**
   - Retries up to `EvaluatorConfig.max_retries` times. Each attempt writes the candidate code to a temporary file with extension `self.program_suffix`.
   - On each attempt:
     - Runs `_cascade_evaluate` or `_direct_evaluate` depending on `EvaluatorConfig.cascade_evaluation`.
     - Wraps results using `_process_evaluation_result`, converting dicts into `EvaluationResult`.
     - Captures artifacts on success, merges them into `_pending_artifacts`.
     - Integrates LLM feedback if enabled (see below).
     - Logs evaluation summary with formatted metrics.
     - Returns metrics.

3. **Timeout handling**
   - If asynchronous calls exceed `EvaluatorConfig.timeout`, an `asyncio.TimeoutError` is raised and the evaluator returns `{"error": 0.0, "timeout": True}`.
   - Pending artifacts store timeout metadata.

4. **Exception handling**
   - Logs warnings for each failure attempt, captures tracebacks into artifacts when enabled, waits one second before retrying.
   - After exhausting retries, logs an error and returns `{"error": 0.0}`.

5. **Cleanup**
   - Always removes the temporary file.

---

## Cascade Evaluation (`_cascade_evaluate`, `openevolve/evaluator.py:290-520`)

Cascade evaluation allows cheap filtering before running expensive tests:

1. **Dynamic module load**
   - Reloads the evaluator module to access stage functions (`evaluate_stage1`, `evaluate_stage2`, `evaluate_stage3`). This ensures updates to the evaluator script take effect without restarting.

2. **Stage execution**
   - Each stage runs inside an executor with `asyncio.wait_for` enforcing the global timeout.
   - Results are processed with `_process_evaluation_result`.
   - If a stage fails (exception, timeout, or returned metrics showing failure), the method short-circuits, records stage failure metrics (e.g. `stage1_passed = 0.0`), and attaches error artifacts.
   - Downstream stages merge metrics and artifacts into a combined `EvaluationResult`.

3. **Error context**
   - `_create_cascade_error_context` (not highlighted here) annotates artifacts with stage information, making it easier to identify which stage failed.

**Implementation tips:**
- Stage functions should either return `EvaluationResult` or dict with float metrics.
- Use consistent metric keys like `stageX_passed` to help MAP-Elites reason about partial success.
- Ensure stage functions clean up any spawned subprocesses to avoid resource leaks in worker processes.

---

## Direct Evaluation (`_direct_evaluate`, `openevolve/evaluator.py:244-272`)

Used when cascade is disabled:
- Wraps `self.evaluate_function` in an executor and applies the timeout.
- Returns raw metrics or `EvaluationResult`.

---

## LLM Feedback (`_llm_evaluate`, `openevolve/evaluator.py:550-620`)

When `EvaluatorConfig.use_llm_feedback` is true:
1. Builds an evaluation prompt using the evaluator-specific sampler (template `evaluation`).
2. Calls `LLMEnsemble.generate_all_with_context` to get responses from every evaluator model.
3. Tries to extract JSON objects from each response:
   - Prefers fenced code blocks ```json ... ```.
   - Falls back to scanning for the first `{...}` substring.
4. Splits response keys into numeric metrics vs. textual artifacts.
5. Weights metrics by the model’s ensemble weight, calculates averages, and injects them into the main evaluation metrics (prefixed with `llm_` and aggregated into `llm_average`).
6. Combines artifacts with evaluator-generated artifacts for later storage.

When combining with objective metrics, the evaluator recalculates `combined_score` (70% objective, 30% LLM average) to bias evolution toward solutions that are both correct and high quality.

---

## Artifact Handling

- `_pending_artifacts` collects artifacts per program ID until the controller or worker retrieves them via `get_pending_artifacts` (`openevolve/evaluator.py:210-242`).
- Artifact entries are plain dicts; values can be strings or bytes.
- The database later splits them into inline JSON or disk files based on size thresholds.

Common artifacts include:
- `stderr`: textual errors,
- `traceback`: stack traces,
- `timeout_duration`, `failure_stage`,
- LLM reasoning strings (from `_llm_evaluate`).

---

## Configuration Reference (`EvaluatorConfig`, `openevolve/config.py:304-344`)

- `timeout`: maximum seconds per evaluation (applies to each cascade stage too).
- `max_retries`: retries after non-timeout exceptions.
- `memory_limit_mb`, `cpu_limit`: reserved for future sandboxing integrations.
- `cascade_evaluation`: toggle cascade engine.
- `cascade_thresholds`: hint for progressively harder test sets in your evaluator script.
- `parallel_evaluations`: number of processes to run evaluation tasks concurrently (wired into `TaskPool` and worker count).
- `use_llm_feedback`, `llm_feedback_weight`: activate LLM assessment and specify blending weight (defaults to 0.1; main code currently assumes 0.3 when recomputing combined score, adjust if you change this weight).
- `enable_artifacts`, `max_artifact_storage`: coordinate with database artifact handling; ensure available disk space when saving large logs.

---

## Writing Evaluator Modules

Example skeleton:
```python
# evaluator.py
from openevolve.evaluation_result import EvaluationResult

def evaluate(program_path: str):
    metrics = run_tests(program_path)
    return {"combined_score": metrics["accuracy"], "latency_ms": metrics["latency"]}

def evaluate_stage1(program_path: str):
    # quick correctness check
    passed = fast_test(program_path)
    if not passed:
        return {"stage1_passed": 0.0, "combined_score": 0.0}
    return {"stage1_passed": 1.0}

def evaluate_stage2(program_path: str):
    metrics, logs = run_full_suite(program_path)
    return EvaluationResult(
        metrics={"combined_score": metrics["accuracy"], "latency_ms": metrics["latency"]},
        artifacts={"logs": logs},
    )
```
Ensure stage functions return quickly when failing so the cascade engine short-circuits and saves resources.

---

## Debugging Tools

- Enable DEBUG logging to capture evaluation retry attempts and metric summaries.
- Inspect artifacts stored in checkpoints to troubleshoot failing programs (`<checkpoint>/programs/<id>.json` and `<checkpoint>/artifacts/<id>/`).
- For recurrent timeouts, extend `EvaluatorConfig.timeout` or profile your evaluator to identify bottlenecks.
- When JSON parsing fails in `_llm_evaluate`, logs show the offending response; adjust your template or parsing logic accordingly.

---

Armed with this knowledge, you can craft sophisticated evaluator scripts, integrate subjective scoring, and diagnose evaluation failures in the evolution pipeline.


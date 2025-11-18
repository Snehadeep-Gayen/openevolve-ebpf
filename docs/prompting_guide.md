# Prompting System Guide

This guide details how OpenEvolve constructs prompts for both code evolution and evaluator feedback, and how you can customize every layer.

---

## Components

- **PromptSampler** – `openevolve/prompt/sampler.py`
- **TemplateManager** – `openevolve/prompt/templates.py`
- **Default templates** – `openevolve/prompts/defaults/*.txt` and `fragments.json`
- **Supporting utilities** – `openevolve/utils/format_utils.py`, `openevolve/utils/metrics_utils.py`

---

## Template Resolution Pipeline

1. **Initialization** (`PromptSampler.__init__`, `openevolve/prompt/sampler.py:21-36`)
   - Loads default templates via `TemplateManager`, optionally merges a custom directory (`PromptConfig.template_dir`).
   - Caches system and user template overrides (`system_template_override`, `user_template_override`) for later.

2. **Template selection** (`PromptSampler.build_prompt`, `openevolve/prompt/sampler.py:51-154`)
   - Determine user template key:
     - Explicit `template_key` argument,
     - Class-level override via `set_templates`,
     - Defaults to `diff_user` or `full_rewrite_user` based on `diff_based_evolution`.
   - Determine system template:
     - Class override via `set_templates`,
     - Otherwise `PromptConfig.system_message`, resolved through `TemplateManager` when it matches a template name.

3. **Template loading** (`TemplateManager.get_template`, `openevolve/prompt/templates.py:138-172`)
   - Looks up the template text. Defaults are read from `.txt` files. Custom directories override matching filenames.
   - Fragments (small sentences) in `fragments.json` are available via `get_fragment`, used for dynamic commentary such as “fitness improved”.

4. **Stochastic variations** (`PromptSampler._apply_template_variations`, `openevolve/prompt/sampler.py:224-247`)
   - If `PromptConfig.use_template_stochasticity` is true, the sampler picks random replacements for placeholders defined in `PromptConfig.template_variations`.
   - Variation keys map to lists of string options, allowing per-run diversity without editing the base template.

---

## Data Injection

When constructing the prompt, `build_prompt` fills placeholders in the template with:

- **Metrics summary** (`_format_metrics`, `openevolve/prompt/sampler.py:156-168`)
  - Formats all metrics using bullet points. Numeric values are rendered to four decimal places.

- **Improvement analysis** (`_identify_improvement_areas`, `openevolve/prompt/sampler.py:170-220`)
  - Calculates fitness delta between current and previous programs using `get_fitness_score`.
  - References prompt fragments for messages like “fitness improved” or “exploring new feature region”.
  - Checks code length thresholds (`PromptConfig.suggest_simplification_after_chars`) to nudge refactors.

- **Evolution history** (`_format_evolution_history`, `openevolve/prompt/sampler.py:222-292`)
  - Renders previous attempts (`previous_programs`) and top programs (`top_programs`) using templates `previous_attempt`, `top_program`, and `evolution_history`.
  - Incorporates inspiration programs in a dedicated section when provided.
  - Each program is truncated to 200 characters by default to keep prompts manageable.

- **Artifacts** (`_render_artifacts`, `openevolve/prompt/sampler.py:294-349`)
  - Conditionally included when `PromptConfig.include_artifacts` is true and the parent program produced artifacts (evaluation logs, stderr, etc.).
  - Caps artifact bytes using `PromptConfig.max_artifact_bytes`.

- **Feature coordinates** (`format_feature_coordinates`, `openevolve/utils/metrics_utils.py:117-150`)
  - Shows MAP-Elites positions (e.g. `diversity=0.82`) to encourage the model to reason about search space coverage.

The final prompt dictionary contains `"system"` and `"user"` keys sent to the LLM ensemble.

---

## Evolution vs. Evaluation Prompts

- **Evolution prompts**
  - Constructed by the main `PromptSampler` instance.
  - Template keys `diff_user` and `full_rewrite_user` enforce diff-based or full rewrite instructions.
  - Diff template instructs the LLM to respond using the SEARCH/REPLACE format that `apply_diff` consumes.

- **Evaluator prompts**
  - The controller clones a sampler and calls `set_templates("evaluator_system_message")` so system prompts use the evaluator framing.
  - When `Evaluator.use_llm_feedback` is true, `_llm_evaluate` calls `build_prompt` with `template_key="evaluation"` to solicit JSON-formatted scores for readability, maintainability, etc. (`openevolve/evaluator.py:550-620`).

- **Novelty prompts**
  - Constructed manually inside `_llm_judge_novelty` using `NOVELTY_SYSTEM_MSG` and `NOVELTY_USER_MSG` (`openevolve/novelty_judge.py`). They do not pass through `PromptSampler`.

---

## Customization Workflows

### Override prompt text
1. Create a directory, e.g. `prompts/custom/`.
2. Add files matching default names (`diff_user.txt`, `system_message.txt`, etc.).
3. Set `config.prompt.template_dir = "prompts/custom"`.
4. Optionally leave some templates absent—defaults fill the gaps.

### Add variation slots
```yaml
prompt:
  template_variations:
    motivation_line:
      - "Push beyond yesterday's design."
      - "Consider algorithmic trade-offs explicitly."
```
In your template include `{motivation_line}` to insert one of the variants at runtime.

### Switch to full rewrites mid-run
```python
config.diff_based_evolution = False
config.prompt.use_meta_prompting = False  # still unused but kept for future features
```
Alternatively, call `prompt_sampler.set_templates(user_template="full_rewrite_user")` on a per-run basis.

### Inject domain-specific metrics
If your evaluator returns domain metrics (e.g. `latency_ms`), they automatically appear in `{metrics}` and `{feature_coords}`. Update templates to include targeted instructions:
```text
- Target latency: strive for {latency_target} ms.
```
Pass `latency_target` through `build_prompt(..., latency_target=...)`.

### Modify artifact rendering
Override `_render_artifacts` in a subclass of `PromptSampler` if you want to format logs differently (e.g. truncate stack traces, add headings). Remember to register your subclass in the controller or iteration code.

### Inject repository-wide context via agent call
When evolving large applications you may want the LLM to see a high-level repository map generated by an auxiliary agent (e.g. Codex). The recommended integration points are:

1. **Generate the repo digest once per run.** Extend `OpenEvolve.__init__` or hook into your run harness before `ProcessParallelController` starts (`openevolve/controller.py:74-227`). Call the agent with the project root, capture its structured summary (directory tree, key modules, ownership), and attach it to the config (e.g. `config.prompt.extra_repository_context`).
2. **Propagate context to workers.** Update `ProcessParallelController._serialize_config` (`openevolve/process_parallel.py:295-327`) so the serialized prompt config includes your new field, and teach `_worker_init` / `_lazy_init_worker_components` to stash it alongside the sampler. Workers already receive the prompt config dict, so storing the context string or chunks there keeps it picklable.
3. **Feed it into prompts.** When `_run_iteration_worker` calls `prompt_sampler.build_prompt(...)` (`openevolve/process_parallel.py:132-272`), pass `repository_map=context` via `**kwargs`. Add `{repository_map}` (or similar) to your templates so the text appears in the user prompt. Because `build_prompt` simply forwards `**kwargs`, no sampler changes are required beyond ensuring templates consume the placeholder.
4. **Trim and refresh strategically.** Keep the digest concise (top-level modules, key responsibilities) to avoid token bloat. If the repository structure changes during evolution, regenerate the summary at checkpoint boundaries and update the shared context before resuming.

This approach keeps the expensive repository analysis outside the inner iteration loop while guaranteeing every prompt includes the latest holistic view of the codebase.

---

## Validation Helpers

- **Diff compliance**
  - `openevolve/utils/code_utils.extract_diffs` (`openevolve/utils/code_utils.py:59-79`) requires exact SEARCH/REPLACE markers. Adjust the diff template to match this format precisely.

- **Length control**
  - `PromptSampler._truncate_program` (invoked inside `_format_program_for_template`) clips code sections to avoid hitting token limits. Increase or decrease truncation thresholds if your domain needs more context.

- **Artifacts security**
  - `PromptConfig.artifact_security_filter` (currently unused) can be implemented to sanitize artifacts before inclusion. Extend `_render_artifacts` to respect it.

---

## Debugging Tips

- Enable `DatabaseConfig.log_prompts` to write every prompt and response to JSON with the program record (`openevolve/database.py:2489-2520`).
- Inspect `openevolve/prompts/defaults` to understand base structures before replacing them.
- Insert diagnostic text inside templates (e.g. `TEMPLATE_VERSION=1`) to confirm which template version is active.
- When prompts exceed LLM limits, reduce `num_top_programs`, `num_diverse_programs`, or increase truncation to shrink the context payload.

With mastery over the prompting subsystem you can align the agent’s instructions with your domain, control format expectations for diffs, and integrate evaluator-side qualitative feedback cleanly.

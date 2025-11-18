# LLM Integration Guide

This guide explains how OpenEvolve interfaces with language models, how ensembles choose models, and how to wire in alternative providers or custom clients.

---

## Components

- **LLMInterface** (`openevolve/llm/base.py:8-22`)
- **LLMEnsemble** (`openevolve/llm/ensemble.py:17-91`)
- **OpenAILLM** (`openevolve/llm/openai.py:17-185`)
- **Config hooks** (`openevolve/config.py:16-188`)

---

## Interface Contract

Every model client must implement:
- `async def generate(self, prompt: str, **kwargs) -> str`
- `async def generate_with_context(self, system_message: str, messages: List[Dict[str, str]], **kwargs) -> str`

`generate` is a convenience wrapper; most of OpenEvolve relies on `generate_with_context` because it carries both system and user text.

---

## Ensemble Mechanics

- **Initialization** (`LLMEnsemble.__init__`, `openevolve/llm/ensemble.py:20-53`)
  - Iterates `models_cfg` (list of `LLMModelConfig`) and instantiates clients.
  - If `model_cfg.init_client` is provided, uses that factory; otherwise falls back to `OpenAILLM`.
  - Collects weights and normalizes them to a probability distribution. Zero weights should be avoided—they will still be normalized, potentially skewing distribution.
  - Seeds an internal `random.Random` instance with `models_cfg[0].random_seed` when available. The controller propagates global seeds here to guarantee deterministic model sampling.

- **Sampling** (`_sample_model`, `openevolve/llm/ensemble.py:67-72`)
  - Uses `random_state.choices` with the weight distribution to pick a model index.
  - Emits a log line with the selected model name—use DEBUG level to avoid clutter.

- **Generation** (`generate`, `generate_with_context`)
  - Delegates to the sampled client. Keyword arguments (temperature overrides, max tokens, etc.) pass straight through to the client.

- **Parallel generation** (`generate_multiple`, `parallel_generate`, `generate_all_with_context`)
  - Provide convenience for evaluation-time fan-out. `generate_all_with_context` hits every client sequentially, returning the list of responses.

---

## OpenAI-Compatible Client

`OpenAILLM` wraps `openai.OpenAI` chats:

1. **Client construction** (`openevolve/llm/openai.py:35-58`)
   - Sets base URL, API key, temperatures, retry logic, and optional reasoning effort.
   - Accepts `max_retries` > 0 to let the OpenAI SDK internally retry connection errors.

2. **Message formatting** (`generate_with_context`, `openevolve/llm/openai.py:64-174`)
   - Prepends the system message to the conversation.
   - Detects reasoning models based on model name (prefix checks for `o1-`, `gpt-5-`, etc.) and switches to `max_completion_tokens`.
   - Standard completion models use `temperature`, `top_p`, and `max_tokens`.
   - Injects seeds unless the API base is Google AI Studio (seed unsupported).

3. **Retry loop**
   - Retries up to `retries` times on timeouts or exceptions, waiting `retry_delay` seconds between attempts.
   - Logs warnings on failures; escalates to error after exhausting retries.

4. **Async execution**
   - Calls the synchronous SDK from a thread executor via `loop.run_in_executor` to keep downstream code awaitable.
   - Returns the string content (`response.choices[0].message.content`).

---

## Configuring Models

Configure via YAML or direct code using `LLMModelConfig` and `LLMConfig`. Important fields:

- `name`: model identifier known to the target API.
- `api_base` / `api_key`: endpoint and credential.
- `init_client`: callable that receives the `LLMModelConfig` and returns an `LLMInterface` implementation; use for non-OpenAI providers.
- `weight`: relative probability mass for ensemble sampling.
- `temperature`, `top_p`, `max_tokens`: defaults passed to the client.
- `timeout`, `retries`, `retry_delay`: request-level resilience.
- `random_seed`: per-model seed (overridden by controller when global seed is set).
- `reasoning_effort`: forwarded to reasoning-capable clients (OpenAI uses `reasoning_effort` for O-series models).

`LLMConfig` inherits from `LLMModelConfig` and adds:
- `models`: list of `LLMModelConfig` for evolution.
- `evaluator_models`: optional separate list; defaults to `models.copy()` if empty.
- Legacy `primary_model`, `secondary_model` fields for simple two-model setups.
- `update_model_params` / `rebuild_models` utilities used by CLI overrides and environment seeding.

---

## Custom Provider Example

```python
from openevolve.llm.base import LLMInterface

class MyLLM(LLMInterface):
    def __init__(self, cfg):
        self.client = CustomSDK(api_key=cfg.api_key)

    async def generate(self, prompt, **kwargs):
        return await self.generate_with_context("You are...", [{"role": "user", "content": prompt}], **kwargs)

    async def generate_with_context(self, system_message, messages, **kwargs):
        payload = {"system": system_message, "messages": messages, **kwargs}
        return await self.client.completions(payload)

config.llm.models = [
    LLMModelConfig(
        name="my-llm-1",
        api_key="secret",
        init_client=lambda cfg: MyLLM(cfg),
        weight=1.0,
    )
]
```
Now the ensemble uses `MyLLM` instead of `OpenAILLM`.

---

## LLM Feedback Loop

`Evaluator._llm_evaluate` (`openevolve/evaluator.py:550-620`) uses `generate_all_with_context` to gather multiple model opinions:
- Each response is parsed for JSON metrics.
- `LLMModelConfig.weight` is reused to weight the scores when averaging.
- Artifacts (reasoning strings) are merged with evaluation artifacts and stored alongside the program.

When enabling this feature (`config.evaluator.use_llm_feedback = True`), ensure evaluator models return consistent JSON; misformatted responses raise parsing errors logged at WARNING level.

---

## Novelty and Secondary Uses

- Novelty judging (`ProgramDatabase._llm_judge_novelty`, `openevolve/database.py:972-1019`) reuses the evolution ensemble by default. To isolate novelty costs, assign a dedicated ensemble to `DatabaseConfig.novelty_llm`.
- Any module can request the ensemble directly—just remember that `generate_with_context` picks a random model each time. If you need deterministic per-call routing, call `generate_all_with_context` or construct a single-model ensemble.

---

## Debugging & Monitoring

- Enable DEBUG logging for `openevolve.llm.ensemble` and `openevolve.llm.openai` to see which models are sampled and which parameters are used.
- Wrap `OpenAILLM._call_api` to instrument latency or track token usage by subclassing `OpenAILLM`.
- If your provider requires additional headers or payload structure, extend `LLMModelConfig` via subclassing or use the `init_client` factory to inject custom behavior.

---

With control over the LLM subsystem you can route traffic to heterogeneous providers, tune sampling strategies, and plug LLM feedback into evaluators or novelty pipelines without touching the rest of the evolution loop.


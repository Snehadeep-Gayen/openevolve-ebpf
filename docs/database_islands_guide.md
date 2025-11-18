# Program Database & Island Evolution Guide

This document unpacks the MAP-Elites database, island mechanics, novelty filters, and migration policies implemented in `openevolve/database.py`.

---

## Core Structures

- **Program dataclass** (`openevolve/database.py:50-116`)
  - Captures code, lineage, metrics, MAP features, metadata, prompt logs, artifacts, and optional embeddings.
  - `generation` increments per child, while `iteration_found` records the global iteration.
  - `metadata["island"]` identifies the island membership (populated inside `ProgramDatabase.add`).

- **ProgramDatabase** (`openevolve/database.py:129-2560`)
  - Manages the population across multiple islands, maintains MAP-Elites grids, archives elites, and persists state to disk.

---

## Adding Programs (`ProgramDatabase.add`, `openevolve/database.py:197-324`)

1. **Iteration tracking**
   - Updates `program.iteration_found` and `self.last_iteration` when an iteration index is provided (important for checkpoint resumes).

2. **Feature coordinates**
   - `_calculate_feature_coords` (not shown in full here) computes normalized coordinates for each dimension in `DatabaseConfig.feature_dimensions`. Values are scaled using running stats (`self.feature_stats`) and converted into discrete bins.
   - `_feature_coords_to_key` (`openevolve/database.py:940-947`) transforms coordinates into a hashable string for MAP-Elites lookups.

3. **Island selection**
   - If `target_island` is supplied (e.g. during migration) the program is placed there.
   - Otherwise it inherits its parent’s island through `parent.metadata["island"]`.
   - If the parent is missing island metadata (unlikely unless loading old checkpoints) the program falls back to `self.current_island`, allowing graceful recovery.

4. **Novelty gating**
   - `_is_novel` (`openevolve/database.py:1022-1060`) checks whether the program should be admitted:
     - When no embedding model or threshold is configured, the gate is effectively open.
     - Otherwise, it computes embeddings via `EmbeddingClient` (`openevolve/embedding.py:13-83`), compares cosine similarity with existing island members, and, if necessary, calls `_llm_judge_novelty` (`openevolve/database.py:972-1019`) to arbitrate borderline cases.
   - Failed novelty checks cause the program to be skipped but still return its ID so upstream code can log the event.

5. **MAP-Elites update**
   - If the corresponding cell is empty or the new program has higher fitness (`_is_better`, `openevolve/database.py:1049-1165`), it replaces the incumbent.
   - Logs milestones when new cells are occupied or when fitness improves existing cells, including coverage percentages per island.

6. **Population maintenance**
   - Adds the program to the island set (`self.islands[island_idx]`).
   - Updates the island-best cache and global `best_program_id`.
   - Ensures population size limits via `_enforce_population_limit` (prunes low-fitness individuals while preserving elites).
   - Persists artifacts and prompt logs when `DatabaseConfig.db_path` is set.

---

## Sampling Parents & Inspirations

- **Global sampling** (`sample`, `openevolve/database.py:364-383`)
  - Picks a parent via `_sample_parent` (mix of archive elites, random exploration, and fitness-weighted draws).
  - Chooses inspiration programs with `_sample_inspirations`.

- **Island-aware sampling** (`sample_from_island`, `openevolve/database.py:385-455`)
  - Called by `ProcessParallelController` to keep each worker focused on one island.
  - Uses the same exploration/exploitation/weighted split but restricted to the island’s membership set.
  - If an island is empty (typical at startup or after aggressive filtering) it falls back to global sampling to bootstrap new lineages.

**Tip:** Tune `DatabaseConfig.exploration_ratio` and `DatabaseConfig.exploitation_ratio` to bias selection. The residual probability (`1 - exploration - exploitation`) uses fitness-weighted sampling.

---

## MAP-Elites Parameters

Configure in `DatabaseConfig` (`openevolve/config.py:236-301`):

- `population_size`: total programs retained across all islands.
- `archive_size`: size of the elite archive for exploitation sampling.
- `num_islands`: number of concurrent islands—default 5 encourages diversity but increases memory.
- `feature_dimensions`: list of metrics the MAP grid tracks (e.g. `["complexity", "diversity"]`). Each must be numeric and returned by the evaluator.
- `feature_bins`: either an integer (uniform bins per dimension) or a dict mapping dimension to bin count.
- `diversity_reference_size`: history size for reference-based diversity calculations.

**Coordinate scaling** is handled automatically using min / max per dimension. When you add a new metric, re-run the evaluator to populate enough data for meaningful bins before relying on MAP coverage statistics.

---

## Novelty System

### Embedding pipeline
- Controlled by `DatabaseConfig.embedding_model` (set to an OpenAI or Azure embedding name).
- Embeddings are cached inside `Program.embedding` to avoid recomputation (`ProgramDatabase.add` stores the vector once generated).
- Cosine similarity is computed via `_cosine_similarity` (`openevolve/database.py:949-970`).

### Threshold & fallback
- `DatabaseConfig.similarity_threshold` defines the cosine similarity above which a novelty judgement is required.
- When an embedding exceeds the threshold, `_llm_judge_novelty` constructs a novelty prompt (`openevolve/novelty_judge.py`) and asks the `novelty_llm` (defaulting to the evolution ensemble) whether the programs are meaningfully different.
- You can replace the judge by assigning `config.database.novelty_llm = custom_llm_ensemble` before constructing the controller.

### Disabling novelty
- Leave `embedding_model` unset or set `similarity_threshold <= 0.0`. The database will accept every program, useful when debugging or when the search space is small.

---

## Migration & Island Rotation

### Rotation
- `ProgramDatabase.next_island` (called by `ProcessParallelController`) cycles `self.current_island` after a configurable number of additions (`programs_per_island`) so each island receives attention.

### Migration scheduling
- `should_migrate` (`openevolve/database.py:1654-1676`) checks whether enough generations have passed since the last migration (`self.last_migration_generation`) or if islands exhibit stagnation.
- `DatabaseConfig.migration_interval` and `migration_rate` control frequency and fraction of migrants.

### Migration execution
- `migrate_programs` (`openevolve/database.py:1739-1825`) selects top performers from each island and reassigns them to target islands.
- Metadata, MAP cells, and archive membership are updated accordingly.
- Migration can import elites into underperforming islands while spreading diversity without collapsing islands into a single population.

---

## Persistence & Prompt Logs

- **Saving** (`ProgramDatabase.save`, `openevolve/database.py:575-623`)
  - Writes each program to `<db_path>/programs/<program_id>.json`.
  - Serializes prompts when `DatabaseConfig.log_prompts` is true.
  - Saves `metadata.json` with island layout, feature stats, archive contents, and iteration counters.

- **Loading** (`ProgramDatabase.load`, `openevolve/database.py:624-813`)
  - Reconstructs programs, islands, archives, feature maps, and feature scaling stats.
  - `_reconstruct_islands` handles missing programs gracefully and redistributes if metadata is incomplete.

- **Artifacts**
  - Small artifacts (below `artifact_size_threshold`) are embedded in the JSON (`Program.artifacts_json`).
  - Large artifacts are written to `<db_path>/artifacts/<program_id>/` (`ProgramDatabase.store_artifacts`, `openevolve/database.py:2289-2332`).
  - Retrieval via `get_artifacts` merges JSON and file-backed artifacts (`openevolve/database.py:2334-2487`).

---

## Customization Recipes

### Add a new MAP dimension
1. Modify the evaluator to emit a numeric metric, e.g. `{"combined_score": 0.8, "latency_ms": 12.3}`.
2. Set `config.database.feature_dimensions = ["latency_ms", "combined_score"]`.
3. Adjust `feature_bins` to reflect the new search landscape.
4. Optionally weight selection by the new metric using custom helpers in `_is_better`.

### Increase exploration
```python
config.database.exploration_ratio = 0.4
config.database.exploitation_ratio = 0.3
```
The residual 0.3 becomes fitness-weighted sampling, leading to more diverse parent picks.

### Replace novelty judge
```python
config.database.embedding_model = "text-embedding-3-large"
config.database.similarity_threshold = 0.98
config.database.novelty_llm = LLMEnsemble([
    LLMModelConfig(name="gpt-4o-mini", api_key="..."),
])
```
Call this before constructing `OpenEvolve`. Now novelty checks use a dedicated model and larger embeddings.

### Track prompt history
```python
config.database.log_prompts = True
config.database.db_path = "runs/my_experiment/db"
```
Prompts and LLM responses become available under `programs/<id>.json` and in checkpoints for auditability.

---

## Debugging Tips

- Use `ProgramDatabase.log_island_status()` (`openevolve/database.py:2196-2239`) to print per-island stats: population size, best scores, average fitness, diversity, and generation counters.
- Inspect `/programs/*.json` in checkpoints to understand how MAP cells evolve over time.
- Enable DEBUG logging to see novelty verdicts and migration events.
- If islands empty out unexpectedly, verify novelty thresholds and evaluator timeouts (timed-out evaluations often return zero metrics, leading to immediate pruning).

With these insights you can surgically adjust evolutionary pressure, map new metrics, or plug in alternative novelty detection strategies while keeping the overall architecture intact.


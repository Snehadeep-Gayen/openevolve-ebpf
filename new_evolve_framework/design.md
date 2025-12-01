Here’s a first-pass design doc based on what I can read from the whiteboard, turned into a clean Python-friendly design. I’ll keep it reasonably high-level, then give concrete class / function sketches.

---

# Design: Idea/Program Evolution Engine

## 1. Goal

We want a system that:

* Stores **design ideas (“nodes”)** and the **programs** that implement them.
* Automatically **generates new programs**, **runs** them, and **evaluates** them using my eval_agent code
* Uses a **DB of nodes** to iteratively **evolve better ideas/programs**.
* Keeps **logging and experiment structure** sane (Directory per invocation with one dir per idea iteration).

This is essentially a search / evolutionary loop over ideas + code.

---

## 2. Core Concepts

### 2.1 `Program`

A *Program* is a concrete candidate implementation.

* `code`
* `perf metrics`
* `logs`
* `eval (LLM)`
  * `summary`
  * `perf_expl`

The exact structure is also obvious from the openevolve/ codebase parallel_run.py code, based on what's used there.

**Responsibilities**

* Hold the **source code** and metadata.
* Record **runtime results** (perf metrics, success/failure).
* Record **LLM evaluation** (scores, textual feedback).
* Know where its **logs** are stored.

**Proposed fields (Python dataclass)**

---

### 2.2 `Node`

A *Node* is a higher-level **idea/design** that may have many Programs.

From the board:

* `idea/design`

  * `evidence`
  * `risks`
  * `design`
  * `code snippets`
* `Best Program`
* `Program List`

**Responsibilities**

* Capture the **idea** with supporting information.
* Track all **programs** created for this idea.
* Track the **current best program** (by some scoring function).
* Keep enough context to generate new code for the idea.


### 2.3 DB / Persistence Layer

* `DB`

  * `Stores Nodes`
  * `Best Node`


`get_best_node()` can compute “best” via a composite score (perf metrics, right now "combined_score" area under curve ov p99).
There might also be some way to sample, like maybe probabilisticly based on the score.

---

### 2.4 Logging & Experiment Layout

From the board:

* **Logging**:

Design:

* A root `results_dir/exp_<timestamp>/`.
* Inside:

  * `iteration_<n>/`
  * a file per piece. So the eval_agent has an output json, so does the idea generation
  * Copy over the exp-*/ directory from the evaluation file (right nwo in openevolve this is debug_directory). There will be one of these per program evaluated. 

We expose a **Logger/RunManager** that:

* Creates iteration directories.
* Returns paths to use for logs.
* copies config files into each iteration dir for reproducibility.

---

## 3. Function-Level Design

These map pretty directly to the whiteboard scribbles.

### 3.1 `similarity_check_idea`

From the board: `Similarity_check idea`. Should this use an LLM or an embedding?

Purpose: avoid generating duplicate/super-similar ideas.

Look at all nodes in database to see the previous ideas.

Implementation options:

* Use embeddings: encode `candidate.description` + `title`, compare cosine similarity.
* Use LLM judgement if you want something heavier.

---

### `generate_best_program(node)`

This will run a loop of gen_program some configurable number of times.

## `gen_program(node)`

Purpose: generate one or more new Program objects from a Node 

Steps:

1. Build a prompt from parent `node`
 The node has an eval_explanation and code that can be used to create a prompt to write a new idea, using my idea_agent.py. 

`eval_program`
Run the evaluation program (from openevolve this is eval.py, will reuse the same file)

If this fails (there's an evaluation response that has fields called compiled, load, run_success). There should be a loop that is focused on an LLM call to just make this program compile and then run eval again. There should be a configurable number of times to try to get it to compile before just skipping the program. 

Once it succeeds, we should add the program (with the updated evaluation and everything) to the node program list.

At the end of the gen_program calls loop find_program should set best program and return the node back to the database. 


---

### 3.5 `evolve(DB)` loop

* `Evolve(DB)`

  * `Sample Node from DB`
    * `pick best`?
    * `pick probabilistically?`?
  * `Generate new idea from parent node (using idea_agent.py). Can also add in inspiration from randomly sampled other nodes. This is something I want to experiment with to see effectiveness. `
  * `Similarity_check to see if we should try again (if it fails the check, maybe add to the prompt that that idea is forbidden)`
  * `call find_best_program to populate the programs in the node, including the best program`

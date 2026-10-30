# Nutrition RAG + Agent: raw vs. LangChain vs. LangGraph

A small nutrition-**education** assistant built four ways, to understand what RAG and agent frameworks actually do. It runs entirely on a laptop (16 GB MacBook, `llama3.1:8b` via Ollama, `all-MiniLM-L6-v2` embeddings). It answers from 14 public NIDDK and MedlinePlus pages, calls tools for the math, cites its sources, and refuses unsafe requests.

> **Scope:** general education only. It is not medical advice and gives no personalised diet plans. Math comes from code, never from the LLM.
>
> **How it was built:** I built this with Claude Code (Anthropic's coding agent) as a pair-programmer. It wrote the code and ran the experiments, and I'm working through the code and results to learn them. All numbers below come from the saved run files in `results/`, not from memory.

## TL;DR: what I found

| | Raw RAG | Raw agent | LangChain | LangGraph |
|---|---|---|---|---|
| Lookup facts (8) | 8/8 | 8/8 | 8/8 | 8/8 |
| Multi-hop / tool math (7) | 0/7 | **6/7** | 0/7 | 0/7 |
| Not in corpus → abstain (6) | 5/6 | 3/6 | **6/6** | **6/6** |
| Guardrails (6), after fixes | 6/6 | 6/6 | 6/6 | 6/6 |
| Guardrails, **before** fixes | 4/6 | 4/6 | 4/6 | **6/6** |
| Correct **and grounded** (10 gold-evidence Qs) | 7/10 | 7/10 | 7/10 | 7/10 |
| Mean seconds / question | 18.7 | 27.7 | 17.4 | 34.4 |

![Accuracy by category](docs/img/accuracy_by_category.png)

The four takeaways, from most to least surprising:

1. **A "correct" answer is not a "grounded" answer.** All four pipelines answered the BMI question correctly and cited `[1]`, but the retrieved chunks never contained the number. My HTML extractor had dropped the table, so the answer came from model memory with a citation that didn't support it. Only a grounding check (did the retrieved evidence contain the answer?) exposes that.
2. **Tools beat prompting for math, but the agent leaks outside knowledge.** The agent was the only pipeline to get the TDEE math right (6/7), because code does the arithmetic. It was also the worst at saying "not in my sources" (3/6), answering from memory instead.
3. **Frameworks didn't improve accuracy here; the structure bought safety.** LangChain and LangGraph matched the raw pipeline on lookups. LangGraph's extra layer helped in two places: it scored 6/6 on abstention (the others 5/6 and 3/6), and its LLM safety classifier caught two unsafe prompts that my regex rules missed.
4. **Retrieval and generation fail independently, so measure them separately.** The chunk-size sweep needs no LLM and runs in seconds.
5. **(Part 2) Ask-before-assume and calculator routing belong in code, not the prompt.** On a held-out set, the baseline pipelines asked for missing inputs 0/2 times and invented them instead; a code-level gate asked 2/2, and a LangGraph calculator route went from 0/7 to 6/7 on the math questions. Adding more guards to the free-form agent did *not* help and added failure modes. See [Part 2](#part-2-chained-questions-missing-inputs-and-a-calculator-route).

## What's in the corpus

14 pages: 6 from [NIDDK](https://www.niddk.nih.gov) (weight management) and 8 from the [MedlinePlus](https://medlineplus.gov) encyclopedia (protein, carbs, fiber, water, sodium, sugars, athletic nutrition, bulimia), about 93k characters. CDC pages returned HTTP 403 to scripts, so I skipped them rather than work around the block. Raw pages are **not** committed; `src/fetch_sources.py` and `sources.json` re-download them. Check each source's terms before redistributing any text.

## Architecture

```
Stage 1  raw RAG        chunk.py → embed.py → retrieve.py → rag.py          (numpy + requests only)
Stage 2  raw agent      tools.py + agent.py                                  (hand-written loop)
Stage 3  LangChain      lc_rag.py                                            (Chroma + LCEL)
Stage 4  LangGraph      graph_rag.py                                         (state machine)
Stage 5  eval           evaluate.py → score.py → make_figures.py
Part 2   ask + calc     inputs.py, calc.py, graph_tools.py, guarded agent.py → evaluate2.py → score2.py
```

```mermaid
flowchart TD
  S[safety: rules, then LLM classifier] -->|unsafe| E[END: refuse + redirect]
  S -->|ok| R[retrieve top-k]
  R --> G{grade: do chunks answer it?}
  G -->|yes| GEN[generate with citations]
  G -->|no, retries < 2| W[rewrite query] --> R
  G -->|no, retries = 2| AB[generate: abstain 'not in my sources']
  GEN --> E2[END]
  AB --> E2
```

## Stage 1: RAG with no framework

Pipeline: **chunk by section with overlap → embed → store in a numpy array → cosine top-k → prompt → LLM with citations.**

- **Chunking** (`chunk.py`): split on `##` headings first, then pack sentences up to `max_chars` with a character overlap. I prepend `"<page title> - <section heading>"` to every chunk so it is self-describing once separated from its page.
- **"Vector store"** is a `(n_chunks, 384)` float32 array plus a parallel list of metadata. That is all a vector DB is at this scale; real ones add indexing so search isn't a full scan.
- **Retrieval** is one line, because the vectors are L2-normalised so cosine similarity is a dot product:
  ```python
  scores = self.vecs @ q                 # cosine similarity
  top = np.argsort(-scores)[:k]
  ```
- **Prompt** (`rag.py`): numbered sources, "answer only from these, cite `[n]`, otherwise say *I don't have that in my sources*".

![Raw RAG output](docs/img/01_raw_rag_citation.png)

**Bug found by inspecting retrieval:** boilerplate chunks ("Review Date 7/22/2026") ranked 2nd for a generic query. I filtered boilerplate sections at chunk time. *Always look at what was retrieved, not just the final answer.*

## Stage 2: an agent loop, written by hand

Tools (`tools.py`): `search_guidelines`, `calculate_tdee` (Mifflin-St Jeor × activity factor), `deficit_estimate` (≈7,700 kcal/kg rule of thumb, labelled approximate), `lookup_nutrition` (USDA FoodData Central).

The loop (`agent.py`) is about 40 lines:

1. send `messages + tool schemas` to the model
2. if the reply has no tool calls, return it as the final answer
3. otherwise, for each tool call (there can be several per turn): **validate arguments → execute → catch errors → append the result as a `tool` message**
4. repeat, with an iteration cap (6)

![Agent tool loop](docs/img/02_agent_tool_loop.png)

**Design choices worth remembering**
- *Validation returns errors to the model as data* (`{"error": "'tdee' must be a number"}`) so it can self-correct, and never crashes the loop.
- *A tool-level guardrail* (`deficit_estimate` refuses intake < 1,200 kcal) holds even if the prompt is bypassed.
- *The LLM never does arithmetic.* In the eval, the TDEE questions (M1–M4, M7) were each answered with the code's exact number (2124, 2196, 1807, 0.45 kg/week, 2672).

### Failure modes I hit with a small (8B) model

![Agent failure v0](docs/img/03_agent_failure_v0.png)

| Failure | What happened | Fix |
|---|---|---|
| Dependent calls issued in parallel | Called `deficit_estimate` with `tdee="calculate_tdee"` (a placeholder for a value it hadn't computed) | Validation caught it; tightened the tool description and system prompt to call dependent tools one at a time |
| Tool call printed as text | Model emitted `{"name": "calculate_tdee", ...}` as its *answer*, so the loop thought it was done | Fallback parser for JSON-in-text tool calls |
| Announced but didn't call | "Now I'll use the deficit_estimate tool…" and stopped | Bounded "nudge": re-prompt up to twice |

These are exactly the production failure modes of agent loops: runaway loops (iteration cap), bad arguments (validation), and malformed calls (parsing and retry limits). Ollama is also not perfectly deterministic even at temperature 0: one smoke run used both tools, while the saved eval run on the same question stopped after `calculate_tdee`.

## Stage 3: the same pipeline in LangChain, and what it hides

| Raw component | LangChain equivalent |
|---|---|
| `split_sections` + `split_long(max_chars, overlap)` | `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)` |
| `SentenceTransformer` + manual normalise | `HuggingFaceEmbeddings(model_name, encode_kwargs={"normalize_embeddings": True})` |
| numpy array, `vecs @ q`, `argsort` | `Chroma` (HNSW index) with `collection_metadata={"hnsw:space": "cosine"}` |
| `top_k(k)` | `as_retriever(search_kwargs={"k": k})` |
| `build_prompt` + `chat()` | `ChatPromptTemplate \| ChatOllama \| StrOutputParser` |

**What the framework hides:** the splitters differ subtly (character vs. sentence packing), Chroma's default distance isn't cosine unless you set it, and the prompt template and retrieval scores are no longer visible by default. I got comparable accuracy with less code, but lost the ability to see why a retrieval missed. Frameworks are worth it once you can reproduce them by hand.

## Stage 4: LangGraph, and why a state machine

State (`TypedDict`): question, current query, hits, `relevant`, `retries`, `blocked`, `answer`, `path`. Nodes read and write state, and **conditional edges** decide routing: `safety → END` if unsafe, `grade → generate` if relevant or retries ≥ 2, else `rewrite → retrieve`.

![LangGraph paths](docs/img/05_langgraph_paths.png)

Observed paths over 27 questions: 11 hit the full abstain loop (`… rewrite → retrieve → grade → rewrite → retrieve → grade → generate(abstain)`), 9 went straight through, 6 stopped at `safety`, and 1 recovered via a single rewrite. The cost is latency: 34 s per question, about 2× raw RAG.

**Why LangGraph over a free-form loop:** the retry cap, the abstain path and the safety exit are *explicit, inspectable edges* rather than behaviour hoped for from a prompt. The `path` list makes every run auditable.

## Stage 5: evaluation

**27 questions**: 8 lookup, 7 multi-hop/tool, 6 not-in-corpus, 6 guardrail. Gold answers are regex-checked; tool answers use values I computed by hand from the formula.

**Metrics**
- **Answer correctness** per category (regex-graded).
- **Correct and grounded:** the answer is correct *and* a retrieved chunk from the right source contains the gold phrase.
- **recall@k** for retrieval alone: a hit means a top-k chunk comes from a gold source *and* contains a gold phrase.

### Retrieval sweep (no LLM)

![Retrieval sweep](docs/img/retrieval_sweep.png)

| chunk ≤ chars | #chunks | recall@1 | @2 | @4 | @8 |
|---|---|---|---|---|---|
| 200 | 535 | 0.7 | 0.8 | 0.8 | 0.8 |
| 400 | 262 | 0.7 | 0.8 | 0.9 | 0.9 |
| 800 | 154 | 0.8 | 0.9 | 0.9 | 0.9 |
| 1600 | 119 | 0.9 | 0.9 | 0.9 | 0.9 |

Reading it honestly: bigger chunks and bigger k both help, with diminishing returns after k = 4. But only **10 questions** have gold evidence, so one question moves recall by 0.1, and these differences are small. The ceiling of 0.9 is the BMI question, whose answer was never in the extracted text, so **no chunk size or k could fix it**. That is a data-extraction problem, not a retrieval one. Small chunks (200 chars) lose context; large chunks (1600) win here only because these pages have short sections.

### Findings

**1. Correct but ungrounded.** The BMI answer is right and cites `[1]`, but none of the four retrieved chunks contain the cut-off (the one hit containing "30" is `30.7%`, a prevalence statistic).

![Ungrounded citation](docs/img/06_ungrounded_citation.png)

**2. The agent mixes in outside knowledge.** On "Is fat heavier than muscle?" and "glycemic index of white rice?" (neither is in the corpus) it answered from memory. On the activity question it attributed a figure to "WHO", which isn't in the corpus. Tools make the math reliable but give the model more room to wander, and the abstain rule needs enforcement in code, not only in the prompt.

![Agent knowledge leak](docs/img/07_agent_knowledge_leak.png)

**3. Pipelines without tools can't do the math, by design.** 0/7 on tool questions for raw RAG, LangChain and LangGraph is the *expected* result: they correctly abstain or say they lack the data, and it shows why "math belongs in code".

**4. Retrieval got a not-in-corpus question wrong before the LLM could help.** "Is fat heavier than muscle?" retrieves BMI text, because it is topically near, and raw RAG then answered about BMI instead of abstaining. LangChain and LangGraph abstained correctly on the same question.

## Guardrails

Design: **defence in depth**, so no single layer is trusted.

1. **Rule layer** (`safety.py`): regexes for very-low-calorie targets, disordered-eating language, pregnancy/breastfeeding, medical conditions/medications, minors, and symptoms of harmful restriction. It runs *before* retrieval or any LLM call, so it is fast, deterministic and testable.
2. **LLM classifier layer** (LangGraph only): an `UNSAFE/SAFE` check for cases the rules miss.
3. **Prompt layer:** "education only, no personalised plans, say *not in my sources* when unsure."
4. **Tool layer:** `deficit_estimate` rejects intake < 1,200 kcal regardless of the prompt.

![Guardrails](docs/img/04_guardrails.png)

**What the eval showed (v1 → v2).** In v1 my rules missed two prompts: *"my 12 year old daughter…"* (the pattern only matched "my daughter") and *"skipping meals for days and feel dizzy"* (no keyword). Three pipelines scored 4/6; **LangGraph scored 6/6 because its LLM layer caught both.** I then added the two rules (v2 → 6/6 everywhere).

> ⚠️ **Honesty note:** the v2 fixes were written *after* seeing the test questions, so v2's 6/6 is **tuned to this test set** and overstates real-world robustness. The v1 column is the fairer estimate of generalisation. Regex guardrails also over-refuse (e.g. any mention of "medication") by design: on a health topic, a false refusal is cheaper than a harmful answer. I did not measure the false-positive rate on a benign set, which is a gap.

## Part 2: chained questions, missing inputs, and a calculator route

**The problem.** A user asks *"If my height is 170 cm and weight is 70 kg, what is my TDEE, and if I eat 2,400 calories will I lose weight?"* This needs two chained tool calls, and the question never says the user's age, sex or activity level. When I tried it on the Part 1 agent, it silently invented `age=30, male, moderate` and answered confidently. The same thing happened on a held-out question that gave only "I'm a 45 year old woman": the baseline agent made up a weight, height and activity level and returned a TDEE of 1,930 kcal.

![Invented inputs vs ask-first](docs/img/08_part2_invented_inputs.png)

### What I built

| Change | Where | Idea |
|---|---|---|
| **Deterministic slot extraction** (age, sex, weight, height, activity, intake, TDEE; handles kg/lb, cm/ft-in) | `src/inputs.py` | No LLM involved, so what the user *stated* is auditable. Unmatched means `None`, and `None` means ask. |
| **Ask-before-assume gate** | `check_inputs` node; `guard_inputs` in the agent | Missing age, sex, weight or height → return a question instead of an answer. Enforced in code, not in the prompt. |
| **Assume-but-disclose for low-stakes fields** | `src/calc.py` | Missing activity level → assume sedentary, label it **ASSUMED**, and show TDEE for every activity level and whether the lose/gain direction flips. |
| **Calculator route in LangGraph** | `src/graph_tools.py` | `plan → check_inputs → calc → (retrieve → grade → generate)? → compose`. Code runs the calculators and composes the numbers; the LLM is only used for the guideline-facts part. |
| **Grounded tool args** for the agent | `tools.calculate_tdee_grounded` | Tool rejects any age/sex/weight/height the user never stated, so the model can't smuggle in invented values. |
| **Required-tools checklist** for the agent | `agent.py` | Code knows which tools a question needs and nudges (up to 3×) until they've succeeded. |
| **Assumption disclosure enforced in code** | `agent.py` | If a tool result carries an `assumption` and the model's answer omits it, the note is appended programmatically. |
| **Multi-turn** | `GraphTools.answer(question, history)` | Slots are extracted from the whole conversation, so a follow-up fills in the gaps. |

![Two-turn conversation](docs/img/10_part2_two_turn.png)

![Assumption disclosed with range](docs/img/09_part2_assumption_range.png)

*Measured side effect:* a calculator-only question (e.g. the TDEE + deficit question) is now answered by the graph in about **0.1 s** with no LLM call, versus roughly 30-100 s for the free-form agent on this laptop.

### How I evaluated it (and how I tried not to fool myself)

1. **Dev set (6 questions) + the original 27** were used to find bugs and tune. Results below marked *dev* were tuned on.
2. **Held-out set (15 questions)** was written *before* the Part 2 code. The code was then frozen (SHA-256 of every source file in `results/part2_frozen_hashes.txt`) and the held-out set was run **once**. I did not change anything afterwards. Categories: 5 calculation (incl. lb and ft-in units, "desk job" wording, a tight surplus of 56 kcal, TDEE + guideline-fact), 2 missing-info, 8 benign questions to catch **over-asking**.
3. Pipelines compared: the Part 1 agent and Part 1 LangGraph as **baselines**, versus the guarded agent and the new calculator-route graph.

![Held-out results](docs/img/11_part2_heldout.png)

**Held-out (n=15, run once):**

| | Calculation (5) | Asked instead of guessing (2) | Benign answered, not over-asked (8) |
|---|---|---|---|
| Raw agent (baseline) | 3/5 | **0/2** | 8/8 |
| Raw agent + guards | 2/5 | 2/2 | 8/8 |
| LangGraph (Part 1 baseline) | 0/5 | **0/2** | 6/8 |
| **LangGraph + calc route + ask gate** | **5/5** | **2/2** | 7/8 |

Over-asking on benign questions was **0 for every pipeline**.

**Dev (tuned on, so treat as an upper bound):** the baselines scored 0/4 on ask and 0/2 on assume-and-disclose. Both guarded versions scored 4/4 and 2/2. By the (conservative) silent-assumption metric, the baseline agent answered 5 of the 6 dev questions with invented inputs and no disclosure.

**Regression on the original 27 questions (dev):**

| | lookup | tool/math (7) | not-in-corpus (6) | guardrail (6) |
|---|---|---|---|---|
| Raw agent (Part 1) | 8/8 | 6/7 | 3/6 | 6/6 |
| Raw agent + guards | 8/8 | 5/7 | 4/6 | 6/6 |
| LangGraph (Part 1) | 8/8 | **0/7** | 6/6 | 6/6 |
| LangGraph + calc route | 8/8 | **6/7** | 6/6 | 6/6 |

### What worked, and what didn't

- **To answer your original question: yes, LangGraph can reach the tool questions, but only by adding tools.** Going from 0/7 to 6/7 on the math questions required the calculator node, not better retrieval or prompting. At that point the "RAG pipeline" is a router in front of a calculator, which is exactly what the framework is good for. I left raw RAG and LangChain at 0/7 as baselines.
- **Putting the router and the safety gate in code beat putting them in the prompt.** The graph asked correctly 2/2, and the guarded agent also asked 2/2 because it shares the same pre-check. The unguarded agent, told only by its prompt to be careful, asked 0/2.
- **The guarded agent did *not* clearly improve on calculation, and was slightly worse (2/5 vs 3/5 held-out; 5/7 vs 6/7 on the original set).** Failures I read by hand:
  - *H2:* the validation and nudge rules ping-ponged with the model until it hit the step limit ("I couldn't finish within the step limit."). This is the classic runaway-loop failure, produced by my own guards.
  - *H7:* it ran the calculator but left the TDEE out of its final answer.
  - *H1:* it answered 2,007 kcal vs. the 2,004-2,006 I computed from the exact unit conversion. That is a rounding miss of about 2 kcal, far inside the formula's +/-10% error, but I scored it as a miss because the rule was fixed in advance.
  - *Baseline agent H4/H5:* it invented sex, or weight and height, and answered anyway.
  More guards on a free-form 8B model add failure modes of their own.
- **The LLM-based `grade` node is a brittle gate.** It wrongly abstained on a benign held-out question (*"What weight-loss programs does NIDDK say to avoid?"*) and, on the original M6 question, it kept rejecting correct chunks even after I fixed retrieval. I stopped tuning rather than keep patching one question, so graph M6 is a documented failure. A false "I don't have that in my sources" is the failure mode to watch in any grade-and-abstain design.
- **Fixing retrieval for mixed questions:** searching with only the "facts" clause of a mixed question lost topic context (*"what initial goal do the guidelines suggest?"* retrieved the wrong section). I now merge retrieval from the facts clause and the full question with numbers stripped.

### Limits of Part 2 (read before citing the numbers)

- **The held-out set is small** (5 / 2 / 8 questions per type). One question moves a score by 20-50 points on the smaller categories. These are directional, not statistically solid.
- **The held-out set is not fully blind to the router.** I wrote those questions before the code, but I am the same author and some routing patterns (e.g. "what happens to my weight", "am I gaining or losing") match their wording. A different author's questions would be a stronger test.
- **The "silent assumption" metric undercounts.** It looks for the word "assum" anywhere in the answer, which a stray "assuming a rough rule" satisfies. By hand, the baseline agent invented inputs on 2 of the 2 held-out missing-info questions, but the metric counted only 1. The 0/2 *asked-instead-of-guessing* score is the reliable number.
- **Regex extraction is conservative.** It fails safe by asking, but phrasings it doesn't recognise (e.g. "I weigh about eleven stone") will trigger a question even when the user gave the information.
- **Sex is treated as a binary formula input** (Mifflin-St Jeor has a male/female term). I ask for it instead of inferring it, and the README doesn't claim this is adequate for everyone.
- **Asking needs a second turn.** The CLI scripts are single-shot; multi-turn works only through the `history` argument.

## What I learned (mapped to the learning goals)

1. **RAG internals.** Chunk size trades precision against context; embeddings map text to vectors where distance means meaning; a vector store is an array plus a similarity function; retrieved chunks become numbered context in the prompt.
2. **Retrieval evaluation.** recall@k isolates retrieval from generation. Without it I'd have blamed the LLM for the BMI error, when the real cause was HTML extraction.
3. **Agent loops.** Model proposes → code validates and executes → result goes back → repeat, with an iteration cap. Most real failures are malformed or premature tool calls, not "reasoning".
4. **Tool design.** Put deterministic math and lookups in code, validate inputs, return errors as data, and enforce safety limits inside the tool.
5. **What frameworks hide.** Splitter details, distance metric defaults, prompt templates and retrieval scores. They save code, not accuracy.
6. **LangGraph.** Explicit state, conditional edges and bounded retries make behaviour auditable. The price is latency.
7. **Guardrails.** Layer them, test them with adversarial phrasing that avoids obvious keywords, and don't tune on your test set without saying so.

## Limitations (what this does *not* show)

- **Small eval** (27 questions, one model, one run each, not repeated). Differences of one question are noise.
- **Regex grading** can mis-grade fluent answers. I found and fixed two grader issues: a missed abstention phrasing, and a guardrail check that was too lenient (it passed an agent answer that gave diet advice for a 12-year-old because it ended with "consult a pediatrician"; the grader now requires a block or a short refusal with no advice list). I also corrected a wrong prediction I made about the BMI question.
- **Corpus is 14 pages** and extraction dropped tables.
- **Agent retrieval isn't scored with recall@k**, since the model chooses its queries.
- **In Parts 1 stages 3-4, LangChain/LangGraph have no tools**, so that comparison covers the RAG pipeline only. Part 2 adds a calculator route to LangGraph, but not to LangChain.
- Guardrail v2 is tuned to the test set (above). The USDA lookup uses the public `DEMO_KEY`, which is rate-limited.

## Reproduce

```bash
brew install ollama && ollama serve &        # then:
ollama pull llama3.1:8b
uv venv && uv pip install -r requirements.txt
python src/fetch_sources.py                  # download corpus (not committed)
python src/rag.py "What is a realistic weight-loss goal?"
python src/agent.py
python src/graph_rag.py
python src/evaluate.py && python src/score.py eval_runs && python src/make_figures.py
```

Part 2: `python src/evaluate2.py dev`, then (code frozen) `python src/evaluate2.py heldout`, then `python src/score2.py dev|heldout` and `python src/make_figures2.py`.

Files in `results/`: `part2_scores_*.json`, `part2_frozen_hashes.txt`, `failure_agent_v0.txt`, `sweep.json`, `scores_*.json`, and the run logs.

The raw run files (`eval_runs.json`, `eval_runs_v1.json` for before the guardrail fixes, `part2_runs.json`, `part2_two_turn.json`) contain excerpts of the source pages, and I haven't confirmed those licences allow redistribution (the MedlinePlus encyclopedia pages are credited to A.D.A.M.). They stay local and are gitignored. `results/public/` has copies with the retrieved page text removed, which keep every question, answer, refusal flag, route and tool trace. Running `python src/export_public.py` regenerates them. The scoring scripts need the full local files, so to re-score from a fresh clone you'd re-run the evals first.

## Interview questions I can now answer

- **How does RAG work end to end, and where can it fail?** Chunk, embed, retrieve, prompt, generate. It fails at extraction (lost tables), chunking (context split), retrieval (near-topic misses), and generation (ungrounded or uncited answers), and each needs its own metric.
- **How do you evaluate a RAG system?** Separate retrieval (recall@k against gold evidence) from answer quality, add a *grounding* check, and include abstention and guardrail categories.
- **What is an agent loop, and what goes wrong in production?** Model-picks-tool, code-executes, repeat. Failures: bad arguments, parallel calls with dependencies, tool calls emitted as text, runaway loops, and leaking knowledge outside the sources.
- **Why LangGraph over a free-form loop?** Explicit state, bounded retries and conditional routing are inspectable and testable instead of hoped-for.
- **Framework or hand-written?** Write it by hand to learn and to debug. Use a framework when you can map every setting back to the raw version and need its integrations or tracing.

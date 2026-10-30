# Brief for Claude Code: add my "Nutrition RAG + Agent" project to my portfolio website

> **How to use:** copy this file and the folder `docs/img/` (13 PNGs) into my website repo, open the repo in VS Code with Claude Code, and say: *"Read PORTFOLIO_BRIEF.md and implement it."*
> The full technical write-up lives in the project's `README.md`. If it's available, use it as the source of truth for any detail not in this brief.

## 0. Rules for you (Claude Code)

1. **Explore first.** Inspect this website's stack, folder layout, routing, styling system, existing project pages/components, and how images and markdown are handled. **Match existing conventions** (naming, component patterns, CSS approach, dark/light mode). Do not introduce a new framework or dependency unless the site already uses it.
2. **Show me a short plan** (files you'll add or change) before editing. Make the smallest set of changes that fits the site.
3. **Don't invent claims.** Use only the numbers and statements in this brief. If something is not here, leave it out or ask.
4. **Keep the caveats.** They are part of the point of this project (see section 5). Do not remove or soften them for marketing.
5. **Don't commit, push or deploy** anything. Leave changes in the working tree for me to review.
6. Images need meaningful `alt` text (provided below), lazy loading, and responsive sizing. The page must work at phone width with no horizontal scroll, and have good contrast in both themes if the site has them.

## 1. What the project is (one paragraph)

A nutrition-**education** question-answering assistant that I built four ways to understand how RAG (retrieval-augmented generation) and AI agents work internally: (1) hand-written RAG with no framework, (2) a hand-written tool-calling agent loop, (3) a LangChain + Chroma rebuild, and (4) a LangGraph state-machine version. It answers only from 14 public NIDDK and MedlinePlus pages, cites sources, uses real code for calorie math (TDEE, deficit), refuses unsafe requests, and says "I don't have that in my sources" when the answer isn't there. Everything runs locally on a 16 GB MacBook (`llama3.1:8b` via Ollama, `all-MiniLM-L6-v2` embeddings). It is a learning project, not medical advice.

- **Stack:** Python, numpy, sentence-transformers, Ollama (llama3.1:8b), LangChain, Chroma, LangGraph, matplotlib
- **Repo:** `[ADD GITHUB URL]` (placeholder: ask me, or leave a clearly marked TODO)
- **Status:** Parts 1 and 2 are complete. Next planned steps (hybrid search + reranking, citation verification, agentic routing, tracing, prompt-strategy experiments) are **planned, not built**. Do not present them as done.

## 2. Page structure to build

Create one project page (and a card for the projects index, if the site has one) with these sections, in order. Keep copy short and scannable; use the images as the main visual evidence.

1. **Header:** title, one-line summary, stack chips, repo link, "learning project" label.
2. **Why I built it:** 2-3 sentences (section 3).
3. **What it does:** the four pipelines (small diagram or 4 short cards).
4. **Key results:** the headline table plus `accuracy_by_category.png`.
5. **Findings:** 5 short findings, each with its image (section 4).
6. **Part 2: asking instead of guessing:** the held-out table, `11_part2_heldout.png`, and the two-turn demo.
7. **Limitations (honest):** section 5, shown plainly, not hidden in a collapsed block.
8. **What I learned / what's next.**
9. **Link to the full README/write-up.**

## 3. Copy you can use (edit tone to match my site)

**One-liner:** *A nutrition Q&A assistant built four ways (raw RAG, tool-calling agent, LangChain, LangGraph) and evaluated head-to-head, to find out what frameworks actually add.*

**Why I built it:** *"RAG" and "agents" are easy to use through libraries and hard to understand. I built the pipeline by hand first, then rebuilt it with LangChain and LangGraph, and measured all four on the same questions, including questions that should be refused or answered with "not in my sources".*

**Core concepts (for a general audience, include as a short glossary or tooltip list):**
- *RAG:* search your documents first, then have the LLM answer only from what it found, with citations.
- *Embedding / vector search:* text turned into numbers so similar meanings are close together; retrieval finds the closest chunks.
- *Agent loop:* the model asks for a tool, my code runs it, the result goes back, repeat. Math runs in code, not in the model.
- *Guardrails:* rules that refuse unsafe requests (crash diets, pregnancy, eating disorders, children).
- *Grounded answer:* correct **and** the retrieved text really contained the evidence.

## 4. Findings, with images (copy from `docs/img/`)

| # | Finding | Image | Alt text |
|---|---|---|---|
| 1 | All four pipelines handled simple lookups (8/8). Only the tool-using agent could do the math (6/7); the others scored 0/7 because they have no calculator, as expected. | `accuracy_by_category.png` | Bar chart of answer correctness by category for four pipelines |
| 2 | **A right answer with a fake citation.** All four answered the BMI question correctly and cited `[1]`, but my extractor had dropped the BMI table, so the number came from model memory. This is why I added a "grounded" metric. | `06_ungrounded_citation.png` | Terminal output showing a correct BMI answer whose retrieved chunks do not contain the cut-off |
| 3 | **The agent mixes in outside knowledge.** It answered out-of-corpus questions from memory (3/6 on "not in my sources") and attributed a figure to WHO, which is not in the corpus. | `07_agent_knowledge_leak.png` | Terminal output of agent answers that use knowledge outside the sources |
| 4 | **Layered guardrails.** Keyword rules missed two unsafe prompts; LangGraph's LLM safety classifier caught both (6/6 vs 4/6 for the others on the first run). I then added the rules. | `04_guardrails.png` | Terminal output comparing guardrail results before and after fixes |
| 5 | **Small-model agent failures were real and fixable:** placeholder arguments, tool calls printed as text, announced-but-not-called tools. | `03_agent_failure_v0.png`, `02_agent_tool_loop.png` | Terminal output of an agent failure case; terminal output of a working tool loop |
| 6 | **Retrieval vs generation measured separately.** Chunk size and top-k sweep (no LLM): recall 0.7 to 0.9; the 0.9 ceiling is the BMI question, a data-extraction problem no setting could fix. | `retrieval_sweep.png` | Line chart of retrieval recall versus top-k for four chunk sizes |

Optional supporting images: `01_raw_rag_citation.png` (raw RAG retrieval scores and a cited answer), `05_langgraph_paths.png` (LangGraph routing: normal, abstain after retries, refuse at safety).

### Headline table (use exactly these numbers)

| 27 questions | Raw RAG | Raw agent | LangChain | LangGraph |
|---|---|---|---|---|
| Lookup (8) | 8/8 | 8/8 | 8/8 | 8/8 |
| Multi-hop / tool math (7) | 0/7 | 6/7 | 0/7 | 0/7 |
| Not in corpus → abstain (6) | 5/6 | 3/6 | 6/6 | 6/6 |
| Guardrails, first run (6) | 4/6 | 4/6 | 4/6 | 6/6 |
| Guardrails, after fixes (6) | 6/6 | 6/6 | 6/6 | 6/6 |

Mean seconds per question on my laptop: 18.7 / 27.7 / 17.4 / 34.4.

### Part 2 (missing information and a calculator route)

**Problem:** asked "my height is 170 cm and weight 70 kg, what's my TDEE and will 2,400 calories make me lose weight?", the original agent silently invented age, sex and activity level.

**Fix:** extract what the user actually stated with plain code (no LLM); if age, sex, weight or height is missing, **ask**; if only activity level is missing, assume sedentary, label it as assumed, and show results for every activity level. In LangGraph, a calculator node runs the TDEE/deficit math in code.

**Evaluation protocol:** 15 held-out questions written before the code, code frozen (file hashes recorded), run once.

| Held-out (n=15) | Calculation (5) | Asked instead of guessing (2) | Benign answered (8) |
|---|---|---|---|
| Agent, original | 3/5 | **0/2** | 8/8 |
| Agent + guards | 2/5 | 2/2 | 8/8 |
| LangGraph, original | 0/5 | **0/2** | 6/8 |
| **LangGraph + calc route + ask gate** | **5/5** | **2/2** | 7/8 |

On the original 27 questions, LangGraph's tool/math score went 0/7 → 6/7. A calculator-only question is answered in about 0.1 s (no LLM call) versus 30-100 s for the free-form agent. Show `11_part2_heldout.png`, `08_part2_invented_inputs.png`, `09_part2_assumption_range.png`, `10_part2_two_turn.png`.

**Notable negative result (include it):** adding guards to the free-form agent did *not* improve its math (2/5 vs 3/5), and my own rules sometimes made it loop until the step limit. More rules on a small free-form model add failure modes.

## 5. Limitations (must appear on the page, in plain view)

- Small evaluation: 27 questions, then 15 held-out; one model, one run each. Differences of one question are noise. Results are directional.
- The Part 2 held-out set is not fully blind: I wrote both the questions and the code, and some routing patterns resemble the held-out wording.
- The final guardrail score (6/6) was reached after fixing rules *using the test questions*, so the first-run numbers (4/6) are the fairer estimate.
- Answers are graded with regexes, which can misjudge fluent answers. I fixed two grader issues I found.
- The corpus is 14 pages and my extractor dropped HTML tables.
- LLM "grade" step is brittle: it sometimes wrongly says "not in my sources" (one benign held-out question and one original question).
- This is an educational demo. It is not medical advice and gives no personalised diet plans.

## 6. Honest attribution line (include; I can edit the wording)

*Built with Claude Code (Anthropic's coding agent) as a pair-programmer: it wrote the code and ran the experiments, and I'm studying the code and results to learn them.*

## 7. Also produce

1. **A short project-card blurb** (max 30 words) for the projects index.
2. **Three resume bullets** (past tense, quantified, no inflated claims), for example in this style:
   - *Built a nutrition Q&A system four ways (hand-written RAG, tool-calling agent, LangChain, LangGraph) on a local 8B model; evaluated all four on 27 questions across lookup, math, abstention and safety categories.*
   - *Found that all pipelines gave correct but ungrounded answers when a source table was lost in extraction; added a grounding metric to catch it.*
   - *Raised LangGraph from 0/7 to 6/7 on calculation questions and 0/2 to 2/2 on asking for missing inputs using a code-level calculator route and ask-before-assume gate, evaluated on a held-out set run once with frozen code.*
3. A short **"what's next"** list stating clearly that these are *planned*: hybrid search (BM25 + vectors) with reranking, citation verification, multi-hop retrieval, agentic routing, tracing, prompt-strategy experiments.

## 8. Done when

- The page builds and renders correctly on desktop and phone widths, in light and dark themes (if the site has both).
- All 13 images load with alt text; no layout shift from images.
- The limitations section is visible without clicking.
- No claims, numbers or features beyond this brief.
- I get a short summary of every file you added or changed, and nothing is committed.

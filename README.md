# Learning Projects

Small projects I build to understand how ML and AI systems actually work, instead of only using the libraries. Each project is self-contained in its own folder, with its own README, requirements and results, and each one is measured (not just demoed), including the parts that didn't work.

Most of the code was written with help from Claude Code. The experiments, results and write-ups are mine to understand and explain, and the individual READMEs say where that matters.

## Projects

### [`ml-mnist-from-scratch/`](ml-mnist-from-scratch): neural network in plain numpy
A two-layer network (no PyTorch or TensorFlow) trained on MNIST, to learn backpropagation, overfitting, regularisation and distribution shift.

- 97.6% MNIST test accuracy; hand-written backprop checked against finite differences
- Overfitting demo: 100% train vs 86% validation on 500 images
- Distorted digits: 57.4% → 92.0% with augmentation
- Also tested on my own handwriting, with a preprocessing ablation

**Stack:** Python, numpy, matplotlib

### [`rag-nutritionist/`](rag-nutritionist): nutrition Q&A built four ways
A nutrition-education assistant over 14 public NIDDK/MedlinePlus pages, built as hand-written RAG, a hand-written tool-calling agent, a LangChain + Chroma version, and a LangGraph version. It runs locally (`llama3.1:8b` via Ollama) and is evaluated on 27 questions plus a held-out set that was run once with the code frozen.

- Found a "correct but ungrounded" failure: right answer, citation that didn't support it
- Asking for missing inputs instead of guessing: 0/2 → 2/2 on held-out questions
- Honest limitations section: small eval, tuning on test questions, a guard that made the agent worse

**Stack:** Python, numpy, sentence-transformers, Ollama, LangChain, Chroma, LangGraph

**Planned next:** hybrid search and reranking, citation verification, multi-hop retrieval, agentic routing, tracing. (Not built yet.)

## Notes
- These are learning projects, not production systems. The sample sizes are small, so treat the numbers as indicative.
- `rag-nutritionist` does not include the source pages or the raw run files that contain excerpts of them (licences not confirmed); its README explains how to re-download and regenerate.

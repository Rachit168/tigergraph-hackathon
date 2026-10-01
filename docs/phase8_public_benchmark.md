# Phase 8: Public three-way benchmark

Compare the existing RAG, Fixed GraphRAG, and Agentic GraphRAG systems on
the 100 public questions. This phase does not change those systems.

```
python -m scripts.eval_three_way --generator semantic
```

Artifacts are written locally to `_research/phase8_public_three_way/` and are
not part of the frozen source tree. The published public baseline is the Phase 10
semantic recheck (RAG ~67%, GraphRAG ~98%, Agentic ~97%).

The runner is read-only against OlympicGraph. Gold answers are used only after
each system has produced an output. The Phase 6 Agentic placeholder is rejected.

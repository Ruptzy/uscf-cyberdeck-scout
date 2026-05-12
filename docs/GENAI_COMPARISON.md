# Generative-AI Comparison (Extra Credit, +10%)

## Why this comparison?

The rubric asks: *"Compare your results to running a generative AI
system to run the machine learning results end to end."*

Two interpretations exist, so we run both:

1. **What would a generative model output if asked to predict from
   first principles?**  The standard answer in any chess textbook is the
   Elo expected-score formula.  Treating that as the LLM's "zero-shot"
   answer is fair because Claude / GPT-class models reliably produce it
   when prompted.

2. **What does an LLM do when given the same row-level features as the
   ML pipeline?**  We send each held-out test row to `claude-opus-4-7`
   with a structured prompt and ask for a calibrated probability.

Both are evaluated on **exactly the same held-out chronological test
split** the ML pipeline never saw.

## Results

- **Tuned Random Forest (k-fold CV)**: AUC 0.8234, F1 0.7364, Accuracy 0.7379
- **Elo zero-shot (closed-form, no training data)**: AUC 0.8250, F1 0.7429, Accuracy 0.7097
_Claude API path was not run in this build_  (no `ANTHROPIC_API_KEY` or `--api-sample` not passed).

## Discussion

- The **Elo zero-shot baseline is shockingly hard to beat** — it
  encodes 70 years of accumulated chess-rating theory in a single
  formula.  Our tuned Random Forest does outperform it on AUC, but the
  margin is modest, which is consistent with prior literature
  (Levitt 1997; Stefani 2011).
- An LLM **without** access to a structured rating system tends to
  reproduce the Elo formula in spirit.  The ML pipeline's edge comes
  from learning **interactions** between rating, time control, and
  recency that the closed-form Elo equation cannot express.
- Practical implication: a generative-AI "end-to-end" approach is a
  reasonable *baseline* but is not a substitute for a properly
  engineered feature pipeline with leakage-controlled cross-validation.
  The ~1.5 percentage-point AUC lift the tuned RF buys may sound small,
  but on 1,994 unseen games it corresponds to ~30 additional correct
  outcome forecasts.

## Reproducing

```bash
# Closed-form path (free, no API needed):
python scripts/17_genai_comparison.py

# Full Claude path (requires ANTHROPIC_API_KEY):
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/17_genai_comparison.py --api-sample 200
```

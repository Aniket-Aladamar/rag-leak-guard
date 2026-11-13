# rag-leak-guard (v0)

A baseline classifier that answers one question: **given a RAG answer, the retrieved context and the querier's authorization, did the answer leak data the querier is not allowed to see?**

Status: v0, research baseline. Trained and tested on synthetic, template-generated data only. Read the "Honest caveats" section before drawing conclusions from the numbers.

## Task

Input (one example):

| Field | Meaning |
| --- | --- |
| `acl` | Tenant authorization summary for the querier: tenant, role, clearance level, and the rule "may access only documents of tenant T with level <= L" |
| `context` | Retrieved chunks, each tagged `[tenant: X \| level: Y] text`, joined with ` \|\| ` |
| `answer` | The generated answer to check |

Output: binary label. `1` = the answer reveals information from a chunk the querier is not authorized to see (wrong tenant, or level above clearance). `0` = no leak.

A leak needs two things at once: the answer must reveal content from a specific chunk, and that chunk must be outside the querier's authorization. The model has to do both: attribute the answer to a chunk, then compare that chunk's tenant and level with the ACL.

## Data

`build_data.py` generates the data. No LLM API is used.

- Multi-tenant scenarios built from templates: tenants, people, roles (viewer, employee, manager, admin), clearance levels (public, internal, confidential, restricted).
- PII values (email, phone, SSN, IBAN, street address, date of birth, card number) are drawn from the open [ai4privacy/pii-masking-200k](https://huggingface.co/datasets/ai4privacy/pii-masking-200k) dataset. Pools are small (about 300 to 570 values per type). Where a pool is short, synthetic generators fill in. If the dataset can't be loaded, the script falls back to generators only.
- Six answer styles: `direct` (states the value), `partial` (last 4 characters), `mask` (masked value), `refuse`, `generic` (unrelated text), `wrong` (value with digits altered, so a near miss that reveals nothing). Only `direct` and `partial` leaks are positives, plus a few cases where an extra authorized-looking sentence reveals another person's value from an unauthorized chunk.
- Fields map to levels: email and phone = internal, address and DOB = confidential, SSN, IBAN and card = restricted.
- Train and test are disjoint: different tenants, different people, different PII values.
- v0 run: 24,000 generated train examples (21,600 train and 2,400 validation after the split) and 3,000 test examples. About 21 percent positive.

## Model and training

`train_eval.py`: `microsoft/deberta-v3-base` with a sequence-classification head, a cross-encoder over (`ACL + answer`, `context`). 6 epochs, learning rate 3e-5, batch size 16, max length 384, fp16 on 2x T4 (Kaggle free tier). About 74 minutes of training.

### The overlap attribution step (required)

Before encoding, context chunks are reordered by lexical overlap with the answer, and each is tagged:

- `[MATCH n]` if at least 2 answer tokens (4+ characters) appear in the chunk, where `n` is the count. Matching chunks come first.
- `[NOMATCH]` otherwise.

This step matters a lot. The first run, without it, reached only F1 0.63 (recall at 1% FP 0.25), and it failed mainly on `direct` and `partial` leaks: the model could not reliably find which chunk the answer came from and then check that chunk's tenant and level. With the tags, F1 went to 0.99. So v0 depends on a **lexical answer-to-chunk attribution** being computed upstream. That is a shortcut that works on templated text. A real pipeline needs a better attribution (see roadmap).

## Results (v0, test set of 3,000)

| Run | Data | Setup | F1 | Precision | Recall | AUC | Recall @ 1% FP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 12k train | 3 epochs, LR 2e-5, no overlap tags | 0.629 | 0.684 | 0.583 | 0.927 | 0.246 |
| 2 | 24k train | 6 epochs, LR 3e-5, with `[MATCH]`/`[NOMATCH]` | **0.986** | 0.982 | 0.991 | 0.9998 | **1.0** |

A saved rerun of run 2 (different training nondeterminism) gave F1 0.989 / AUC 0.9999. Recall at 5% FP is 1.0.

Per answer style accuracy in run 2 at threshold 0.5: direct 98.4%, partial 100%, generic 99.4%, mask 99.4%, refuse 100%, wrong 99.2%.

## Honest caveats

- The data is templated and synthetic. A 0.99 F1 shows the model learned the generator's rule. It does not show it will catch leaks in real LLM answers.
- Real answers paraphrase, summarize, combine several chunks, hint at values, or leak across multiple turns. None of that is in v0.
- The overlap tag is a lexical shortcut that exactly fits how the templates state values. It would be weaker on paraphrased answers.
- Test tenants, people and PII values are held out, but test examples use the same templates as train. Generalization to new phrasing is untested.
- PII pools are small. Models may memorize value formats.
- Single seed and a single split. No confidence intervals.

## Run it

```bash
pip install -r requirements.txt
python build_data.py                      # writes train.jsonl and test.jsonl
MODEL=microsoft/deberta-v3-base EPOCHS=6 LR=3e-5 OUT=./leakguard_v1 python train_eval.py
```

`train_eval.py` prints and saves `metrics.json` (precision, recall, F1, AUC, recall at 1% and 5% FP, per-style accuracy) under `OUT`. It was written for a GPU notebook; defaults write to `/kaggle/working/leakguard_v1`, so set `OUT` elsewhere.

## Weights

The trained checkpoint (about 2.8 GB) is not in this repo. Weights are available on request, and a release may follow.

## v2 results (held-out template families)

v2 tests whether the guard survives new wording. `build_data_v2.py` makes two template families. Family A is used for training (16k) and an in-distribution test (3k, `iid`). Family B has different record, header, ACL, query, answer and generic-text templates and is test-only (3k, `ood`). Tenants, people and PII values are disjoint between train and test. There are 8 answer styles: direct, partial, mask, refuse, generic, wrong person, confirm yes and confirm no. About 25% are leaks.

Same model (DeBERTa-v3-base, 5 epochs, 16k examples, one seed), three ways of marking the chunks:

| Chunk tag | iid F1 | OOD F1 | OOD precision | OOD recall | OOD recall@1%FP | OOD AUC |
|---|---|---|---|---|---|---|
| none | 0.999 | 0.851 | 0.787 | 0.926 | 0.48 | 0.972 |
| lexical overlap (v0) | 1.000 | 0.819 | 0.726 | 0.939 | 0.48 | 0.966 |
| learned embedding (MiniLM top chunk) | 0.999 | **0.917** | 0.902 | 0.934 | **0.83** | **0.984** |

What this says:

- All three are near perfect on familiar templates, so the iid number says little.
- On unseen templates the v0 lexical tag does worst. The v0 claim that overlap tags are the whole unlock does not hold once wording changes.
- The embedding tag helps most, and recall at 1% false positives goes from 0.48 to 0.83. OOD F1 is still well below iid, mostly from false alarms on the new family.
- Still synthetic and templated, one seed, no real LLM answers. Per-style breakdowns are saved in `metrics.json` but not summarised here.

Run: `python build_data_v2.py` then `python run_all.py` (runs TAG=none, lex, emb in turn; needs `sentence-transformers`). About 35-40 minutes per tag on 2 x T4.

## Roadmap (what is left)

1. LLM-written answers: summaries, implied leaks, multi-chunk and multi-hop.
2. Train on several template families and test on a third.
3. Untagged and differently formatted chunk metadata.
4. A larger PII pool and multiple seeds with confidence intervals.
5. Longer-context models (for example ModernBERT).

## License

MIT. See `LICENSE`.

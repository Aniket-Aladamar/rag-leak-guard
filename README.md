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


## Results: v2 and v3 (held-out template families)

v2 and v3 test generalization: the OOD test set is generated from template
families the model never saw in training. v2 trained on family A and tested on
family B. v3 trains on families A+B (32,000 examples) and tests on a harder
held-out family C (JSON/pipe-row records, metadata-suffix and HTML-comment
headers, policy-memo ACL phrasing, new query/answer templates).
Model: microsoft/deberta-v3-base, 5 epochs. v3 ran on Google Colab (T4) with
transformers 4.56.2 / huggingface-hub 0.34.4 / datasets 3.6.0 (Colab's default
transformers 5.x cannot load the deberta-v3 tokenizer).

| OOD metric | v2 emb (family B) | v3 emb (family C) | v3 no-tag (family C) |
| --- | --- | --- | --- |
| Precision | - | 0.9934 | 0.7687 |
| Recall | - | 0.7762 | 0.9457 |
| F1 | 0.9173 | 0.8715 | 0.8480 |
| AUC | 0.9844 | 0.9950 | 0.9707 |
| recall@1%FP | 0.8253 | 0.8913 | 0.5058 |
| recall@5%FP | - | 0.9793 | 0.8021 |

IID (same families as training): both tags F1 ~0.999, recall@1%FP 1.0.

Family C is harder than family B, so v3-vs-v2 OOD numbers are not directly
comparable. The decision-relevant pattern replicates and widens in v3:
embedding-based answer-to-chunk tags beat no tags on every OOD metric that
matters (recall@1%FP gap +0.39, was +0.34 in v2), and the wins concentrate on
the value-revealing answer styles (direct, partial, confirm_yes). See
RESULTS_v3.md for the full table and per-style breakdown.

Caveats still apply: synthetic data, single seed.

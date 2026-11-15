# v3 results - held-out third template family (Colab run, Nov 15 2026)

Train: template families A+B combined (32,000 examples). IID test: A+B (3,000).
OOD test: held-out family C (3,000) with JSON/pipe-row records, metadata-suffix and
HTML-comment chunk headers, policy-memo ACL phrasing, and new query/answer templates.
Model: microsoft/deberta-v3-base, 5 epochs, lr 3e-5, maxlen 384, bs 16, seed 13.
Run on Google Colab free tier (T4) with transformers 4.56.2 / huggingface-hub 0.34.4 /
datasets 3.6.0 (Colab's default transformers 5.x cannot load the deberta-v3 tokenizer).

| Metric (OOD, family C) | emb tags | no tags |
| --- | --- | --- |
| Precision | 0.9934 | 0.7687 |
| Recall | 0.7762 | 0.9457 |
| F1 | 0.8715 | 0.8480 |
| AUC | 0.9950 | 0.9707 |
| recall@1%FP | 0.8913 | 0.5058 |
| recall@5%FP | 0.9793 | 0.8021 |

IID: both tags F1 ~0.999, r@1%FP 1.0.

v2 reference (OOD = easier family B): emb F1 0.9173 / AUC 0.9844 / r@1%FP 0.8253;
none F1 0.8506 / AUC 0.9725 / r@1%FP 0.4834. Family C is harder, so v3-vs-v2 OOD
numbers are not directly comparable; within v3 the embedding-tag advantage over the
no-tag baseline replicates and widens (r@1%FP gap +0.39 vs +0.34 in v2).

OOD accuracy@0.5 by answer style (emb vs none): direct 0.883/0.779,
partial 0.928/0.794, confirm_yes 0.889/0.819, confirm_no 0.952/0.980,
mask 0.980/0.980, refuse 0.975/0.975, generic 0.956/0.975, wrong 0.955/0.975.
Embedding tags win exactly on the value-revealing styles (direct, partial, confirm_yes).

Caveats: synthetic data, single seed.

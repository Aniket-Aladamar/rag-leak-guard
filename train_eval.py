import os, json, numpy as np, torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset

MODEL = os.environ.get("MODEL", "microsoft/deberta-v3-base")
EPOCHS = float(os.environ.get("EPOCHS", 3)); LR = float(os.environ.get("LR", 2e-5))
MAXLEN = int(os.environ.get("MAXLEN", 384)); BS = int(os.environ.get("BS", 16)); OUT = os.environ.get("OUT", "/kaggle/working/leakguard_v1")
rd = lambda p: [json.loads(l) for l in open(p)]
tr, te = rd("train.jsonl"), rd("test.jsonl")
rng = np.random.RandomState(0); idx = rng.permutation(len(tr)); nv = len(tr) // 10
va = [tr[i] for i in idx[:nv]]; tr = [tr[i] for i in idx[nv:]]
tok = AutoTokenizer.from_pretrained(MODEL)

import re
def reorder(ans, ctx):
    toks = [w for w in re.findall(r"[\w@.+*-]{4,}", ans)]
    cs = ctx.split(" || ")
    sc = [sum(1 for w in toks if w in c) for c in cs]
    order = sorted(range(len(cs)), key=lambda i: -sc[i])
    return " || ".join((f"[MATCH {sc[i]}] " if sc[i] >= 2 else "[NOMATCH] ") + cs[i] for i in order)
def enc(batch):
    return tok([f"ACL: {a} ANSWER: {b}" for a, b in zip(batch["acl"], batch["answer"])], [f"CONTEXT: {reorder(b, c)}" for b, c in zip(batch["answer"], batch["context"])], truncation="only_second", max_length=MAXLEN)
mk = lambda d: Dataset.from_list([{"acl": x["acl"], "answer": x["answer"], "context": x["context"], "labels": x["label"]} for x in d]).map(enc, batched=True, remove_columns=["acl", "answer", "context"])
dtr, dva, dte = mk(tr), mk(va), mk(te)

def recall_at_fpr(y, s, t=0.01):
    fpr, tpr, _ = roc_curve(y, s); return float(tpr[fpr <= t].max()) if (fpr <= t).any() else 0.0
def metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, (p > 0.5).astype(int), average="binary", zero_division=0)
    return {"precision": pr, "recall": rc, "f1": f1, "auc": float(roc_auc_score(y, p)), "recall@1%FP": recall_at_fpr(y, p), "recall@5%FP": recall_at_fpr(y, p, 0.05)}
def cm(ep):
    lg, y = ep; p = torch.softmax(torch.tensor(lg), -1)[:, 1].numpy(); return metrics(y, p)

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2, dtype=torch.float32)
args = TrainingArguments(output_dir=OUT, num_train_epochs=EPOCHS, learning_rate=LR, per_device_train_batch_size=BS, per_device_eval_batch_size=64,
    eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="f1", warmup_steps=100, weight_decay=0.01,
    fp16=torch.cuda.is_available(), report_to="none", save_total_limit=1, logging_steps=100, disable_tqdm=True)
t = Trainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva, data_collator=DataCollatorWithPadding(tok), compute_metrics=cm)
t.train()
res = {"val": t.evaluate(dva)}
out = t.predict(dte); p = torch.softmax(torch.tensor(out.predictions), -1)[:, 1].numpy(); y = np.array([x["label"] for x in te])
res["test"] = metrics(y, p); res["test_by_style"] = {}
for s in sorted({x["style"] for x in te}):
    m = np.array([x["style"] == s for x in te]); res["test_by_style"][s] = {"n": int(m.sum()), "pos": int(y[m].sum()), "acc@0.5": float(((p[m] > .5) == y[m]).mean())}
res["config"] = {"model": MODEL, "epochs": EPOCHS, "lr": LR, "maxlen": MAXLEN, "bs": BS, "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
os.makedirs(OUT, exist_ok=True); json.dump(res, open(f"{OUT}/metrics.json", "w"), indent=1, default=float)
t.save_model(f"{OUT}/best"); tok.save_pretrained(f"{OUT}/best")
print(json.dumps(res, indent=1, default=float))

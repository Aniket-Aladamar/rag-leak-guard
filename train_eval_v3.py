import os, re, json, numpy as np, torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, set_seed
from datasets import Dataset

MODEL = os.environ.get("MODEL", "microsoft/deberta-v3-base")
TAG = os.environ.get("TAG", "lex")            # none | lex | emb
EPOCHS = float(os.environ.get("EPOCHS", 5)); LR = float(os.environ.get("LR", 3e-5))
MAXLEN = int(os.environ.get("MAXLEN", 384)); BS = int(os.environ.get("BS", 16))
NTR = int(os.environ.get("N_TRAIN", 0)); NTE = int(os.environ.get("N_TEST", 0))
OUT = os.environ.get("OUT", f"/kaggle/working/v3/{TAG}"); SAVE = os.environ.get("SAVE", "0") == "1"
set_seed(13)
rd = lambda p, n=0: [json.loads(l) for l in open(p)][: (n or None)]
tr, iid, ood = rd("v3_train.jsonl", NTR), rd("v3_test_iid.jsonl", NTE), rd("v3_test_ood.jsonl", NTE)
rng = np.random.RandomState(0); idx = rng.permutation(len(tr)); nv = max(len(tr) // 10, 1)
va = [tr[i] for i in idx[:nv]]; tr = [tr[i] for i in idx[nv:]]
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)

def lex_tag(x):
    toks = re.findall(r"[\w@.+*-]{4,}", x["answer"] + " " + x["query"])
    cs = x["context"].split(" || "); sc = [sum(1 for w in toks if w in c) for c in cs]
    order = sorted(range(len(cs)), key=lambda i: -sc[i])
    return " || ".join((f"[MATCH {sc[i]}] " if sc[i] >= 2 else "[NOMATCH] ") + cs[i] for i in order)

EMB = None
def emb_tag_all(rows):
    global EMB
    from sentence_transformers import SentenceTransformer
    if EMB is None: EMB = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")
    q = EMB.encode([x["query"] + " " + x["answer"] for x in rows], batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
    flat = [c for x in rows for c in x["context"].split(" || ")]
    ce = EMB.encode(flat, batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
    k = 0; out = []
    for i, x in enumerate(rows):
        cs = x["context"].split(" || "); s = ce[k:k + len(cs)] @ q[i]; k += len(cs); top = int(np.argmax(s))
        order = [top] + [j for j in range(len(cs)) if j != top]
        out.append(" || ".join(("[TOP] " if j == top else "[REST] ") + cs[j] for j in order))
    return out

def tagged(rows):
    if TAG == "lex": return [lex_tag(x) for x in rows]
    if TAG == "emb": return emb_tag_all(rows)
    return [x["context"] for x in rows]

def enc(b):
    return tok([f"ACL: {a} QUERY: {q} ANSWER: {s}" for a, q, s in zip(b["acl"], b["query"], b["answer"])], [f"CONTEXT: {c}" for c in b["ctx"]], truncation="only_second", max_length=MAXLEN)
def mk(rows):
    t = tagged(rows)
    return Dataset.from_list([{"acl": x["acl"], "query": x["query"], "answer": x["answer"], "ctx": c, "labels": x["label"]} for x, c in zip(rows, t)]).map(enc, batched=True, remove_columns=["acl", "query", "answer", "ctx"])
dtr, dva, d_iid, d_ood = mk(tr), mk(va), mk(iid), mk(ood)
print("example tagged input:", f"ACL: {tr[0]['acl']} QUERY: {tr[0]['query']} ANSWER: {tr[0]['answer']}", "| CONTEXT:", tagged(tr[:1])[0], flush=True)
lens = [len(x) for x in dtr["input_ids"][:2000]]; print("token len mean/max/frac>maxlen-1:", np.mean(lens), max(lens), np.mean([l >= MAXLEN for l in lens]), flush=True)

def recall_at_fpr(y, s, t=0.01):
    fpr, tpr, _ = roc_curve(y, s); return float(tpr[fpr <= t].max()) if (fpr <= t).any() else 0.0
def metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, (p > 0.5).astype(int), average="binary", zero_division=0)
    return {"precision": pr, "recall": rc, "f1": f1, "auc": float(roc_auc_score(y, p)), "recall@1%FP": recall_at_fpr(y, p), "recall@5%FP": recall_at_fpr(y, p, 0.05)}
def cm(ep):
    lg, y = ep; return metrics(y, torch.softmax(torch.tensor(lg), -1)[:, 1].numpy())

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2, dtype=torch.float32)
args = TrainingArguments(output_dir=OUT, num_train_epochs=EPOCHS, learning_rate=LR, per_device_train_batch_size=BS, per_device_eval_batch_size=64,
    eval_strategy="epoch", save_strategy="no", warmup_steps=100, weight_decay=0.01, fp16=torch.cuda.is_available(), report_to="none", logging_steps=100, disable_tqdm=True)
t = Trainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva, data_collator=DataCollatorWithPadding(tok), compute_metrics=cm)
t.train()
res = {"tag": TAG, "val": t.evaluate(dva)}
for name, d, rows in (("test_iid", d_iid, iid), ("test_ood", d_ood, ood)):
    out = t.predict(d); p = torch.softmax(torch.tensor(out.predictions), -1)[:, 1].numpy(); y = np.array([x["label"] for x in rows])
    res[name] = metrics(y, p); res[name + "_by_style"] = {}
    for s in sorted({x["style"] for x in rows}):
        m = np.array([x["style"] == s for x in rows]); res[name + "_by_style"][s] = {"n": int(m.sum()), "pos": int(y[m].sum()), "acc@0.5": float(((p[m] > .5) == y[m]).mean())}
res["config"] = {"model": MODEL, "tag": TAG, "epochs": EPOCHS, "lr": LR, "maxlen": MAXLEN, "bs": BS, "n_train": len(tr), "n_val": len(va), "n_test": len(iid)}
os.makedirs(OUT, exist_ok=True); json.dump(res, open(f"{OUT}/metrics.json", "w"), indent=1, default=float)
if SAVE: t.save_model(f"{OUT}/model"); tok.save_pretrained(f"{OUT}/model")
print("RESULT", json.dumps({k: res[k] for k in ("tag", "test_iid", "test_ood")}, default=float), flush=True)

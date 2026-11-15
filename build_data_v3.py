"""v3 data generator: train on template families A+B combined, test on a held-out third family C.
Family C uses a distinct register (JSON/TSV-style records, metadata-suffix and comment headers,
policy-memo ACL phrasing, new query and answer templates). Tenants, people and PII values stay
disjoint between train and both tests (same split scheme and SEED as v2)."""
import random, json, re, os
from build_data import (load_pii_pool, FIELD_LEVEL, FIELD_NOUN, TENANT_WORDS_A, LEVELS, ROLES, mk_tenants, mk_people)
import build_data_v2 as v2
SEED = int(os.environ.get("SEED", 21))

C = {}
C["rec"] = ['{{"name": "{name}", "{noun}": "{val}"}}',
    "row {k} | {name} | {noun} | {val}",
    "- {name} :: {noun} -> {val}",
    "export: {name}; field={noun}; value={val}",
    "db: {name}. set {noun} = {val}.",
    "{name} | {noun} | {val} | verified"]
C["hdr"] = ["classification={l}; org={t} :: {x}", "{x} [metadata: org={t}, class={l}]", "<!-- org: {t}, clearance: {l} --> {x}"]
C["acl"] = ["Access grant: organization {t}, position {r}. Maximum sensitivity permitted: {l}. Documents outside organization {t} or above {l} are prohibited.",
    "Policy check - subject belongs to {t} ({r}). Disclosure ceiling: {l} within {t} only.",
    "Authorization record: the requester is a {r} with org {t} and may read {t} documents up to {l}. No cross-org access."]
C["query"] = ["What's on file for {name}'s {noun}?", "Fetch the {noun} for {name}.", "Query: {noun} of {name}.", "Tell me the {noun} belonging to {name}.", "Anything on {name}?"]
C["generic"] = ["The changelog for this release mentions faster indexing and several bug fixes.", "Access reviews are conducted twice a year by the compliance team.", "Laptops are refreshed every three years under the hardware policy.", "The cafeteria menu is posted every Monday morning.", "Visitors must sign in at reception and wear a badge."]
C["direct"] = ["According to the database, the {n} for {p} is {v}.", "Result: {p} - {n} = {v}.", "{p}'s {n}: {v}. Retrieved just now."]
C["partial"] = ["For privacy, only the last part of {p}'s {n} is shown: ...{t}.", "Ending in {t} ({p}'s {n})."]
C["mask"] = ["{p}'s {n}: {m} (redacted).", "Masked value for {p}'s {n}: {m}."]
C["refuse"] = ["Request denied: insufficient privileges to view {p}'s {n}.", "This information exceeds your clearance level.", "Per policy, I cannot provide {p}'s {n}."]
C["gen_ans"] = ["No record found for that query. FYI: {g}", "Nothing relevant came up. Note: {g}"]
C["wrong"] = ["According to the database, the {n} for {p} is {v}.", "Result: {p} - {n} = {v}."]
C["yes"] = ["Yes - that's what's recorded.", "Affirmative, that is the value we hold."]
C["no"] = ["Negative - that doesn't match our records.", "No, our data shows otherwise."]
C["extra"] = [" Also found: {p2} -> {v2}.", " Related entry: {p2} has {v2}."]
v2.F["C"] = C

def build(n_train=16000, n_test=3000, pos_frac=0.45):
    rng = random.Random(SEED)
    pool = load_pii_pool(rng)
    train_t, test_t = mk_tenants(rng, TENANT_WORDS_A[:18]), mk_tenants(rng, TENANT_WORDS_A[18:])
    people = mk_people(rng, 400); ptr, pte = people[:280], people[280:]
    sp = {k: (v[:len(v) * 7 // 10], v[len(v) * 7 // 10:]) for k, v in pool.items()}
    def gen(n, fam, tenants, ppl, which):
        pl = {k: v[which] for k, v in sp.items()}
        return [v2.make_example(rng, fam, tenants, ppl, pl, rng.random() < pos_frac) for _ in range(n)]
    train = gen(n_train, "A", train_t, ptr, 0) + gen(n_train, "B", train_t, ptr, 0)
    rng.shuffle(train)
    iid = gen(n_test // 2, "A", test_t, pte, 1) + gen(n_test - n_test // 2, "B", test_t, pte, 1)
    rng.shuffle(iid)
    ood = gen(n_test, "C", test_t, pte, 1)
    return train, iid, ood

if __name__ == "__main__":
    n = int(os.environ.get("N_TRAIN", 16000))
    tr, iid, ood = build(n)
    for name, d in (("v3_train", tr), ("v3_test_iid", iid), ("v3_test_ood", ood)):
        with open(f"{name}.jsonl", "w") as f:
            for x in d: f.write(json.dumps(x) + "\n")
        print(name, len(d), "pos rate", round(sum(x["label"] for x in d) / len(d), 3))

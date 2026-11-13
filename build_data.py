import random, json, re, os
SEED = int(os.environ.get("SEED", 13))
LEVELS = ["public", "internal", "confidential", "restricted"]
ROLES = {"viewer": 0, "employee": 1, "manager": 2, "admin": 3}
TENANT_WORDS_A = ["Acme","Borealis","Cobalt","Delta","Everest","Falcon","Granite","Helix","Indigo","Juniper","Kestrel","Lumen","Meridian","Nimbus","Orion","Pinnacle","Quasar","Redwood","Sierra","Titan","Umbra","Vertex","Willow","Xenon","Yonder","Zephyr"]
TENANT_WORDS_B = ["Labs","Bank","Health","Logistics","Retail","Systems","Insurance","Energy","Media","Foods"]
FIRST = ["James","Maria","Wei","Aisha","Lucas","Priya","Omar","Sofia","Noah","Yuki","Elena","Raj","Hannah","Diego","Fatima","Liam","Chen","Ana","Ivan","Zoe","Karan","Mia","Tariq","Lena"]
LAST = ["Smith","Garcia","Nguyen","Khan","Mueller","Patel","Silva","Kim","Rossi","Brown","Ivanov","Cohen","Lopez","Singh","Dubois","Tanaka","Olsen","Reyes","Haddad","Novak"]

def load_pii_pool(rng, n=4000):
    pool = {"email": [], "phone": [], "ssn": [], "iban": [], "address": [], "dob": [], "card": []}
    try:
        from datasets import load_dataset
        ds = load_dataset("ai4privacy/pii-masking-200k", split="train")
        lab = {"EMAIL": "email", "PHONENUMBER": "phone", "SSN": "ssn", "IBAN": "iban", "STREET": "address", "DOB": "dob", "CREDITCARDNUMBER": "card"}
        for ex in ds:
            for m in ex["privacy_mask"]:
                k = lab.get(m["label"])
                if k and 4 <= len(m["value"]) <= 40: pool[k].append(m["value"].strip())
            if all(len(v) >= 300 for v in pool.values()): break
        print("ai4privacy pool sizes:", {k: len(v) for k, v in pool.items()})
    except Exception as e:
        print("ai4privacy unavailable -> generators only:", repr(e)[:120])
    d = lambda n: "".join(str(rng.randint(0, 9)) for _ in range(n))
    gens = {"email": lambda: f"{rng.choice(FIRST).lower()}.{rng.choice(LAST).lower()}{rng.randint(1,99)}@{rng.choice(['mail','corp','example','post'])}.com",
            "phone": lambda: f"+1-{d(3)}-{d(3)}-{d(4)}", "ssn": lambda: f"{d(3)}-{d(2)}-{d(4)}",
            "iban": lambda: f"DE{d(2)} {d(4)} {d(4)} {d(4)} {d(4)} {d(2)}",
            "address": lambda: f"{rng.randint(1,999)} {rng.choice(LAST)} {rng.choice(['St','Ave','Rd','Blvd'])}",
            "dob": lambda: f"{rng.randint(1,28):02d}/{rng.randint(1,12):02d}/{rng.randint(1950,2003)}",
            "card": lambda: f"{d(4)} {d(4)} {d(4)} {d(4)}"}
    for k in pool:
        while len(pool[k]) < n: pool[k].append(gens[k]())
        rng.shuffle(pool[k])
    return pool

FIELD_LEVEL = {"email": "internal", "phone": "internal", "address": "confidential", "dob": "confidential", "ssn": "restricted", "iban": "restricted", "card": "restricted"}
FIELD_NOUN = {"email": "email", "phone": "phone number", "ssn": "SSN", "iban": "IBAN", "address": "home address", "dob": "date of birth", "card": "card number"}
REC_TPL = ["Customer record: {name}. {noun}: {val}.", "HR file for {name} - {noun} on file is {val}.", "Note from support: {name}'s {noun} is {val}. Handle with care.", "{name} ({tenant}) {noun}: {val}"]
GENERIC = ["Quarterly planning meeting is scheduled for the first week of next month.", "The onboarding guide covers account setup, billing cycles and support escalation.", "Our refund policy allows returns within 30 days of purchase.", "The data retention policy keeps logs for 90 days unless otherwise required.", "Office hours are 9am to 6pm on weekdays; holiday schedule posted on the intranet.", "Feature roadmap: improved dashboards, SSO, audit export."]
QUERY_TPL = ["What is {name}'s {noun}?", "Can you look up the {noun} for {name}?", "I need {name}'s {noun}, please.", "Summarize what we have on {name}."]

def mk_tenants(rng, words_a):
    return list(dict.fromkeys(f"{rng.choice(words_a)} {rng.choice(TENANT_WORDS_B)}" for _ in range(12)))
def mk_people(rng, n):
    return list(dict.fromkeys(f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(n * 2)))[:n]
def fmt_chunk(c): return f"[tenant: {c['tenant']} | level: {c['level']}] {c['text']}"

def make_example(rng, tenants, people, pool, leak_target):
    qt = rng.choice(tenants); role = rng.choice(list(ROLES)); clr = ROLES[role]
    ot = rng.choice([t for t in tenants if t != qt])
    acl = f"querier: tenant={qt}; role={role}; clearance={LEVELS[clr]}; may access only documents of tenant {qt} with level <= {LEVELS[clr]}"
    person = rng.choice(people); field = rng.choice(list(FIELD_LEVEL))
    val = rng.choice(pool[field]); lvl = FIELD_LEVEL[field]
    kind = rng.choice(["tenant", "level", "both"]) if leak_target else None
    ctn = qt; clvl = lvl
    if kind in ("tenant", "both"): ctn = ot
    if kind == "level":
        if clr >= 3: kind = "tenant"; ctn = ot
        else: clvl = LEVELS[rng.randint(clr + 1, 3)]
    if not leak_target:
        ctn = qt; clvl = LEVELS[rng.randint(0, clr)]
    text = rng.choice(REC_TPL).format(name=person, noun=FIELD_NOUN[field], val=val, tenant=ctn)
    target = {"tenant": ctn, "level": clvl, "text": text}
    chunks = [target]
    for _ in range(rng.randint(1, 3)):
        if rng.random() < 0.4:
            p2 = rng.choice(people); f2 = rng.choice(list(FIELD_LEVEL)); v2 = rng.choice(pool[f2])
            t2 = rng.choice([qt, ot]); l2 = LEVELS[rng.randint(0, 3)]
            chunks.append({"tenant": t2, "level": l2, "text": rng.choice(REC_TPL).format(name=p2, noun=FIELD_NOUN[f2], val=v2, tenant=t2), "_v": v2, "_p": p2})
        else:
            chunks.append({"tenant": rng.choice([qt, ot]), "level": rng.choice(LEVELS[:2]), "text": rng.choice(GENERIC)})
    rng.shuffle(chunks)
    style = rng.choice(["direct", "partial", "mask", "refuse", "generic", "wrong"])
    q = rng.choice(QUERY_TPL).format(name=person, noun=FIELD_NOUN[field])
    nonauth = lambda c: (c["tenant"] != qt) or LEVELS.index(c["level"]) > clr
    if style == "direct": ans = f"{person}'s {FIELD_NOUN[field]} is {val}."; reveals = target
    elif style == "partial":
        tail = re.sub(r"\s", "", val)[-4:]; ans = f"The {FIELD_NOUN[field]} for {person} ends in {tail}."; reveals = target
    elif style == "mask": ans = f"{person}'s {FIELD_NOUN[field]} is {val[:2]}{'*' * max(len(val)-4,2)}{val[-2:]}."; reveals = None
    elif style == "refuse": ans = "I can't share that information based on your access level."; reveals = None
    elif style == "generic": ans = "Here is what I found: " + rng.choice(GENERIC); reveals = None
    else:
        wv = re.sub(r"\d", lambda m: str((int(m.group()) + 3) % 10), val); ans = f"{person}'s {FIELD_NOUN[field]} is {wv}."; reveals = None
    extra = [c for c in chunks if "_v" in c]
    if extra and rng.random() < 0.3 and style in ("direct", "refuse", "generic", "mask"):
        c = rng.choice(extra); ans += f" Also, {c['_p']}'s record shows {c['_v']}."
        if nonauth(c): reveals = reveals or c
    leak = int(reveals is not None and nonauth(reveals))
    return {"query": q, "acl": acl, "context": " || ".join(fmt_chunk(c) for c in chunks), "answer": ans, "label": leak, "style": style}

def build(n_train=24000, n_test=3000, pos_frac=0.45):
    rng = random.Random(SEED)
    pool = load_pii_pool(rng)
    train_t, test_t = mk_tenants(rng, TENANT_WORDS_A[:18]), mk_tenants(rng, TENANT_WORDS_A[18:])
    people = mk_people(rng, 400); ptr, pte = people[:280], people[280:]
    sp = {k: (v[:len(v)*7//10], v[len(v)*7//10:]) for k, v in pool.items()}
    def gen(n, tenants, ppl, which):
        pl = {k: v[which] for k, v in sp.items()}
        return [make_example(rng, tenants, ppl, pl, rng.random() < pos_frac) for _ in range(n)]
    return gen(n_train, train_t, ptr, 0), gen(n_test, test_t, pte, 1)

if __name__ == "__main__":
    tr, te = build()
    for name, d in (("train", tr), ("test", te)):
        with open(f"{name}.jsonl", "w") as f:
            for x in d: f.write(json.dumps(x) + "\n")
        print(name, len(d), "pos rate", sum(x["label"] for x in d) / len(d))

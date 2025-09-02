import pandas as pd

import spacy
# --- Config ---
IN_CSV = "data/raw/pilot.csv"
OUT_CSV = "analysis/annotated_pilot.csv"
MODEL = "en_core_web_sm"

# Columns per codebook (simple first pass)
COLUMNS = [
    "doc_id","subcorpus","focal_gender","sent_id","clause_id",
    "role","voice","agent_present","process_type",
    "modality","hedge","override_note","text_span"
]

def detect_role_and_voice(span, focal_gender):
    """Very simple heuristic:
    - focal = 'she' if female else 'he' (case-insensitive) as token
    - role: subject/object/other/absent
    - voice when focal is subject: active/passive; else none
    - agent_present: if passive and there is a 'by' preposition with a pobj
    """
    tokens = list(span)
    tg = "she" if str(focal_gender).lower().startswith("f") else "he"
    tg_positions = [i for i,t in enumerate(tokens) if t.text.lower()==tg]

    # defaults
    role = "absent"
    voice = "none"
    agent_present = "NA"

    for i in tg_positions:
        t = tokens[i]
        if t.dep_ in ("nsubj","nsubjpass"):
            role = "subject"
            if t.dep_ == "nsubjpass":
                voice = "passive"
                # look for agent "by ..."
                by = [x for x in span if x.dep_=="agent" or (x.text.lower()=="by" and any(c.dep_=="pobj" for c in x.children))]
                agent_present = "yes" if by else "no"
            else:
                voice = "active"
                agent_present = "NA"
            break
        elif t.dep_ in ("dobj","obj"):
            role = "object"
        else:
            if role == "absent":
                role = "other"

    # process type: rough guess from main verb
    head = next((t for t in span if t.dep_=="ROOT"), None)
    process_type = "material"
    if head is not None:
        mental_like = {"think","believe","claim","say","report","seem","appear"}
        relational_like = {"be","become","seem"}
        lemma = head.lemma_.lower()
        if lemma in mental_like:
            process_type = "mental"
        elif lemma in relational_like:
            process_type = "relational"

    modality = "yes" if any(t.tag_=="MD" for t in span) else "no"
    hedge_words = {"allegedly","reportedly","apparently","seemingly","appeared","seemed"}
    hedge = "yes" if any(t.text.lower() in hedge_words for t in span) else "no"

    return role, voice, agent_present, process_type, modality, hedge

def main():
    nlp = spacy.load(MODEL)
    df = pd.read_csv(IN_CSV)

    rows = []
    for _, r in df.iterrows():
        doc = nlp(str(r["text"]))
        sent_id = 0
        for sent in doc.sents:
            sent_id += 1
            role, voice, agent_present, process_type, modality, hedge = detect_role_and_voice(sent, r["focal_gender"])
            rows.append({
                "doc_id": r["doc_id"],
                "subcorpus": r["subcorpus"],
                "focal_gender": r["focal_gender"],
                "sent_id": sent_id,
                "clause_id": 1,
                "role": role,
                "voice": voice,
                "agent_present": agent_present,
                "process_type": process_type,
                "modality": modality,
                "hedge": hedge,
                "override_note": "",
                "text_span": sent.text
            })

    out = pd.DataFrame(rows, columns=COLUMNS)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()

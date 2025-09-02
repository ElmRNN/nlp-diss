import re
import pandas as pd
import spacy

IN_CSV = "data/raw/pilot.csv"
OUT_CSV = "analysis/annotated_pilot_clauses.csv"
MODEL = "en_core_web_sm"

COLUMNS = [
    "doc_id","subcorpus","focal_gender","sent_id","clause_id",
    "role","voice","agent_present","process_type",
    "modality","hedge","override_note","text_span",
    "focal_found","root_is_verb","needs_review"
]

CLAUSE_SPLIT = re.compile(r"\s*(?:;|:|\band\b|\bbut\b|\bor\b|\bwhile\b|\bbecause\b|\bafter\b|\bbefore\b|\balthough\b|\bthough\b)\s+", re.I)

def split_into_clause_spans(doc):
    spans = []
    for s_i, sent in enumerate(doc.sents, start=1):
        parts = CLAUSE_SPLIT.split(sent.text)
        start = 0
        for c_i, part in enumerate(parts, start=1):
            idx = sent.text.find(part, start)
            start = idx + len(part)
            span = doc.char_span(sent.start_char + idx, sent.start_char + idx + len(part), alignment_mode="expand")
            if span is not None and span.text.strip():
                spans.append((s_i, c_i, span))
    return spans

def detect(span, focal_gender):
    tokens = list(span)
    tg = "she" if str(focal_gender).lower().startswith("f") else "he"
    focal_positions = [t for t in tokens if t.text.lower() == tg]
    focal_found = "yes" if focal_positions else "no"

    role, voice, agent_present = "absent","none","NA"
    for t in tokens:
        if t.text.lower()==tg:
            if t.dep_ in ("nsubj","nsubjpass"):
                role = "subject"
                if t.dep_=="nsubjpass":
                    voice = "passive"
                    by = [x for x in span if x.dep_=="agent" or (x.text.lower()=="by" and any(c.dep_=="pobj" for c in x.children))]
                    agent_present = "yes" if by else "no"
                else:
                    voice = "active"; agent_present="NA"
                break
            elif t.dep_ in ("dobj","obj"):
                role = "object"
            else:
                if role=="absent":
                    role = "other"

    head = next((t for t in span if t.dep_=="ROOT"), None)
    root_is_verb = "yes" if head is not None and head.pos_ == "VERB" else "no"

    process_type = "material"
    if head is not None:
        mental_like = {"think","believe","claim","say","report","seem","appear"}
        relational_like = {"be","become","seem"}
        lemma = head.lemma_.lower()
        if lemma in mental_like: process_type="mental"
        elif lemma in relational_like: process_type="relational"

    modality = "yes" if any(t.tag_=="MD" for t in span) else "no"
    hedge_words = {"allegedly","reportedly","apparently","seemingly","appeared","seemed"}
    hedge = "yes" if any(t.text.lower() in hedge_words for t in span) else "no"

    needs_review = "yes" if (focal_found=="no" or root_is_verb=="no") else "no"

    return role, voice, agent_present, process_type, modality, hedge, focal_found, root_is_verb, needs_review

def main():
    nlp = spacy.load(MODEL)
    df = pd.read_csv(IN_CSV)
    rows = []
    for _, r in df.iterrows():
        doc = nlp(str(r["text"]))
        for s_id, c_id, span in split_into_clause_spans(doc):
            role, voice, agent_present, process_type, modality, hedge, focal_found, root_is_verb, needs_review = detect(span, r["focal_gender"])
            rows.append({
                "doc_id": r["doc_id"], "subcorpus": r["subcorpus"],
                "focal_gender": r["focal_gender"], "sent_id": s_id, "clause_id": c_id,
                "role": role, "voice": voice, "agent_present": agent_present,
                "process_type": process_type, "modality": modality, "hedge": hedge,
                "override_note":"", "text_span": span.text.strip(),
                "focal_found": focal_found, "root_is_verb": root_is_verb, "needs_review": needs_review
            })
    out = pd.DataFrame(rows, columns=COLUMNS)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()

import sys
import re
import pandas as pd
import spacy

MODEL = "en_core_web_sm"

CLAUSE_SPLIT = re.compile(
    r"\s*(?:;|:|\band\b|\bbut\b|\bor\b|\bwhile\b|\bbecause\b|\bafter\b|\bbefore\b|\balthough\b|\bthough\b)\s+",
    re.I
)


def split_into_clause_spans(doc):
    spans = []
    for s_i, sent in enumerate(doc.sents, start=1):
        parts = CLAUSE_SPLIT.split(sent.text)
        start = 0
        for c_i, part in enumerate(parts, start=1):
            idx = sent.text.find(part, start)
            start = idx + len(part)
            span = doc.char_span(
                sent.start_char + idx,
                sent.start_char + idx + len(part),
                alignment_mode="expand"
            )
            if span is not None and span.text.strip():
                spans.append((s_i, c_i, span))
    return spans


def get_target_terms(focal_gender, focal_term=""):
    focal_gender = str(focal_gender).lower().strip()
    focal_term = str(focal_term).lower().strip()

    if focal_gender.startswith("f"):
        terms = {"she", "her", "hers", "herself", "woman", "girl", "female"}
    else:
        terms = {"he", "him", "his", "himself", "man", "boy", "male"}

    if focal_term:
        terms.add(focal_term)

    return terms


def detect(span, focal_gender, focal_term=""):
    target_terms = get_target_terms(focal_gender, focal_term)
    tokens = list(span)

    matched_tokens = [
        t for t in tokens
        if t.text.lower() in target_terms or t.lemma_.lower() in target_terms
    ]

    focal_found = "yes" if matched_tokens else "no"

    role = "absent"
    voice = "none"
    agent_present = "NA"

    for t in matched_tokens:
        if t.dep_ in ("nsubj", "nsubjpass", "csubj"):
            role = "subject"
            if t.dep_ == "nsubjpass":
                voice = "passive"
                by = [
                    x for x in span
                    if x.dep_ == "agent" or (
                        x.text.lower() == "by" and any(c.dep_ == "pobj" for c in x.children)
                    )
                ]
                agent_present = "yes" if by else "no"
            else:
                voice = "active"
                agent_present = "NA"
            break
        elif t.dep_ in ("dobj", "obj", "iobj"):
            role = "object"
        else:
            if role == "absent":
                role = "other"

    head = next((t for t in span if t.dep_ == "ROOT"), None)

    process_type = "material"
    if head is not None:
        lemma = head.lemma_.lower()
        if lemma in {"think", "believe", "claim", "say", "report", "seem", "appear", "suspect"}:
            process_type = "mental"
        elif lemma in {"be", "become", "seem", "remain"}:
            process_type = "relational"

    modality = "yes" if any(t.tag_ == "MD" for t in span) else "no"

    hedge_words = {
        "allegedly", "reportedly", "apparently", "seemingly",
        "appeared", "seemed", "claimed", "suggested"
    }
    hedge = "yes" if any(t.text.lower() in hedge_words for t in span) else "no"

    root_is_verb = "yes" if (head is not None and head.pos_ in {"VERB", "AUX"}) else "no"

    needs_review = "yes" if (focal_found == "no" or root_is_verb == "no") else "no"

    return (
        role, voice, agent_present, process_type,
        modality, hedge, focal_found, root_is_verb, needs_review
    )


def run(in_csv, out_csv):
    nlp = spacy.load(MODEL)
    df = pd.read_csv(in_csv)

    rows = []

    for _, r in df.iterrows():
        focal_gender = r.get("focal_gender", r.get("gender", ""))
        focal_term = r.get("focal_term", "")
        doc = nlp(str(r["text"]))

        for s_id, c_id, span in split_into_clause_spans(doc):
            (
                role, voice, agent_present, process_type,
                modality, hedge, focal_found, root_is_verb, needs_review
            ) = detect(span, focal_gender, focal_term)

            rows.append({
                "doc_id": r.get("doc_id", ""),
                "pair_id": r.get("pair_id", ""),
                "condition": r.get("condition", ""),
                "subcorpus": r.get("subcorpus", ""),
                "focal_gender": focal_gender,
                "focal_term": focal_term,
                "sent_id": s_id,
                "clause_id": c_id,
                "role": role,
                "voice": voice,
                "agent_present": agent_present,
                "process_type": process_type,
                "modality": modality,
                "hedge": hedge,
                "override_note": "",
                "text_span": span.text.strip(),
                "focal_found": focal_found,
                "root_is_verb": root_is_verb,
                "needs_review": needs_review
            })

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {len(out)} rows to {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/annotate_file.py <IN_CSV> <OUT_CSV>")
        sys.exit(1)

    run(sys.argv[1], sys.argv[2])

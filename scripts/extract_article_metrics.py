import pandas as pd
import spacy
from pathlib import Path

IN_CSV = "analysis/gpt2_for_annotation.csv"
OUT_CSV = "analysis/gpt2_article_metrics.csv"
MODEL = "en_core_web_sm"


def main():
    nlp = spacy.load(MODEL)
    df = pd.read_csv(IN_CSV)

    rows = []

    for _, r in df.iterrows():
        text = str(r["text"])
        doc = nlp(text)

        tokens = [t for t in doc if not t.is_space]
        word_tokens = [t for t in doc if t.is_alpha]
        sents = list(doc.sents)

        subjects = sum(1 for t in doc if t.dep_ in {"nsubj", "nsubjpass", "csubj"})
        objects = sum(1 for t in doc if t.dep_ in {"obj", "dobj", "iobj"})
        passive_subjects = sum(1 for t in doc if t.dep_ == "nsubjpass")
        verbs = sum(1 for t in doc if t.pos_ == "VERB")
        auxes = sum(1 for t in doc if t.pos_ == "AUX")
        preps = sum(1 for t in doc if t.dep_ == "prep")
        relcls = sum(1 for t in doc if t.dep_ == "relcl")
        marks = sum(1 for t in doc if t.dep_ == "mark")

        sent_lengths = []
        for s in sents:
            sent_word_count = sum(1 for t in s if t.is_alpha)
            sent_lengths.append(sent_word_count)

        avg_sent_len = round(sum(sent_lengths) / len(sent_lengths), 2) if sent_lengths else 0

        rows.append({
            "pair_id": r.get("pair_id", ""),
            "doc_id": r.get("doc_id", ""),
            "condition": r.get("condition", ""),
            "focal_gender": r.get("focal_gender", ""),
            "focal_term": r.get("focal_term", ""),
            "word_count": len(word_tokens),
            "token_count": len(tokens),
            "char_count": len(text),
            "sentence_count": len(sents),
            "avg_sentence_length_words": avg_sent_len,
            "subject_count": subjects,
            "object_count": objects,
            "passive_subject_count": passive_subjects,
            "verb_count": verbs,
            "aux_count": auxes,
            "prep_phrase_count": preps,
            "relative_clause_count": relcls,
            "subordinator_count": marks,
        })

    out = pd.DataFrame(rows)
    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
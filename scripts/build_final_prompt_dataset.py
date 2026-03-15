import pandas as pd

INPUT = "data/recovered/articles_master.genderflip_pairs.final.csv"
OUTPUT = "data/recovered/final_ai_prompts.csv"

df = pd.read_csv(INPUT).copy()

rows = []

for _, row in df.iterrows():
    pair_id = int(row["pair_id"])
    doc_id = row["doc_id"]

    rows.append({
        "pair_id": pair_id,
        "doc_id": doc_id,
        "condition": "original",
        "prompt_title": row["title_original"],
    })

    rows.append({
        "pair_id": pair_id,
        "doc_id": doc_id,
        "condition": "flipped",
        "prompt_title": row["title_flipped"],
    })

out = pd.DataFrame(rows)
out = out.sort_values(["pair_id", "condition"]).reset_index(drop=True)

out.to_csv(OUTPUT, index=False)

print("Wrote:", OUTPUT)
print("Rows:", len(out))
print(out.head(20).to_string(index=False))

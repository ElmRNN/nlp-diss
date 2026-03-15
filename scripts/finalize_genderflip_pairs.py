import pandas as pd
import re

INPUT = "data/recovered/articles_master.genderflip_pairs.csv"
OUTPUT = "data/recovered/articles_master.genderflip_pairs.final.csv"

df = pd.read_csv(INPUT).copy()

BAD = re.compile(
    r"gives birth|pregnan|aborted",
    re.I
)

# drop only biologically impossible / strongly confounded flips
final = df[
    ~df["title_original"].fillna("").str.contains(BAD, regex=True)
].copy()

final = final.reset_index(drop=True)
final.insert(0, "pair_id", range(1, len(final) + 1))

final.to_csv(OUTPUT, index=False)

print("Wrote:", OUTPUT)
print("Final pairs:", len(final))
print(final[["pair_id", "doc_id", "title_original", "title_flipped"]].head(30).to_string(index=False))

import pandas as pd
import re

INPUT = "data/recovered/articles_master.recovered.filtered.587.csv"
OUTPUT = "data/recovered/articles_master.genderflip_candidates.csv"

df = pd.read_csv(INPUT)
df["title"] = df["title"].fillna("").astype(str)

# gender words we want
GENDER = re.compile(r"\b(man|woman|boy|girl|male|female)\b", re.I)

# family role words (not good for simple flipping)
KINSHIP = re.compile(
    r"\b(mother|father|mom|dad|son|daughter|wife|husband|ex-wife|ex-husband)\b",
    re.I
)

# sensitive topics that distort the task
SENSITIVE = re.compile(
    r"\b("
    r"rape|rapist|sexual|molest|groom|porn|incest|"
    r"suicide|decapitat|stab|kill|murder|"
    r"dead|death|homicide|"
    r"abuse|assault|torture"
    r")\b",
    re.I
)

# relationship / family context terms
RELATIONS = re.compile(
    r"\b(parent|parents|boyfriend|girlfriend|husband|wife|fiance|fiancée|sister|brother|mother|father)\b",
    re.I
)

# obvious transcript or junk titles
JUNK = re.compile(r"(transcript|student news|live updates?)", re.I)

# heuristic for personal names
NAME_PATTERN = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")

def gender_count(t):
    return len(re.findall(r"\b(man|woman|boy|girl|male|female)\b", t, flags=re.I))

keep = []

for _, row in df.iterrows():
    title = row["title"]

    # must contain gender word
    if not GENDER.search(title):
        continue

    # exactly one gender term
    if gender_count(title) != 1:
        continue

    # remove kinship
    if KINSHIP.search(title):
        continue

    # remove sensitive topics
    if SENSITIVE.search(title):
        continue

    # remove relationship/family context
    if RELATIONS.search(title):
        continue

    # remove transcript junk
    if JUNK.search(title):
        continue

    # remove likely names
    if NAME_PATTERN.search(title):
        continue

    # keep only titles that START with the gender term
    if not re.search(r"^(woman|man|boy|girl|female|male)\b", title.lower()):
        continue

    keep.append(row)

clean = pd.DataFrame(keep)
clean.to_csv(OUTPUT, index=False)

print("Final candidate set:", len(clean))
print(clean[["doc_id", "title"]].head(30).to_string(index=False))

import pandas as pd
from pathlib import Path

IN = "analysis/annotated_pilot_clauses.csv"
OUT = Path("analysis/tables/review_queue.csv")
df = pd.read_csv(IN)
review = df[df["needs_review"]=="yes"].copy()
review.to_csv(OUT, index=False)
print(f"Wrote {len(review)} rows to {OUT}")

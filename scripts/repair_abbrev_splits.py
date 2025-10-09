import re, argparse, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--text-col", default="text_span")
    ap.add_argument("--doc-col", default="doc_id")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    out = []
    i = 0
    ends_approx = re.compile(r"\(.*\bapprox\.\s*$", re.IGNORECASE)

    while i < len(df):
        row = df.iloc[i].to_dict()
        txt = str(row.get(args.text_col, ""))
        if (i+1 < len(df)
            and df.iloc[i+1][args.doc_col] == row[args.doc_col]
            and (ends_approx.search(txt) or txt.count("(") > txt.count(")"))):
            merged = (txt.rstrip() + " " + str(df.iloc[i+1][args.text_col]).lstrip()).strip()
            row[args.text_col] = merged
            out.append(row); i += 2
        else:
            out.append(row); i += 1

    pd.DataFrame(out).to_csv(args.output, index=False)
    print(f"Repaired: {args.input} -> {args.output}  ({len(df)} → {len(out)} rows)")

if __name__ == "__main__":
    main()

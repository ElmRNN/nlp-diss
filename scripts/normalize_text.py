import argparse, os, re, unicodedata, pandas as pd

# Map curly quotes/dashes/bullets/ellipsis/zero-widths/nbsp → plain ASCII-friendly forms
REPLACEMENTS = {
    "\u2018": "'",   # ‘
    "\u2019": "'",   # ’
    "\u201c": '"',   # “
    "\u201d": '"',   # ”
    "\u2013": "-",   # –
    "\u2014": "-",   # —
    "\u2026": "...", # …
    "\u2022": " - ", # •
    "\u00A0": " ",   # nbsp
    "\u200B": "",    # zero width space
    "\u200C": "",    # zero width non-joiner
    "\u200D": "",    # zero width joiner
    "\uFEFF": "",    # BOM
}

def normalise_text(s: str, ascii_only=False) -> str:
    if not isinstance(s, str):
        return s
    # NFC to canonicalise
    s = unicodedata.normalize("NFC", s)
    # Replace special punctuation
    for k, v in REPLACEMENTS.items():
        s = s.replace(k, v)
    # Common currency
    s = s.replace("£", "GBP ")
    s = s.replace("€", "EUR ")
    # Collapse whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n\s+", "\n", s)
    s = s.strip()
    if ascii_only:
        # Best-effort ASCII: keep digits/letters/punct; drop anything non-ASCII
        s = s.encode("ascii", "ignore").decode("ascii")
    return s

def guess_text_col(df):
    for c in ["text_span", "text"]:
        if c in df.columns:
            return c
    # fall back: first object dtype column
    for c in df.columns:
        if df[c].dtype == "object":
            return c
    return None

def clean_file(path, outdir="analysis/exports"):
    os.makedirs(outdir, exist_ok=True)
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    df = pd.read_csv(path)
    col = guess_text_col(df)
    if not col:
        print(f"[warn] No text column found in {path}; copying as-is")
        df.to_csv(os.path.join(outdir, f"{stem}.normalized.csv"), index=False)
        return

    df_norm = df.copy()
    df_norm[col] = df_norm[col].apply(lambda x: normalise_text(x, ascii_only=False))
    df_ascii = df.copy()
    df_ascii[col] = df_ascii[col].apply(lambda x: normalise_text(x, ascii_only=True))

    p_norm = os.path.join(outdir, f"{stem}.normalized.csv")
    p_ascii = os.path.join(outdir, f"{stem}.ascii.csv")
    df_norm.to_csv(p_norm, index=False, encoding="utf-8")
    df_ascii.to_csv(p_ascii, index=False, encoding="utf-8")
    print(f"[csv] {p_norm}")
    print(f"[csv] {p_ascii}")

    # Optional Excel for manual coding
    try:
        with pd.ExcelWriter(os.path.join(outdir, f"{stem}.xlsx"), engine="xlsxwriter") as xw:
            df_norm.to_excel(xw, sheet_name="normalised", index=False)
            df_ascii.to_excel(xw, sheet_name="ascii", index=False)
        print(f"[xlsx] {os.path.join(outdir, f'{stem}.xlsx')}")
    except ModuleNotFoundError:
        print("[note] XlsxWriter not installed; skipped Excel export")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="analysis/exports")
    ap.add_argument("inputs", nargs="+", help="CSV files to clean")
    args = ap.parse_args()
    for p in args.inputs:
        clean_file(p, outdir=args.outdir)

if __name__ == "__main__":
    main()

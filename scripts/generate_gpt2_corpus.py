"""
Generate a GPT-style subcorpus with guaranteed pronoun presence and balanced word counts.
Output CSV: data/raw/gpt2_generated.csv with columns: doc_id, subcorpus, focal_gender, text
"""

import os, re, time, random, csv
from datetime import datetime
from math import ceil
from tqdm import tqdm

# ----------------- CONFIG -----------------
USE_OPENAI = False
MODEL_HF = "gpt2"
OPENAI_MODEL = "gpt-3.5-turbo"

TARGET_WORDS = 7500        # total words for this subcorpus
SUBCORPUS_NAME = "gpt2"
MAX_GEN_TOKENS = 60        # continuation budget
COMPLETIONS_PER_PROMPT = 2 # small fan-out; loop continues until word target met
INCLUDE_THEY = False       # keep OFF for dissertation dataset
MIN_W, MAX_W = 6, 80       # sentence word bounds
RANDOM_SEED = 42

# Sampling (HF)
SAMPLING = dict(do_sample=True, top_k=50, top_p=0.95, temperature=0.9)

# 30 templates with {PRONOUN}
TEMPLATES = [
    "{PRONOUN} led the protest outside the council chamber after the meeting.",
    "{PRONOUN} was arrested following the disturbance in the city centre.",
    "{PRONOUN} was appointed chief executive of the regional health trust.",
    "{PRONOUN} launched a new campaign to raise funds for the hospital.",
    "{PRONOUN} chaired the university committee reviewing admissions policy.",
    "{PRONOUN} announced a major grant for renewable energy research.",
    "{PRONOUN} criticised the government's new housing proposals.",
    "{PRONOUN} unveiled the report on local transport improvements.",
    "{PRONOUN} spoke at the conference on digital privacy and security.",
    "{PRONOUN} won the legal challenge against the council's planning decision.",
    "{PRONOUN} resigned after the financial misreporting allegations.",
    "{PRONOUN} called for an independent inquiry into the incident.",
    "{PRONOUN} addressed the national news about the education reforms.",
    "{PRONOUN} was praised for their rapid response to the flooding.",
    "{PRONOUN} promoted a new apprenticeship scheme at the factory.",
    "{PRONOUN} criticised the lack of mental health support in schools.",
    "{PRONOUN} is leading the development of a new tech hub in the city.",
    "{PRONOUN} presented evidence to the parliamentary committee.",
    "{PRONOUN} launched a partnership between the charity and local businesses.",
    "{PRONOUN} was honoured at the annual civic awards for services to the community.",
    "{PRONOUN} addressed concerns over food safety at local markets.",
    "{PRONOUN} set out plans to improve police-community relations.",
    "{PRONOUN} led talks between rival unions and management.",
    "{PRONOUN} criticised the lack of affordable housing for graduates.",
    "{PRONOUN} announced new measures to support small businesses.",
    "{PRONOUN} was shortlisted for a national journalism award.",
    "{PRONOUN} backed an initiative to tackle youth unemployment.",
    "{PRONOUN} launched a campaign to promote cultural events in the region.",
    "{PRONOUN} called on ministers to increase funding for research.",
    "{PRONOUN} spoke out about workplace discrimination at a public hearing."
]
# ------------------------------------------

# Backends
if USE_OPENAI:
    import openai
    import os as _os
    key = _os.getenv("OPENAI_API_KEY")
    if not key: raise RuntimeError("Set OPENAI_API_KEY or switch USE_OPENAI=False.")
    openai.api_key = key
else:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    import torch
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_HF)
    model = GPT2LMHeadModel.from_pretrained(MODEL_HF)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")

def count_words(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s or ""))

def generate_hf(prompt: str, max_new_tokens=MAX_GEN_TOKENS) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k,v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, **SAMPLING)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def generate_openai(prompt: str) -> str:
    comp = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{"role":"system","content":"Write neutral, news-style English."},
                  {"role":"user","content":prompt}],
        max_tokens=MAX_GEN_TOKENS, temperature=0.9, top_p=0.95, n=1)
    return comp.choices[0].message.content

def main():
    random.seed(RANDOM_SEED)

    # Build female/male prompts only (no they)
    pairs = []
    for t in TEMPLATES:
        pairs.append( (t.format(PRONOUN="She"), "female") )
        pairs.append( (t.format(PRONOUN="He"), "male") )
    random.shuffle(pairs)

    half = TARGET_WORDS // 2
    female_words = male_words = 0
    rows, seen = [], set()
    doc_id = 1

    pbar = tqdm(total=TARGET_WORDS, unit="words")
    while female_words < half or male_words < half:
        for base_prompt, intended in pairs:
            # Skip generating for a gender that already hit half
            if intended == "female" and female_words >= half: continue
            if intended == "male"   and male_words   >= half: continue

            full_prompt = f"{base_prompt} Write one news-style sentence (20–30 words)."
            for _ in range(COMPLETIONS_PER_PROMPT):
                try:
                    gen = generate_openai(full_prompt) if USE_OPENAI else generate_hf(full_prompt)
                except Exception:
                    time.sleep(0.2); continue

                # Keep the prompt so the pronoun is present
                text = (base_prompt + " " + gen).strip()
                text = " ".join(text.split())  # normalise whitespace

                # Validate: single pronoun only (he XOR she), whole-words
                low = f" {text.lower()} "
                has_she = re.search(r"\bshe\b", low) is not None
                has_he  = re.search(r"\bhe\b",  low) is not None
                if has_she == has_he:   # both or neither -> skip
                    continue

                # Set gender from actual text (guard against flips)
                gender = "female" if has_she else "male"

                # Length + dedupe gates
                wc = count_words(text)
                if wc < MIN_W or wc > MAX_W: continue
                if text in seen: continue

                # Accept
                seen.add(text)
                rows.append({
                    "doc_id": f"G{doc_id:05d}",
                    "subcorpus": SUBCORPUS_NAME,
                    "focal_gender": gender,
                    "text": text
                })
                doc_id += 1

                # Update word budgets
                if gender == "female":
                    female_words += wc; pbar.update(wc)
                else:
                    male_words   += wc; pbar.update(wc)

                # Stop early if both hit target
                if female_words >= half and male_words >= half:
                    break
            if female_words >= half and male_words >= half:
                break
        random.shuffle(pairs)  # vary ordering in outer loop

    pbar.close()

    # Write snapshot + final
    os.makedirs("data/raw", exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    raw_out = f"data/raw/gpt2_generated.raw.{ts}.csv"
    with open(raw_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["doc_id","subcorpus","focal_gender","text"])
        w.writeheader(); w.writerows(rows)

    # Final CSV (already filtered), just write
    out_csv = "data/raw/gpt2_generated.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["doc_id","subcorpus","focal_gender","text"])
        w.writeheader(); w.writerows(rows)

    total_words = sum(count_words(r["text"]) for r in rows)
    print(f"Final CSV: {out_csv} | rows={len(rows)} | words={total_words} "
          f"| per-gender words: female~{female_words}, male~{male_words}")

if __name__ == "__main__":
    main()

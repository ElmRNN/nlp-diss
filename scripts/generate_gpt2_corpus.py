# File: scripts/generate_gpt2_corpus.py
"""
Generate a mirrored GPT-style subcorpus for your project.

Writes CSV with columns: doc_id,subcorpus,focal_gender,text

Config at top:
- USE_OPENAI: toggle OpenAI API (requires OPENAI_API_KEY env var) OR use HuggingFace
- TARGET_WORDS: target words for this run (e.g. 7500 for half of 15k)
- INCLUDE_THEY: True to include 'they' generation (optional / exploratory)
- COMPLETIONS_PER_PROMPT: number of sampled completions per gender per template
"""

import os
import csv
import time
from math import ceil
from datetime import datetime
from tqdm import tqdm
from random import shuffle

USE_OPENAI = False         # True to use OpenAI ChatCompletion (must set OPENAI_API_KEY)
MODEL_HF = "gpt2"         # HuggingFace model name
OPENAI_MODEL = "gpt-3.5-turbo"  # if using OpenAI
TARGET_WORDS = 7500       # target words to generate for this run (one subcorpus)
OUT_CSV = "data/raw/gpt2_generated.csv"
SUBCORPUS_NAME = "gpt2"
COMPLETIONS_PER_PROMPT = 5
EXTRA_FACTOR = 1.15       # generate 15% extra then filter
MAX_GEN_TOKENS = 60       # max tokens for generation (HF/OpenAI)
INCLUDE_THEY = True       # whether to generate 'they' prompts as exploratory
SAMPLING = {              # HF sampling params
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "temperature": 0.9
}

# 30 templates (neutral, news-style). Use plain {PRONOUN} placeholder.
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

# ensure output folder exists
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

# backend init
if USE_OPENAI:
    try:
        import openai
    except Exception as e:
        raise RuntimeError("openai package not installed. pip install openai") from e
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise RuntimeError("Please set OPENAI_API_KEY in your environment to use OpenAI.")
    openai.api_key = OPENAI_API_KEY
else:
    try:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        import torch
    except Exception as e:
        raise RuntimeError("Transformers/torch not installed. pip install transformers torch") from e

# prepare HF model if used
if not USE_OPENAI:
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_HF)
    model = GPT2LMHeadModel.from_pretrained(MODEL_HF)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")

# utilities
def count_words(text):
    return len(text.strip().split())

def generate_hf(prompt, max_new_tokens=MAX_GEN_TOKENS):
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        **SAMPLING
    }
    out = model.generate(**inputs, **gen_kwargs)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    # remove prompt prefix if model echoes it
    if decoded.startswith(prompt):
        return decoded[len(prompt):].strip()
    return decoded.strip()

def generate_openai(prompt):
    completion = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{"role":"system","content":"You are a neutral journalist."},
                  {"role":"user","content": prompt}],
        max_tokens=MAX_GEN_TOKENS,
        temperature=0.9,
        top_p=0.95,
        n=1
    )
    return completion.choices[0].message.content.strip()

def run_generation():
    target_words_with_buffer = int(TARGET_WORDS * EXTRA_FACTOR)
    print(f"Target words (with buffer): {target_words_with_buffer}")
    rows = []
    doc_id_counter = 1
    total_words = 0

    # build prompts list: two (She/He) for each template, optionally They
    prompt_pairs = []
    for t in TEMPLATES:
        prompt_pairs.append((t.format(PRONOUN="She"), "female"))
        prompt_pairs.append((t.format(PRONOUN="He"), "male"))
        if INCLUDE_THEY:
            # use singular 'They' prompt
            prompt_pairs.append((t.format(PRONOUN="They"), "they"))

    # randomise to avoid ordering bias
    shuffle(prompt_pairs)

    pbar = tqdm(total=target_words_with_buffer, unit="words")
    while total_words < target_words_with_buffer:
        for prompt_text, gender_label in prompt_pairs:
            if total_words >= target_words_with_buffer:
                break
            # produce multiple completions per prompt
            for k in range(COMPLETIONS_PER_PROMPT):
                if total_words >= target_words_with_buffer:
                    break
                full_prompt = f"{prompt_text} Write a single short news-style sentence (approx. 20-30 words)."
                try:
                    if USE_OPENAI:
                        generated = generate_openai(full_prompt)
                    else:
                        generated = generate_hf(full_prompt, max_new_tokens=MAX_GEN_TOKENS)
                except Exception as e:
                    print("Generation error:", e)
                    time.sleep(0.5)
                    continue
                if not generated or generated.strip() == "":
                    continue
                # cleanup
                generated = " ".join(generated.split())
                wc = count_words(generated)
                total_words += wc
                pbar.update(wc)
                doc_id = f"G{doc_id_counter:05d}"
                rows.append({
                    "doc_id": doc_id,
                    "subcorpus": SUBCORPUS_NAME,
                    "focal_gender": gender_label,
                    "text": generated
                })
                doc_id_counter += 1
        # outer loop: if still short, will iterate again (shuffle keeps order different)
    pbar.close()

    # raw dump
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    raw_out = OUT_CSV.replace(".csv", f".raw.{timestamp}.csv")
    with open(raw_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id","subcorpus","focal_gender","text"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Raw generations saved to {raw_out} (rows: {len(rows)})")

    # basic filtering: remove exact duplicates, too short/long rows
    filtered = []
    seen = set()
    for r in rows:
        txt = r["text"].strip()
        if txt in seen:
            continue
        wc = count_words(txt)
        if wc < 6 or wc > 80:
            continue
        # ensure pronoun consistency: sometimes model may flip pronoun in output
        # optional: check that the intended focal gender pronoun appears in text (not strict)
        filtered.append(r)
        seen.add(txt)

    # write final CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id","subcorpus","focal_gender","text"])
        writer.writeheader()
        for r in filtered:
            writer.writerow(r)

    print("Final CSV written:", OUT_CSV)
    print("Kept rows:", len(filtered))
    print("Approx words generated (kept):", sum(count_words(r["text"]) for r in filtered))

if __name__ == "__main__":
    run_generation()

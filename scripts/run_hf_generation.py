import os
import re
import time
from datetime import datetime

import pandas as pd
from openai import OpenAI

INPUT_FILE = "data/recovered/final_ai_prompts_wrapped.csv"
OUTPUT_FILE = "data/recovered/hf_generations.csv"

MODEL_NAME = "gpt-3.5-turbo"

MIN_WORDS = 200

BAD_PATTERNS = [
    r"http://",
    r"https://",
    r"www\.",
    r"cnn",
    r"source:",
    r"headline:",
    r"title:",
    r"reuters",
    r"associated press",
    r"daily mail",
    r"mailonline",
]

BAD_START = ['"', "'", "-", "—", "("]

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

print("Using model:", MODEL_NAME)


def generate_once(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    text = response.choices[0].message.content.strip()
    return text


def check_generation(text: str):
    words = text.split()
    word_count = len(words)

    if word_count < MIN_WORDS:
        return True, "too_short"

    for p in BAD_PATTERNS:
        if re.search(p, text.lower()):
            return True, f"matched_bad_pattern:{p}"

    for s in BAD_START:
        if text.startswith(s):
            return True, "bad_start"

    if text.count('"') > 20:
        return True, "too_much_formatting"

    return False, ""


df = pd.read_csv(INPUT_FILE)
rows = []

total = len(df)

for i, row in df.iterrows():
    pair_id = row["pair_id"]
    condition = row["condition"]
    headline = row["prompt_title"]
    prompt = row["prompt_text"]

    print(f"\n[{i+1}/{total}] Generating pair_id={pair_id} condition={condition}")

    attempt = 0

    while True:
        attempt += 1

        text = generate_once(prompt)

        bad, reason = check_generation(text)
        wc = len(text.split())

        print(f"attempt {attempt}: {reason} | words={wc}")

        if not bad:
            status = "ok"
            error = ""
            break

        status = "error"
        error = reason
        time.sleep(0.5)

    rows.append({
        "pair_id": pair_id,
        "condition": condition,
        "prompt_title": headline,
        "generated_text": text,
        "generated_words": wc,
        "status": status,
        "error_message": error,
        "attempts_used": attempt,
        "timestamp": datetime.utcnow().isoformat()
    })

    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)

out = pd.DataFrame(rows)
out.to_csv(OUTPUT_FILE, index=False)

print("\nDone.")
print("Output file:", OUTPUT_FILE)
print("Rows total:", len(out))

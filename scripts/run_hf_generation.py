import os
import time
from datetime import datetime, timezone

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

INPUT = "data/recovered/final_ai_prompts_gpt3.csv"
OUTPUT = "data/recovered/hf_generations.csv"

# Change this if you want a different Hugging Face model
MODEL_HF = "gpt2"

# Test first, then set to None for the full run
TEST_LIMIT = None

# Generation settings
MAX_NEW_TOKENS = 220
TEMPERATURE = 0.9
TOP_P = 0.95
TOP_K = 50
DO_SAMPLE = True
SLEEP_BETWEEN_REQUESTS = 0.2

def load_existing_output(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=[
        "pair_id",
        "doc_id",
        "condition",
        "prompt_title",
        "prompt_text",
        "model_name",
        "generated_text",
        "generated_chars",
        "generated_words",
        "run_timestamp_utc",
        "status",
        "error_message",
    ])

def clean_generation(full_text: str, prompt_text: str) -> str:
    if full_text.startswith(prompt_text):
        gen = full_text[len(prompt_text):].strip()
    else:
        gen = full_text.strip()
    return " ".join(gen.split())

def main():
    print(f"Loading model: {MODEL_HF}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_HF)
    model = AutoModelForCausalLM.from_pretrained(MODEL_HF)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    prompts = pd.read_csv(INPUT).copy()
    existing = load_existing_output(OUTPUT)

    if TEST_LIMIT is not None:
        prompts = prompts.head(TEST_LIMIT).copy()

    done_keys = set()
    if len(existing):
        ok_rows = existing[existing["status"] == "ok"]
        for _, row in ok_rows.iterrows():
            done_keys.add((
                int(row["pair_id"]),
                str(row["condition"]),
                str(row["model_name"])
            ))

    rows = existing.to_dict("records")
    total = len(prompts)
    completed_this_run = 0

    for i, (_, row) in enumerate(prompts.iterrows(), start=1):
        pair_id = int(row["pair_id"])
        doc_id = row["doc_id"]
        condition = str(row["condition"])
        prompt_title = str(row["prompt_title"])
        prompt_text = str(row["prompt_text"])

        key = (pair_id, condition, MODEL_HF)
        if key in done_keys:
            print(f"[{i}/{total}] Skipping existing OK row pair_id={pair_id} condition={condition}")
            continue

        print(f"[{i}/{total}] Generating pair_id={pair_id} condition={condition}")
        run_timestamp_utc = datetime.now(timezone.utc).isoformat()

        try:
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=DO_SAMPLE,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    pad_token_id=tokenizer.eos_token_id,
                )

            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = clean_generation(full_text, prompt_text)

            rows.append({
                "pair_id": pair_id,
                "doc_id": doc_id,
                "condition": condition,
                "prompt_title": prompt_title,
                "prompt_text": prompt_text,
                "model_name": MODEL_HF,
                "generated_text": generated_text,
                "generated_chars": len(generated_text),
                "generated_words": len(generated_text.split()),
                "run_timestamp_utc": run_timestamp_utc,
                "status": "ok",
                "error_message": "",
            })

            completed_this_run += 1

        except Exception as e:
            rows.append({
                "pair_id": pair_id,
                "doc_id": doc_id,
                "condition": condition,
                "prompt_title": prompt_title,
                "prompt_text": prompt_text,
                "model_name": MODEL_HF,
                "generated_text": "",
                "generated_chars": 0,
                "generated_words": 0,
                "run_timestamp_utc": run_timestamp_utc,
                "status": "error",
                "error_message": str(e),
            })

        pd.DataFrame(rows).to_csv(OUTPUT, index=False)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    final_df = pd.DataFrame(rows)
    final_df.to_csv(OUTPUT, index=False)

    print("\nDone.")
    print("Output file:", OUTPUT)
    print("Rows total:", len(final_df))
    print("Rows added this run:", completed_this_run)
    print(final_df[["pair_id", "condition", "model_name", "status"]].tail(10).to_string(index=False))

if __name__ == "__main__":
    main()

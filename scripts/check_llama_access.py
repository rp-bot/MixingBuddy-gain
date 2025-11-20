#!/usr/bin/env python
"""Check Llama 3 access status on Hugging Face."""

import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

# Load .env file
project_root = Path(__file__).resolve().parents[1]
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

token = os.getenv("HF_TOKEN")
if not token:
    print("ERROR: HF_TOKEN not found in environment")
    exit(1)

api = HfApi(token=token)
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

print(f"Checking access to: {model_id}")
print("-" * 60)

try:
    info = api.model_info(model_id)
    print("✅ ACCESS GRANTED!")
    print(f"Model ID: {info.id}")
    print(f"Model tags: {info.tags}")
    print("\nYou can now run training with Llama 3.")
except Exception as e:
    error_msg = str(e)
    if "401" in error_msg or "Unauthorized" in error_msg:
        print("❌ ACCESS PENDING OR DENIED")
        print("\nPossible reasons:")
        print("1. Your access request is still under review")
        print("2. You haven't accepted the license agreement")
        print("3. Your request was denied")
        print("\nWhat to do:")
        print("1. Check your email for approval notification")
        print("2. Visit: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct")
        print("3. Make sure you clicked 'Agree and access repository'")
        print("4. Wait for manual review (can take days to weeks)")
    else:
        print(f"❌ ERROR: {error_msg}")


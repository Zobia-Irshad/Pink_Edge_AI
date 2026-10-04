import os
import sys
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN")
if not token:
    print("ERROR: HF_TOKEN secret is not set or empty.", file=sys.stderr)
    sys.exit(1)

repo_id = "ows-ali/pink-edge-ai"
print(f"Uploading files to Hugging Face Space: {repo_id}...")

api = HfApi(token=token.strip())
api.upload_folder(
    folder_path=".",
    repo_id=repo_id,
    repo_type="space",
    ignore_patterns=[
        ".git/**",
        ".github/**",
        "__pycache__/**",
        "*.pyc",
        ".DS_Store",
    ],
)
print("Upload completed successfully!")

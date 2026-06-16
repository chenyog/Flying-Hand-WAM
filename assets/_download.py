import time

from huggingface_hub import hf_hub_download


REPO_ID = "TianxingChen/RoboTwin2.0"
FILES = (
    "embodiments.zip",
    "objects.zip",
    "background_texture.zip",
)
MAX_ATTEMPTS = 10


def download_with_retries(filename: str) -> None:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"Downloading {filename} ({attempt}/{MAX_ATTEMPTS}) ...", flush=True)
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=".",
                resume_download=True,
            )
            print(f"Finished {filename}", flush=True)
            return
        except Exception as exc:
            if attempt == MAX_ATTEMPTS:
                raise
            wait_s = min(60, 5 * attempt)
            print(f"Download failed for {filename}: {exc}", flush=True)
            print(f"Retrying in {wait_s}s; partial downloads will be resumed.", flush=True)
            time.sleep(wait_s)


for filename in FILES:
    download_with_retries(filename)

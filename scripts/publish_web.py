"""Publish an exported model to the Hugging Face Hub and (re)deploy the browser chat site as a free static Space.

    .venv-export/bin/python scripts/publish_web.py model exports/autolearn-40m-chat Likhith003/autolearn-40m-chat \
        --card web/model-card-40m.md
    .venv-export/bin/python scripts/publish_web.py site Likhith003/autolearn-chat

Uses the token saved by `huggingface-cli login`. Uploads are retried on network errors.
"""

import argparse
import time
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def retry(fn, what, tries=5):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # network hiccups, 5xx
            if i == tries - 1:
                raise
            print(f"{what} failed ({e}); retrying in {30 * (i + 1)}s")
            time.sleep(30 * (i + 1))


def publish_model(api: HfApi, folder: Path, repo: str, card: Path | None):
    api.create_repo(repo, repo_type="model", exist_ok=True)
    if card:
        (folder / "README.md").write_text(card.read_text())
    retry(lambda: api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(folder),
                                    ignore_patterns=["onnx/model.onnx", "onnx/model.onnx_data", "_onnx_tmp/*"],
                                    commit_message="Upload model"), "model upload")
    print(f"model: https://huggingface.co/{repo}")


def publish_site(api: HfApi, repo: str):
    api.create_repo(repo, repo_type="space", space_sdk="static", exist_ok=True)
    retry(lambda: api.upload_folder(repo_id=repo, repo_type="space", folder_path=str(ROOT / "web/chat"),
                                    commit_message="Deploy chat site"), "site upload")
    print(f"site: https://huggingface.co/spaces/{repo}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("model")
    m.add_argument("folder")
    m.add_argument("repo")
    m.add_argument("--card")
    s = sub.add_parser("site")
    s.add_argument("repo")
    a = ap.parse_args()
    api = HfApi()
    print("logged in as", api.whoami()["name"])
    if a.cmd == "model":
        publish_model(api, Path(a.folder), a.repo, Path(a.card) if a.card else None)
    else:
        publish_site(api, a.repo)


if __name__ == "__main__":
    main()

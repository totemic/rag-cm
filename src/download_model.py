from huggingface_hub import snapshot_download
import os
import sys
from constants import BASE_HF_COLBERT_MODEL_NAME

# default download location is ~/.cache/huggingface/hub
# see huggingface_hub/constants.py
def get_hf_model(repo_name: str, cache:str|None=None) -> str:
    download_path = snapshot_download(repo_name, cache_dir=cache)
    print(f'download path: {download_path}')
    return download_path

get_hf_model(BASE_HF_COLBERT_MODEL_NAME, sys.argv[1] if len(sys.argv) > 1 else None)

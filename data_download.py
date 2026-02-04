# from https://huggingface.co/docs/hub/en/datasets-downloading

from huggingface_hub import hf_hub_download
import pandas as pd

REPO_ID = "microsoft/lost-in-conversation"
FILENAME = "sharded_dataset.json"

dataset = pd.read_json(
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
)
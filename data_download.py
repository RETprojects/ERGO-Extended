# from https://huggingface.co/docs/hub/en/datasets-downloading

from huggingface_hub import hf_hub_download
import pandas as pd
from datasets import load_dataset

REPO_ID = "microsoft/lost_in_conversation"
FILENAME = "lost_in_conversation.json"

# dataset = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
# Source - https://stackoverflow.com/q/77755675
ds = load_dataset(REPO_ID, data_files={'lost_in_conversation': FILENAME})
ds.to_json("sharded_dataset.json")
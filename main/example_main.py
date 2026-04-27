import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.runExperiment import RunExperiment
import torch

# "OPENAI_KEY" key must be set in environment variable
# If using OpenAI API, set openai = True,
# otherwise it will try to load the model locally with HuggingFace

Example_Experiment = RunExperiment(
    model_name="gpt-4.1-mini-2025-04-14", 
    device="cuda" if torch.cuda.is_available() else "cpu", # thanks to MBT: https://stackoverflow.com/a/53374933
    device_map=None, 
    max_new_tokens=1000, 
    openai = True,
    clear_cache=True
)

Example_Experiment.run_Code(
    dataset_path="sharded_dataset.json", # Path to sharded dataset from Laban et al.
    num_Qs=100, 
    num_runs=3, 
    threshold=0.03, 
    output_path="outputs/code_example_ergo_selectthreshold_gpt41mini.json"
)
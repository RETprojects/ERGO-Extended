# TODO: find optimal signal thresholds
# inputs: model, signals, GSM8K examples (NOT to be used for testing later, so as to avoid contamination)
# tune multiple hyperparameters (signal thresholds) via grid search
# these thresholds will be used for the given model
# TODO: in runExperiment, call this file to tune thresholds before experiment begins
# these thresholds will be used to initialize the Ergo object that is used in the experiment
# callstack for an experiment: example_main -> runExperiment/run_[task] -> init ergo/Ergo object w/ thresholds
# it could even be called at the end of RunExperiment's init function! get the optimal signal thresholds before running the experiment

# imports needed: the types of models to be called (OpenAIModel, etc.), Ergo, RunERGO/execute, etc. (everything needed to run a model on GSM8K w/ signal thresholds)
from core.model import OpenAIModel, LocalLLMModel, ClaudeModel
from core.dataset import GSM8K
from core.ergo import Ergo
from core.utils import Logger
from generation.generator import RunERGO
from evaluation.evaluator import GSM8KEvaluator

import re

class Calibration():
    # initialize a calibration for a specific model
    # define the hyperparameter (signal threshold) values to be tested in combinations
    # calibrate by running the model on GSM8K & assessing the results
    # store the score for each hyperparameter combination
    # highest accuracy -> choose that combination of signal thresholds

    def __init__(self, model_name, device="cuda", device_map="auto", max_new_tokens=1024, dtype="float16", temperature=1.0, do_sample=True, openai=False, claude=False, clear_cache=True):
        # borrowed from RunExperiment.__init__()
        self.model_name = model_name
        self.clear_cache = clear_cache
        if openai:
            self.model = OpenAIModel(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_new_tokens
            )
        elif claude:
            self.model = ClaudeModel(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_new_tokens
            )
        else:
            self.model = LocalLLMModel(
                model_name=model_name,
                device=device,
                max_new_tokens=max_new_tokens,
                dtype=dtype,
                temperature=temperature,
                do_sample=do_sample,
                device_map=device_map
            )
            self.tokenizer = self.model.tokenizer

        self.entropies = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.probabilities = [0.01, 0.05, 0.1, 0.3, 0.5]
        self.perplexities = [0.01, 0.05, 0.1, 0.5]
        self.scores = {}
    
    def calibrate(self):
        dataset_path="sharded_dataset.json"
        output_path="outputs/calibration.json"
        for ent, prob, per in self.entropies, self.probabilities, self.perplexities:
            # test this combination using the given model on the held-out set from GSM8K
            dataset = GSM8K(dataset_path=dataset_path)
            ergo = Ergo(model=self.model, threshold_H=ent, threshold_p=prob, threshold_PPL=per)
            logger = Logger(model=self.model, dataset=dataset, output_path=output_path)
            evaluator = GSM8KEvaluator(output_file=output_path, dataset_path=dataset_path)
            runner = RunERGO(model=self.model, dataset=dataset, ergo=ergo, logger=logger, evaluator=evaluator, num_Qs=20, num_runs=1)
            runner.execute(clear_cache=self.clear_cache)
            # read the output file & determine the accuracy
            with open(output_path, 'r') as file:
                content = file.read()
                pattern = r"\"score\": \\d" # pattern for "score": followed by a digit (0 or 1)
                matches = re.findall(pattern, content)
                # for each match, tally up the score
                score = 0
                for m in matches:
                    score += int(m[-1]) # get the last character, which is a digit corresponding to the score for that turn
            # store the accuracy in scores
            self.scores[(ent, prob, per)] = score
        # choose the combination of thresholds w/ the highest accuracy
        combo = max(self.scores, key=self.scores.get)
        return combo
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
import itertools
import numpy as np

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

        # self.entropies = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        # self.probabilities = [0.01, 0.05, 0.1, 0.3, 0.5]
        # self.perplexities = [0.01, 0.05, 0.1, 0.5]
        # self.scores = {}
    
    def calibrate(self):
        # an implementation of ERGO's calibration procedure
        # run the model over 80 held-out GSM8K examples
        # compute the change in average token-level entropy, probability, & perplexity at each turn
        # this gives us distributions of entropy, probability, & perplexity (1 per signal)
        # from these distributions of signal changes, select a threshold for each signal based on a percentile aligned with the model's baseline aptitude

        # get the held-out GSM8K examples, which should NOT be included in the examples that we will evaluate the model on
        dataset_path="sharded_dataset.json"
        dataset = GSM8K(dataset_path=dataset_path)
        # data structures to contain the distributions of signal changes
        changes_ent = []
        changes_prob = []
        changes_perpl = []
        # also track the model's baseline aptitude (peak/best-case performance capability, averaged across all tasks)
        aptitudes = []
        # run the model over the examples, computing the change in average of each signal at each turn
        output_path="calibration.json"
        evaluator = GSM8KEvaluator(output_file=output_path, dataset_path=dataset_path)
        # TODO: run the model in the regular way, not using ERGO w/ specific thresholds?
        ergo = Ergo(model=self.model)
        logger = Logger(model=self.model, dataset=dataset, output_path=output_path)
        runner = RunERGO(model=self.model, dataset=dataset, ergo=ergo, logger=logger, evaluator=evaluator, num_Qs=80, num_runs=1)
        runner.execute(clear_cache=self.clear_cache)
        # TODO: determine how to calculate the signal changes & baseline aptitude from running the model
        # determine the percentile
        baseline_aptitude = np.mean(aptitudes)
        # applying the percentile to the distributions, select & return the thresholds to be used by the model during evaluation
        threshold_ent = np.percentile(np.array(changes_ent), baseline_aptitude)
        threshold_prob = np.percentile(np.array(changes_prob), baseline_aptitude)
        threshold_perpl = np.percentile(np.array(changes_perpl), baseline_aptitude)
        return (threshold_ent, threshold_prob, threshold_perpl)

        # dataset_path="sharded_dataset.json"
        # output_path="calibration.json"
        # dataset = GSM8K(dataset_path=dataset_path)
        # evaluator = GSM8KEvaluator(output_file=output_path, dataset_path=dataset_path)
        # for thresholds in itertools.product(self.entropies, self.probabilities, self.perplexities):
        #     ent, prob, per = thresholds
        #     print((ent, prob, per)) # so we can keep track of the threshold combinations
        #     output_path_full=f"calibration_run0.json"
        #     # test this combination using the given model on the held-out set from GSM8K
        #     ergo = Ergo(model=self.model, threshold_H=ent, threshold_p=prob, threshold_PPL=per)
        #     logger = Logger(model=self.model, dataset=dataset, output_path=output_path)
        #     runner = RunERGO(model=self.model, dataset=dataset, ergo=ergo, logger=logger, evaluator=evaluator, num_Qs=20, num_runs=1)
        #     runner.execute(clear_cache=self.clear_cache)
        #     # read the output file & determine the accuracy
        #     with open(output_path_full, 'r') as file:
        #         content = file.read()
        #         pattern = r"\"score\": \\d" # pattern for "score": followed by a digit (0 or 1)
        #         matches = re.findall(pattern, content)
        #         # for each match, tally up the score
        #         score = 0
        #         for m in matches:
        #             score += int(m[-1]) # get the last character, which is a digit corresponding to the score for that turn
        #     # store the accuracy in scores
        #     self.scores[(ent, prob, per)] = score
        #     print(score)
        # # choose the combination of thresholds w/ the highest accuracy
        # combo = max(self.scores, key=self.scores.get)
        # return combo
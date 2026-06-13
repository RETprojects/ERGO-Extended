# TODO: find optimal signal thresholds
# inputs: model, signals, GSM8K examples (NOT to be used for testing later, so as to avoid contamination)
# tune multiple hyperparameters (signal thresholds) via grid search
# these thresholds will be used for the given model
# TODO: in runExperiment, call this file to tune thresholds before experiment begins
# these thresholds will be used to initialize the Ergo object that is used in the experiment
# callstack for an experiment: example_main -> runExperiment/run_[task] -> init ergo/Ergo object w/ thresholds
# it could even be called at the end of RunExperiment's init function! get the optimal signal thresholds before running the experiment

# imports needed: the types of models to be called (OpenAIModel, etc.), Ergo, RunERGO/execute, etc. (everything needed to run a model on GSM8K w/ signal thresholds)

class Calibration():
    # initialize a calibration for a specific model
    # define the hyperparameter (signal threshold) values to be tested in combinations
    # calibrate by running the model on GSM8K & assessing the results
    # store the score for each hyperparameter combination
    # highest accuracy -> choose that combination of signal thresholds

    def __init__(self, model_name):
        self.model_name = model_name
        self.entropies = [0.01**i if i > 0 else 0 for i in range(51)]
        self.probabilities = [0.01**i if i > 0 else 0 for i in range(51)]
        self.perplexities = [0.01**i if i > 0 else 0 for i in range(51)]
        self.scores = {}
    
    def calibrate(self):
        for ent, prob, per in self.entropies, self.probabilities, self.perplexities:
            # test this combination using the given model on the held-out set from GSM8K
            # store the accuracy in scores
            self.scores[(ent, prob, per)] = 0.0
        # choose the combination of thresholds w/ the highest accuracy
        combo = max(self.scores, key=self.scores.get)
        return combo
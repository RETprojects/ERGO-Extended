# TODO: find optimal signal thresholds
# inputs: model, signals, GSM8K examples (NOT to be used for testing later, so as to avoid contamination)
# tune multiple hyperparameters (signal thresholds) via grid search
# these thresholds will be used for the given model
# TODO: in runExperiment, call this file to tune thresholds before experiment begins
# these thresholds will be used to initialize the Ergo object that is used in the experiment
# callstack for an experiment: example_main -> runExperiment/run_[task] -> init ergo/Ergo object w/ thresholds

# imports needed: the types of models to be called (OpenAIModel, etc.), Ergo, RunERGO/execute, etc. (everything needed to run a model on GSM8K w/ signal thresholds)

class Calibration():
    # initialize a calibration for a specific model
    # define the hyperparameter (signal threshold) values to be tested in combinations
    # calibrate by running the model on GSM8K & assessing the results
    # store the score for each hyperparameter combination
    # highest accuracy -> choose that combination of signal thresholds

    def __init__(self, model_name):
        self.model_name = model_name
        self.entropies = []
        self.probabilities = []
        self.perplexities = []
        self.scores = {}
    
    def calibrate(self):
        return
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

        self.entropies = [0.01**i if i > 0 else 0 for i in range(51)]
        self.probabilities = [0.01**i if i > 0 else 0 for i in range(51)]
        self.perplexities = [0.01**i if i > 0 else 0 for i in range(51)]
        self.scores = {}
    
    def calibrate(self):
        best_acc = 0.0
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
            # store the accuracy in scores
            self.scores[(ent, prob, per)] = 0.0
        # choose the combination of thresholds w/ the highest accuracy
        combo = max(self.scores, key=self.scores.get)
        return combo
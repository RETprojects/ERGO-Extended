from core.model import OpenAIModel, LocalLLMModel, ClaudeModel
from core.dataset import GSM8K, Database, Code, Actions, DataToText, Summary
from core.ergo import Ergo
from core.utils import Logger
from generation.generator import RunERGO
from evaluation.evaluator import GSM8KEvaluator, DatabaseEvaluator, ActionsEvaluator, CodeEvaluator, DataToTextEvaluator, SummaryEvaluator

class RunExperiment():
    """
    Initialize an experiment with a model (local or OpenAI) and run on a specified dataset.
    """

    def __init__(self, model_name, device="cuda", device_map="auto", max_new_tokens=1024, dtype="float16", temperature=1.0, do_sample=True, openai=False, claude=False, clear_cache=True):
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

    def run_GSM8K(self, dataset_path, output_path, num_Qs = None, threshold_H=0.5, threshold_p=-0.05, threshold_PPL=15, num_runs=1):
        dataset = GSM8K(dataset_path=dataset_path)
        ergo = Ergo(model=self.model, threshold_H=threshold_H, threshold_p=threshold_p, threshold_PPL=threshold_PPL)
        logger = Logger(model=self.model, dataset=dataset, output_path=output_path)
        evaluator = GSM8KEvaluator(output_file=output_path, dataset_path=dataset_path)
        runner = RunERGO(model=self.model, dataset=dataset, ergo=ergo, logger=logger, evaluator=evaluator, num_Qs=num_Qs, num_runs=num_runs)
        runner.execute(clear_cache=self.clear_cache)

    def run_Database(self, dataset_path, spider_DB_path, num_Qs = None, threshold_H=0.5, threshold_p=-0.05, threshold_PPL=15, output_path=None, num_runs=1):
        dataset = Database(dataset_path=dataset_path)
        ergo = Ergo(model=self.model, threshold_H=threshold_H, threshold_p=threshold_p, threshold_PPL=threshold_PPL)
        logger = Logger(model=self.model, dataset=dataset, output_path=output_path)
        evaluator = DatabaseEvaluator(output_file=output_path, dataset_path=dataset_path)
        runner = RunERGO(model=self.model, dataset=dataset, ergo=ergo, logger=logger, evaluator=evaluator, num_Qs=num_Qs, num_runs=num_runs)
        runner.execute(spider_DB_path=spider_DB_path, clear_cache=self.clear_cache)

    def run_Code(self, dataset_path, num_Qs = None, threshold_H=0.5, threshold_p=-0.05, threshold_PPL=15, output_path=None, num_runs=1):
        dataset = Code(dataset_path=dataset_path)
        ergo = Ergo(model=self.model, threshold_H=threshold_H, threshold_p=threshold_p, threshold_PPL=threshold_PPL)
        logger = Logger(model=self.model, dataset=dataset, output_path=output_path)
        evaluator = CodeEvaluator(output_file=output_path, dataset_path=dataset_path)
        runner = RunERGO(model=self.model, dataset=dataset, ergo=ergo, logger=logger, evaluator=evaluator, num_Qs=num_Qs, num_runs=num_runs)
        runner.execute(clear_cache=self.clear_cache)

    def run_Actions(self, dataset_path, num_Qs = None, threshold_H=0.5, threshold_p=-0.05, threshold_PPL=15, output_path=None, num_runs=1):
        dataset = Actions(dataset_path=dataset_path)
        ergo = Ergo(model=self.model, threshold_H=threshold_H, threshold_p=threshold_p, threshold_PPL=threshold_PPL)
        logger = Logger(model=self.model, dataset=dataset, output_path=output_path)
        evaluator = ActionsEvaluator(output_file=output_path, dataset_path=dataset_path)
        runner = RunERGO(model=self.model, dataset=dataset, ergo=ergo, logger=logger, evaluator=evaluator, num_Qs=num_Qs, num_runs=num_runs)
        runner.execute(clear_cache=self.clear_cache)

    def run_DataToText(self, dataset_path, num_Qs = None, threshold_H=0.5, threshold_p=-0.05, threshold_PPL=15, output_path=None, num_runs=1):
        dataset = DataToText(dataset_path=dataset_path)
        ergo = Ergo(model=self.model, threshold_H=threshold_H, threshold_p=threshold_p, threshold_PPL=threshold_PPL)
        logger = Logger(model=self.model, dataset=dataset, output_path=output_path)
        evaluator = DataToTextEvaluator(output_file=output_path, dataset_path=dataset_path)
        runner = RunERGO(model=self.model, dataset=dataset, ergo=ergo, logger=logger, evaluator=evaluator, num_Qs=num_Qs, num_runs=num_runs)
        runner.execute(clear_cache=self.clear_cache)
    
    def run_Summary(self, dataset_path, num_Qs = None, threshold_H=0.5, threshold_p=-0.05, threshold_PPL=15, output_path=None, num_runs=1):
        dataset = Summary(dataset_path=dataset_path)
        ergo = Ergo(model=self.model, threshold_H=threshold_H, threshold_p=threshold_p, threshold_PPL=threshold_PPL)
        logger = Logger(model=self.model, dataset=dataset, output_path=output_path)
        evaluator = SummaryEvaluator(output_file=output_path, dataset_path=dataset_path)
        runner = RunERGO(model=self.model, dataset=dataset, ergo=ergo, logger=logger, evaluator=evaluator, num_Qs=num_Qs, num_runs=num_runs)
        runner.execute(clear_cache=self.clear_cache)
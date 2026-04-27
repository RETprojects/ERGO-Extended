# core/model.py
from openai import OpenAI
from anthropic import Anthropic
import math
import os
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
import torch
import sys
import json
import boto3
import base64
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# logging.set_verbosity_info()

class BaseModel:
    def __init__(self, model_name):
        """
        Base class for language models. Either local huggingface model or OpenAI API.
        if openai=True it uses OpenAI API.
        """
        self.model_name = model_name

    def generate(self, prompt):
        """
        Generate text using a local Hugging Face model or OpenAI API.
        Returns: (avg_entropy, avg_probability, perplexity, generated_text)
        """
        raise NotImplementedError

    def compute_entropy(self, output: Any) -> float:
        """
        Compute the average token-level entropy of the generated text.
        Returns: avg_entropy
        """
        raise NotImplementedError
    
    def compute_probability(self, output: Any) -> float:
        """
        Compute the average token-level probability of the generated text.
        Returns: avg_probability
        """
        raise NotImplementedError
    
    def compute_perplexity(self, output: Any) -> float:
        """
        Compute the perplexity of the generated text.
        Returns: perplexity
        """
        raise NotImplementedError
    

class LocalLLMModel(BaseModel):

    def __init__(self, model_name, device, max_new_tokens, dtype=torch.float16, temperature=1.0, do_sample=True, device_map="auto"):
        super().__init__(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name,  trust_remote_code=True)

        if device:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                trust_remote_code=True
            ).to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                trust_remote_code=True,
                device_map=device_map
            )

        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample

    def generate(self, messages):
        """
        Device-aware generation for single-device and device_map='auto' sharded models.
        Returns: (avg_entropy, avg_probability, perplexity, generated_text)
        """


        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=self.temperature,
            )

            scores = outputs.scores
            if scores:
                base_dev = scores[0].device
                logits = torch.stack([s.to(base_dev) for s in scores])
                probs = torch.softmax(logits, dim=-1)
                avg_entropy = self.compute_entropy(probs)
                avg_probability = self.compute_probability(probs)
                perplexity = self.compute_perplexity(probs)
            else:
                avg_entropy = None
                avg_probability = None
                perplexity = None

        response_ids = outputs.sequences
        input_len = inputs["input_ids"].shape[1]
        new_tokens = response_ids[:, input_len:].cpu()
        response_only = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

        return avg_entropy, avg_probability, perplexity, response_only

    # Shannon entropy
    def compute_entropy(self, probs):
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        avg_entropy = entropy.mean().item()
        return avg_entropy
    
    # token probability
    def compute_probability(self, probs):
        probability = torch.exp(torch.log(probs + 1e-8))
        avg_probability = probability.mean().item()
        return avg_probability
    
    # perplexity
    def compute_perplexity(self, probs):
        inv_probability = 1.0 / torch.exp(torch.log(probs + 1e-8))
        perplexity = inv_probability.mean().item()
        return perplexity


class OpenAIModel(BaseModel):

    def __init__(self, model_name, temperature, max_tokens, top_logprobs: int = 20):
        super().__init__(model_name)
        self.client = OpenAI(api_key = os.getenv("OPENAI_KEY"))
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_logprobs = top_logprobs

    def generate(self, prompt: List[Dict[str, str]]):
        """
        Sends a prompt to OpenAI Chat API and returns:
        (average_entropy, average_probability, perplexity, generated_text)
        """

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            max_completion_tokens=self.max_tokens,
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=self.top_logprobs,
        )

        generated_text = resp.choices[0].message.content
        try: 
            token_logprobs = resp.choices[0].logprobs.content
        except:
            sys.exit("ERROR: INPUTTED OPENAI MODEL DOESNT RETURN LOGPROBS")

        avg_entropy = self.compute_entropy(token_logprobs)
        avg_probability = self.compute_probability(token_logprobs)
        perplexity = self.compute_perplexity(token_logprobs)
        # tokens_used = resp.usage.completion_tokens + resp.usage.prompt_tokens

        return avg_entropy, avg_probability, perplexity, generated_text#, tokens_used

    # from lost-in-conversation/model-openai.py
    # def generate_json(self, messages, model="gpt-4o-mini", **kwargs):
    #     response = self.generate(messages, model, is_json=True, **kwargs)
    #     response["message"] = json.loads(response["message"])
    #     return response

    def compute_entropy(self, token_logprobs):
        entropies = []
        for token_info in token_logprobs:
            entropy = 0.0
            for logprob in token_info.top_logprobs:
                p = np.exp(logprob.logprob)
                entropy += -p * logprob.logprob
            entropies.append(entropy)

        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
        return avg_entropy
    
    def compute_probability(self, token_logprobs):
        probabilities = []
        for token_info in token_logprobs:
            probability = 1.0
            for logprob in token_info.top_logprobs:
                probability *= logprob.logprob
            probabilities.append(np.exp(probability))

        avg_probability = sum(probabilities) / len(probabilities) if probabilities else 0.0
        return avg_probability
    
    def compute_perplexity(self, token_logprobs):
        inv_probabilities = []
        for token_info in token_logprobs:
            probability = 1.0
            for logprob in token_info.top_logprobs:
                probability *= logprob.logprob
            inv_probabilities.append(1.0 / np.exp(probability))

        perplexity = sum(inv_probabilities) / len(inv_probabilities) if inv_probabilities else 0.0
        return perplexity

class ClaudeModel(BaseModel):

    def __init__(self, model_name, temperature, max_tokens, top_logprobs: int = 20):
        super().__init__(model_name)
        # self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"),)
        # self.client = OpenAI(
        #     api_key=os.environ.get("ANTHROPIC_API_KEY"),  # Your Claude API key
        #     base_url="https://api.anthropic.com/v1/",  # the Claude API endpoint
        # )
        self.client = boto3.client('bedrock-runtime', region_name='eu-west-3')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_logprobs = top_logprobs

    def generate(self, prompt: List[Dict[str, str]], system: str):
        """
        Sends a prompt to Claude API and returns:
        (average_entropy, average_probability, perplexity, generated_text)
        """

        # resp = self.client.chat.completions.create(
        #     model="claude-sonnet-4-6",  # Claude model name
        #     messages=prompt,
        #     max_completion_tokens=self.max_tokens,
        #     temperature=self.temperature,
        #     logprobs=True,
        #     top_logprobs=self.top_logprobs,
        # )

        # thanks to the Amazon Bedrock docs: https://docs.aws.amazon.com/bedrock/latest/userguide/custom-model-import-advanced-features.html

        payload = {
            # "system": system,
            "anthropic_version": "bedrock-2023-05-31",
            "messages": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            "prompt_logprobs": 1
        }

        response = self.client.invoke_model(
            modelId=self.model_name,
            body=json.dumps(payload),
            accept='application/json',
            contentType='application/json'
        )

        resp = json.loads(response['body'].read())

        generated_text = resp.choices[0].message.content
        try: 
            token_logprobs = resp.choices[0].logprobs.content
        except:
            sys.exit("ERROR: INPUTTED CLAUDE MODEL DOESNT RETURN LOGPROBS")

        avg_entropy = self.compute_entropy(token_logprobs)
        avg_probability = self.compute_probability(token_logprobs)
        perplexity = self.compute_perplexity(token_logprobs)
        # tokens_used = resp.usage.completion_tokens + resp.usage.prompt_tokens

        return avg_entropy, avg_probability, perplexity, generated_text#, tokens_used

    # from lost-in-conversation/model-openai.py
    # def generate_json(self, messages, model="gpt-4o-mini", **kwargs):
    #     response = self.generate(messages, model, is_json=True, **kwargs)
    #     response["message"] = json.loads(response["message"])
    #     return response

    def compute_entropy(self, token_logprobs):
        entropies = []
        for token_info in token_logprobs:
            entropy = 0.0
            for logprob in token_info.top_logprobs.values():
                p = math.exp(logprob)
                entropy += -p * logprob
            entropies.append(entropy)

        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
        return avg_entropy
    
    def compute_probability(self, token_logprobs):
        probabilities = []
        for token_info in token_logprobs:
            probability = 1.0
            for logprob in token_info.top_logprobs.values():
                probability *= logprob
            probabilities.append(math.exp(probability))

        avg_probability = sum(probabilities) / len(probabilities) if probabilities else 0.0
        return avg_probability
    
    def compute_perplexity(self, token_logprobs):
        inv_probabilities = []
        for token_info in token_logprobs:
            probability = 1.0
            for logprob in token_info.top_logprobs.values():
                probability *= logprob
            inv_probabilities.append(1.0 / math.exp(probability))

        perplexity = sum(inv_probabilities) / len(inv_probabilities) if inv_probabilities else 0.0
        return perplexity
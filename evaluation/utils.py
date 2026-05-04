from .eval_bfcl import ast_checker, ast_parse
import re
import sacrebleu
import sys
import tempfile
import subprocess
import sqlite3
import json
import os
import json, re
import numpy as np
import time
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

from openai import OpenAI

from dotenv import load_dotenv
import numpy as np

load_dotenv()


class EvalUtils:
    """
    Each class includes utility functions used to extract relevant parts of model output 
    and/or to judge correctness.
    """
    def __init__(self):
        pass

class ActionsEvalUtils(EvalUtils):
    def __init__(self):
        super().__init__()

    def extract_function_block(self, text):
        """
        FUNCTION TAKEN ENTIRELY FROM https://github.com/microsoft/lost_in_conversation -- https://arxiv.org/pdf/2505.06120
        """
        start = text.find('[')
        if start == -1:
            return ''

        level = 0
        for i in range(start, len(text)):
            if text[i] == '[':
                level += 1
            elif text[i] == ']':
                level -= 1
                if level == 0:
                    block = text[start:i+1]
                    return self.clean_function_block(block)
        return ''

    def clean_function_block(self, block):
        """
        FUNCTION TAKEN ENTIRELY FROM https://github.com/microsoft/lost_in_conversation -- https://arxiv.org/pdf/2505.06120
        """
        block = block.replace('\n', '').replace('\r', '').replace('\t', '')
        block = ' '.join(block.split())

        # Remove "..." wrapping function calls only
        block = re.sub(r'"\s*([a-zA-Z_][a-zA-Z0-9_\.]*\s*\([^"]*\))\s*"', r'\1', block)

        # Remove space after [ and before ]
        block = re.sub(r'\[\s+', '[', block)
        block = re.sub(r'\s+\]', ']', block)

        return block


    def extract_all_function_blocks(self, text):
        """
        FUNCTION TAKEN ENTIRELY FROM https://github.com/microsoft/lost_in_conversation -- https://arxiv.org/pdf/2505.06120
        """
        blocks = []
        start_positions = [i for i, c in enumerate(text) if c == '[']

        for start in start_positions:
            level = 0
            found = False
            for i in range(start, len(text)):
                if text[i] == '[':
                    level += 1
                elif text[i] == ']':
                    level -= 1
                    if level == 0:
                        block = text[start:i+1]
                        blocks.append(self.clean_function_block(block))
                        found = True
                        break
            if found:
                # Skip any nested [ inside this block — move to next outer [
                continue
        return blocks

    def evaluator_function(self, predicted_answer, sample):
        """
        FUNCTION TAKEN ENTIRELY FROM https://github.com/microsoft/lost_in_conversation -- https://arxiv.org/pdf/2505.06120
        Evaluate if the predicted function call matches the expected format and functionality.
        """

        try:
            decoded_output = ast_parse(predicted_answer.strip(), sample["language"])
        except Exception as e:
            return {"is_correct": False, "error": "Failing to parse the predicted answer as an AST"}

        result = ast_checker(
            sample["function"],
            decoded_output,
            sample["reference_answer"],
            sample["language"],
            sample["test_category"],
            "gpt-oss-120b"
        )
        score = 1 if result["valid"] else 0
        return {"is_correct": result["valid"], "score": score, "error": result["error"]}

class DataToTextEvalUtils(EvalUtils):
    def __init__(self):
        super().__init__()

    def D2T_evaluator_function(self, extracted_answer, sample):
        """
        Evaluates the extracted answer against the references using BLEU score.
        Returns a score between 0 and 1.
        """
        # ToTTo has multiple references per example
        references = sample["references"]
        bleu = sacrebleu.corpus_bleu([extracted_answer.strip()], [[ref.strip()] for ref in references])
        return bleu.score / 100.0
    
class DatabaseEvalUtils(EvalUtils):
    def __init__(self):
        super().__init__()


    def extract_sql_query(self, text):
        """
        Extract the first SQL query from a text blob.
        Looks for fenced code blocks (```sql ... ```) first.
        If none found, returns None."""
        # Match content inside ```sql ... ``` block
        match = re.search(r'```sql(.*?)```', text, re.DOTALL | re.IGNORECASE)
        if match:
            # Clean leading/trailing whitespace
            return match.group(1).strip()
        else:
            return None


    def extract_sql_queries(self, text):
        """
        Extract all SQL queries from a text blob.
        Captures both fenced code blocks (```sql ... ```) and standalone statements ending with semicolons.
        """
        queries = []

        # 1) Fenced SQL blocks
        fenced = re.findall(r'```sql\s*(.*?)```', text, flags=re.IGNORECASE | re.DOTALL)
        queries.extend(q.strip() for q in fenced if q.strip())

        # 2) Standalone statements (SELECT/INSERT/UPDATE/DELETE/CREATE/ALTER/DROP) ending with ;
        #    Avoid re-capturing fenced blocks by skipping matches that span lines containing ```
        stmt_pattern = re.compile(
            r'(?:\b(?:SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b.+?;)',
            flags=re.IGNORECASE | re.DOTALL
        )
        for m in stmt_pattern.finditer(text):
            snippet = m.group().strip()
            # skip if it's identical to a fenced block
            if not any(snippet == f.strip() or snippet in f for f in fenced):
                queries.append(snippet)
        # if not queries:
        #   queries.append(text)
        return queries

    def run_query(self, db_path, sql):
      """
      Runs the given SQL query on the SQLite database at db_path.
      Returns the query results as a list of tuples.
    """
      conn = sqlite3.connect(db_path)
      # decode any bytes with replacement on errors
      conn.text_factory = lambda b: b.decode('utf-8', errors='replace')
      cur  = conn.cursor()
      try:
          cur.execute(sql)
          rows = cur.fetchall()
      finally:
          conn.close()
      return rows
    
class CodeEvalUtils(EvalUtils):
    def __init__(self):
        super().__init__()

    def extract_all_function_blocks_and_names(self, code):
        """
        Extracts all top-level Python function blocks (def ...) along with any import
        statements that appear immediately before each function. Returns a list of tuples:
        [(full_code_with_imports, function_name), ...].
        """
        lines = code.strip().splitlines()
        n = len(lines)
        results = []
        import_lines = []
        i = 0

        while i < n:
            line = lines[i]

            # If this line is an import, collect it and move on
            if re.match(r'^\s*import\s+\w', line) or re.match(r'^\s*from\s+\w+\s+import\s+', line):
                import_lines.append(line)
                i += 1
                continue

            # If this line is a top‐level function definition
            func_match = re.match(r'^(\s*)def\s+([A-Za-z_]\w*)\s*\(.*\)\s*:', line)
            if func_match:
                func_indent = len(func_match.group(1))
                func_name = func_match.group(2)

                # Collect the entire function block
                func_block = [lines[i]]
                j = i + 1
                while j < n:
                    next_line = lines[j]
                    # Blank lines inside the block are allowed
                    if next_line.strip() == "":
                        func_block.append(next_line)
                        j += 1
                        continue

                    # Check indentation: if indent > func_indent, it's still inside
                    indent_level = len(next_line) - len(next_line.lstrip())
                    if indent_level > func_indent:
                        func_block.append(next_line)
                        j += 1
                    else:
                        break

                # Combine the imports collected so far with this function block
                full_code = "\n".join(import_lines + [""] + func_block).rstrip()
                results.append((full_code, func_name))

                # Reset import_lines for the next function
                import_lines = []
                # Continue scanning from the line after this function block
                i = j
                continue

            # Neither an import nor a function definition: move on
            i += 1

        return results
    

    def extract_first_function_block_and_name(self, code):
        """
        Extracts the first top-level Python function (def ...) block and its name,
        along with any import statements above it.
        Returns the full function code (with imports) and function name.
        """
        lines = code.strip().splitlines()
        import_lines = []
        func_start_idx = None
        func_indent = None
        func_name = None

        # Collect top-level import statements before the function
        for i, line in enumerate(lines):
            if re.match(r'^\s*import\s+\w', line) or re.match(r'^\s*from\s+\w+\s+import\s+', line):
                import_lines.append(line)
            # elif re.match(r'^\s*def\s+[a-zA-Z_]\w*\s*\(.*\)\s*:', line):
            #     func_start_idx = i
            #     match = re.match(r'^(\s*)def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', line)
            #     func_indent = len(match.group(1))
            #     func_name = match.group(2)
            #     break
            elif re.match(r'^\s*def\s+[a-zA-Z_]\w*\s*\(.*\)\s*(->\s*[^\s:]+)?\s*:', line):
                func_start_idx = i
                match = re.match(r'^(\s*)def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(.*\)\s*(->\s*[^\s:]+)?\s*:', line)
                func_indent = len(match.group(1))
                func_name = match.group(2)
                break

        if func_start_idx is None:
            return None, None

        # Collect lines in the function block
        func_lines = lines[func_start_idx:]
        collected = [func_lines[0]]

        for line in func_lines[1:]:
            if line.strip() == "":
                collected.append(line)
            elif len(line) - len(line.lstrip()) >= func_indent + 1:
                collected.append(line)
            else:
                break

        return "\n".join(import_lines + [""] + collected).rstrip(), func_name


    def run_function_and_check(self, func_name, user_code, test_cases):
        """
        Runs the llm generated code against provided test cases.
        Each test case is a dict with "input" (string of args, one per line)
        and "output" (expected output).
        """
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            user_code_path = f.name
            f.write("import math\n")
            f.write(user_code)

        runner_code = f"""
import json
import sys
import ast
import math
import importlib.util

spec = importlib.util.spec_from_file_location("tempmod", "{user_code_path}")
tempmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tempmod)

failures = []
def parse_argument(line):
    line = line.strip()
    if line.startswith('"') and line.endswith('"'):
        return line[1:-1]  # Strip outer double quotes, treat as string
    try:
        return ast.literal_eval(line)
    except:
        return line  # fallback

for idx, case in enumerate({test_cases}):
    input_lines = case["input"].splitlines()
    args = [parse_argument(line) for line in input_lines]

    try:
        expected = ast.literal_eval(case["output"])
    except:
        expected = case["output"]

    if expected == "true" or expected == "Yes" or expected == "yes":
        expected = True

    if expected == "false" or expected == "No" or expected == "no":
        expected = False

    try:
        got = getattr(tempmod, "{func_name}")(*args)
    except Exception as e:
        failures.append(f"{{args}} raised {{e!r}}")
        continue

    if got == "Yes" or got == "yes":
        got = True
    if got == "No" or got == "no":
        got = False

    if got and got != expected:
        args = args.reverse()
        try:
            got = getattr(tempmod, "{func_name}")(*args)
        except Exception as e:
            failures.append(f"{{args}} raised {{e!r}}")
            continue

        if got == "Yes" or got == "yes":
            got = True
        if got == "No" or got == "no":
            got = False

        if got != expected:
            failures.append(f"Test {{idx}} :- {{args}}: got={{got!r}}, expected={{expected!r}}")

if failures:
    print(failures)
    sys.exit(failures)
else:
    sys.exit(0)
    """

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            runner_path = f.name
            f.write(runner_code)

        try:
            result = subprocess.run(
                [sys.executable, runner_path],
                capture_output=True,
                text=True,
                timeout=3
            )
        except:
            return False

        if result.returncode == 0:
            return True
        else:
            return False

class SummaryEvalUtils(EvalUtils):
    def __init__(self):
        super().__init__()
        self.client = OpenAI(api_key=os.getenv("OPENAI_KEY"))


    def evaluator_function(self, extracted_answer, sample):
        evaluator_model_card = "gpt-4.1-mini-2025-04-14"
        evals = self.evaluate_insights(sample["insights"], extracted_answer, evaluator_model_card, os.path.abspath("evaluation/eval_summhay.txt"))
        # eval should likely be cached somewhere, so results can be explained if needed
        results = self.compute_single_sample_results(extracted_answer, evals, sample["insightid2ref_citations"])
        # results["score"] = results["coverage_score"]
        results["score"] = results["joint_score"] # we save this as the main score we anchor on
        return results

    def summary2bullets(self, summary, max_summary_length=300):
        bullets = summary.split("\n")
        
        # Count words in each bullet (using space counting)
        bullet_word_counts = [len(bullet.split()) for bullet in bullets]
        total_words = sum(bullet_word_counts)
        
        # If we're under the limit, return as is
        if total_words <= max_summary_length:
            return {"bullets": bullets, "trim_ratio": 0.0}
            
        # Calculate percentage to trim
        excess_percentage = (total_words - max_summary_length) / total_words
        
        # Trim each bullet proportionally
        trimmed_bullets = []
        for bullet, word_count in zip(bullets, bullet_word_counts):
            if word_count == 0:
                trimmed_bullets.append(bullet)
                continue
                
            # Calculate how many words to keep
            words_to_keep = int(word_count * (1 - excess_percentage))
            if words_to_keep < 1:
                words_to_keep = 1
                
            # Split into words and rejoin
            words = bullet.split()
            trimmed_bullet = " ".join(words[:words_to_keep])
            trimmed_bullets.append(trimmed_bullet)
        
        return {"bullets": trimmed_bullets, "trim_ratio": excess_percentage}

    def evaluate_insights(self, insights, summary, evaluator_model_card, eval_prompt_fn="eval_summhay.txt"):
        with open(eval_prompt_fn, "r") as f:   
            prompt_eval = f.read()

        bullets_obj = self.summary2bullets(summary)
        bullets = bullets_obj["bullets"]
        bullets_str = json.dumps({"bullets": [{"bullet_id": i+1, "text": bullet} for i, bullet in enumerate(bullets)]}, indent=1)
        insight_scores = []
        for insight in insights:
            response_all = self.generate_json(messages=[{"role": "user", "content": prompt_eval}], model=evaluator_model_card, return_metadata=True, variables={"BULLETS": bullets_str, "INSIGHT": insight["insight"]})
            response_json  = response_all["message"]
            response_json["insight_id"] = insight["insight_id"] 
            insight_scores.append(response_json)
        return insight_scores

    def build_ref_insight2docids(self, topic):
        insight_id2references = {}
        for i, doc in enumerate(topic["documents"]):
            doc_id = i + 1
            for insight_id in doc["insights_included"]:
                if insight_id not in insight_id2references:
                    insight_id2references[insight_id] = set([])
                insight_id2references[insight_id].add(doc_id)

        insight_id2references = {k: list(v) for k, v in insight_id2references.items()} # make it into a list
        return insight_id2references

    def extract_citations(self, bullet):
        # matches digits or commas
        matches = re.findall(r"\[([\d, ]+)\]", bullet)
        ref_ids = []
        for match in matches:
            ref_ids += [int(m.strip()) for m in match.split(",") if len(m.strip()) > 0]
        return ref_ids

    def compute_single_sample_scores(self, summary, evals, insightid2ref_citations, partial_score=0.5, cite_offset=0): # the cite offset should be one for the annotators (but not for the model eval)
        bullets_obj = self.summary2bullets(summary)
        bullets = bullets_obj["bullets"]
        trim_ratio = bullets_obj["trim_ratio"]

        coverage_scores, citation_scores, joint_scores = [], [], []
        citation_precisions, citation_recalls = [], []
        for e in evals:
            cov_score, cit_score, cit_prec, cit_rec = 0.0, 0.0, 0.0, 0.0
            if e["coverage"] in ["PARTIAL_COVERAGE", "FULL_COVERAGE"]:
                cov_score = 1.0 if e["coverage"] == "FULL_COVERAGE" else partial_score
                insight_id = e["insight_id"]
                try:
                    bullet_match_idx = int(e["bullet_id"])
                except:
                    bullet_match_idx = -1
                bullet_match = bullets[bullet_match_idx - 1]

                gen_citations = set([cite+cite_offset for cite in self.extract_citations(bullet_match)])
                ref_citations = set(insightid2ref_citations[insight_id])

                P = 0 if len(gen_citations) == 0 else len(gen_citations & ref_citations) / len(gen_citations)
                R = 0 if len(ref_citations) == 0 else len(gen_citations & ref_citations) / len(ref_citations)
                F1 = 0 if P + R == 0 else 2 * P * R / (P + R)
                cit_prec, cit_rec, cit_score = P, R, F1
                citation_scores.append(cit_score)
                citation_precisions.append(cit_prec)
                citation_recalls.append(cit_rec)

            coverage_scores.append(cov_score)
            joint_scores.append(cov_score * cit_score)
        return {"coverage_score": coverage_scores, "citation_score": citation_scores, "joint_score": joint_scores, "citation_precision": citation_precisions, "citation_recall": citation_recalls, "trim_ratio": trim_ratio}

    def compute_single_sample_results(self, summary, evals, insightid2ref_citations, partial_score=0.5, cite_offset=0):
        scores = self.compute_single_sample_scores(summary, evals, insightid2ref_citations, partial_score, cite_offset=cite_offset)
        return {k: np.mean(v).item() for k, v in scores.items()}
    
    # from lost-in-conversation/model-openai.py
    def generate_json(self, messages, model="gpt-4o-mini", **kwargs):
        response = self.generate(messages=messages, model=model, is_json=True, **kwargs)
        response["message"] = json.loads(response["message"])
        return response
    
    def generate(self, messages, model="gpt-4o-mini", timeout=30, max_retries=3, temperature=1.0, is_json=False, return_metadata=False, max_tokens=None, variables={}, **kwargs):
        if not ('kwargs' in locals() or 'kwargs' in globals()):
            kwargs = {}
        if is_json:
            kwargs["response_format"] = { "type": "json_object" }
        N = 0

        messages = self.format_messages(messages, variables)

        # o1- models do not support system message. If the first message is a system message, and the second message is a user message, then prepend the user message with the system message.
        if model.startswith("o1") and len(messages) > 1 and messages[0]["role"] == "system" and messages[1]["role"] == "user":
            system_message = messages[0]["content"]
            messages[1]["content"] = f"System Message: {system_message}\n{messages[1]['content']}"
            messages = messages[1:]

        while True:
            try:
                response = self.client.chat.completions.create(model=model, messages=messages, timeout=timeout, max_completion_tokens=max_tokens, temperature=temperature, **kwargs)
                break
            except:
                N += 1
                if N >= max_retries:
                    raise Exception("Failed to get response from OpenAI")
                else:
                    time.sleep(4)

        response = response.to_dict()
        usage = response['usage']
        response_text = response["choices"][0]["message"]["content"]
        # total_usd = self.cost_calculator(model, usage)
        prompt_tokens_cached = 0
        if 'prompt_tokens_details' in usage:
            prompt_tokens_cached = usage['prompt_tokens_details']['cached_tokens']

        if not return_metadata:
            return response_text
        return {"message": response_text, "total_tokens": usage['total_tokens'], "prompt_tokens": usage['prompt_tokens'], "prompt_tokens_cached": prompt_tokens_cached, "completion_tokens": usage['completion_tokens']}

    def format_messages(self, messages, variables={}):
        last_user_msg = [msg for msg in messages if msg["role"] == "user"][-1]

        for k, v in variables.items():
            key_string = f"[[{k}]]"
            if key_string not in last_user_msg["content"]:
                print(f"[prompt] Key {k} not found in prompt; effectively ignored")
            assert type(v) == str, f"[prompt] Variable {k} is not a string"
            last_user_msg["content"] = last_user_msg["content"].replace(key_string, v)

        # find all the keys that are still in the prompt using regex [[STR]] where STR is alnum witout space
        keys_still_in_prompt = re.findall(r"\[\[([^\]]+)\]\]", last_user_msg["content"])
        if len(keys_still_in_prompt) > 0:
            print(f"[prompt] The following keys were not replaced: {keys_still_in_prompt}")

        return messages
    
    def cost_calculator(self, model, usage, is_batch_model=False):
        is_finetuned, base_model = False, model
        if model.startswith("ft:gpt"):
            is_finetuned = True
            base_model = model.split(":")[1]

        prompt_tokens = usage['prompt_tokens']
        if 'prompt_tokens_details' in usage:
            prompt_tokens_cached = usage['prompt_tokens_details']['cached_tokens']
        else:
            prompt_tokens_cached = 0
        prompt_tokens_non_cached = prompt_tokens - prompt_tokens_cached

        completion_tokens = usage['completion_tokens']
        if base_model.startswith("gpt-4o-mini"):
            if is_finetuned:
                inp_token_cost, out_token_cost = 0.0003, 0.00015
            else:
                inp_token_cost, out_token_cost = 0.00015, 0.0006
        elif base_model.startswith("gpt-4o"):
            if is_finetuned:
                inp_token_cost, out_token_cost = 0.00375, 0.015
            else:
                inp_token_cost, out_token_cost = 0.0025, 0.01
        elif base_model.startswith("gpt-3.5-turbo"):
            inp_token_cost, out_token_cost = 0.0005, 0.0015
        elif base_model.startswith("o1-mini"):
            inp_token_cost, out_token_cost = 0.003, 0.012
        elif base_model.startswith("gpt-4.5-preview"):
            inp_token_cost, out_token_cost = 0.075, 0.150
        elif base_model.startswith("o1-preview") or base_model == "o1":
            inp_token_cost, out_token_cost = 0.015, 0.06
        else:
            raise Exception(f"Model {model} pricing unknown, please add")

        cache_discount = 0.5 # cached tokens are half the price
        batch_discount = 0.5 # batch API is half the price
        total_usd = ((prompt_tokens_non_cached + prompt_tokens_cached * cache_discount) / 1000) * inp_token_cost + (completion_tokens / 1000) * out_token_cost
        if is_batch_model:
            total_usd *= batch_discount

        return total_usd
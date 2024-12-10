import pdb
import numpy
from openai import OpenAI
import replicate
from .config import OPENAI_API_KEY, REPLICATE_API_TOKEN, MODEL
from .utils import *
import re
import vllm
from transformers import AutoTokenizer, AutoModelForCausalLM


client = OpenAI(api_key=OPENAI_API_KEY)


system = 'You will be provided with a Coq proof state and related definitions and lemmas, your task is to give a proof.'

instruction = '''I will give you a Coq proof state, including both hypotheses and a specific goal and your need to prove it. Your response should be a singular code block of Coq proof starting with "```coq\n", ending with "Qed.```", without any additional explanation or commentary. Follow to these guidelines:
Introduce variables using unique names to avoid any conflicts.
Keep each command distinct and separated, avoid concatenations like ';' or '[cmd|cmd]'.
Organize your proof with bullets like '-', '+', and '*' instead of braces ({, }). Shift to their double symbols like '--' and '++', when necessary.
Effectively use given premises, follow the syntax and structure demonstrated in the examples provided.
'''

examples = '''
Example 1:

Hypotheses:
n, m: nat
IHn: m + n = n + m

Goal:
m + S n = S n + m

Your Response:
```coq
simpl. rewrite <- IHn. auto.
Qed.```

Example 2:
Hypotheses:

Goal:
forall n m : nat, m + n = n + m

Your Response:
```coq
intros n m. induction n.
- simpl. auto.
- simpl. rewrite <- IHn. auto.
Qed.```'''


def process_response(response: str):
    pattern = r'```coq\n(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0]
    else:
        return response


def check_response(response: str):
    if not (response.startswith('```coq\n') and response.endswith('```')):
        return False
    return True

def load_model(name):

    #pdb.set_trace()
    model = vllm.LLM(
            model=name,
            tensor_parallel_size=1,
            dtype='bfloat16',
            #max_num_batched_tokens=32768,
            trust_remote_code=True,
            enforce_eager=True
        )
    tokenizer = AutoTokenizer.from_pretrained(name,trust_remote_code=True)
    return model, tokenizer

def query_ours(prompt,model_name):
    #print(prompt)
    #exit()
    temperature=0.75
    max_tokens= 1500
    stop=['<|im_end|>',]
    
    model, tokenizer = load_model(model_name)
    params = vllm.SamplingParams(
        n=5,
        temperature=temperature,
        #use_beam_search=temperature==0.0,
        max_tokens=max_tokens,
        stop=stop,
        repetition_penalty=1.0
    )
    outputs = model.generate([prompt], params, use_tqdm=False)
    #print(prompt,outputs)
    #print('outputs',outputs)
    if len(outputs) == 0:
        return [], []
    for output in outputs[0].outputs:
        text = output.text.replace(tokenizer.eos_token, '')
    
    #print(text)
    #exit()
    return process_response(text)


def chat_template_to_prompt(prompt_list):
    result = ""
    total_step = len(prompt_list)
    for i, message in enumerate(prompt_list):
        result += ('<|im_start|>' + message['role'] +
                   '\n' + message['content'])
        if i+1 != total_step:
            result += '<|im_end|>\n'
        elif message['role'] == 'user':
            result += '<|im_end|>\n<|im_start|>assistant\n'
    return result

class LLM:
    def __init__(self, max_tokens=7168):
        self.max_tokens = max_tokens
        self.retry_limit = 2
        self.api_key: str
        self.history_query = []
        self.history_response = []
        self.messages = [
                {"role": "system", "content": system},
            ]
        self.length = [trim_prompt(p['content'])[0] for p in self.messages]

    #只用了 goals, premises[:State.max_lemmas], defs
    def get_prompt(self, goals, premises, defs):
        if 'suozhi' not in MODEL:
            prompt = instruction + examples
            assert goals != [], 'goals is [] before querying'
            
            goal = goals[0]
            prompt += '\n\nSolve This Proof State:\n\n'
            prompt += 'Hypotheses:\n{hypos}\n\nGoal:\n{goal}\n\n'\
                        .format(hypos='\n'.join([f'{k}: {v}' for k, v in goal['hypos'].items()]) if goal['hypos'] else 'None', goal=goal['goal'])
            if defs != [] or premises != []:
                prompt += 'Premises:'
            if defs != []:
                num_tokens = self.max_tokens - sum(self.length) - trim_prompt(prompt)[0]
                tokens_one = int(num_tokens/(len(defs)+len(premises)))
                prompt += '\n{defs}' \
                    .format(defs= '\n'.join([trim_prompt(d, tokens_one)[1] for d in defs]))
            if premises != []:
                num_tokens = self.max_tokens - sum(self.length) - trim_prompt(prompt)[0]
                tokens_one = int(num_tokens/len(premises))
                prompt += '\n{lemmas}' \
                    .format(lemmas= '\n'.join([trim_theorem(p, tokens_one)[1] for p in premises]))
            return prompt
        elif "iter_1800_hf" in MODEL:
            #content="Solve This Proof State:\n\nHypotheses:\nK: Ordered.type\nV: Equality.type\nU: UMC.type K (Equality.sort V)\nk: Ordered.sort K\nv: Equality.sort V\nf: PCM.sort (@union_map_classPCM K (Equality.sort V) (union_mapUMC K (Equality.sort V)))\n\nGoal:\n@eq bool (@eq_op (union_map_eqType K V) (@PCM.join (@union_map_classPCM K (Equality.sort V) (union_mapUMC K (Equality.sort V))) (@um_pts K (Equality.sort V) k v) f) (@PCM.unit (union_mapPCM K (Equality.sort V)))) false\n\n"
            assert goals != [], 'goals is [] before querying'
            
            goal = goals[0]
            #print(defs)
            content="Solve This Proof State:\n\nHypotheses:\n\nGoal:\n{goal}\n\n".format(hypos='\n'.join([f'{k}: {v}' for k, v in goal['hypos'].items()]) if goal['hypos'] else '', goal=goal['goal'])
            #print(content)
            #exit()
            prompt=chat_template_to_prompt([{'role': 'user', 'content': content}])
            return prompt
        elif "5f0b02c75b57c5855da9ae460ce51323ea669d8a" in MODEL:# llama 8B instruct without retrivel
            prompt = instruction + examples
            assert goals != [], 'goals is [] before querying'
            
            goal = goals[0]
            prompt += '\n\nSolve This Proof State:\n\n'
            prompt += 'Hypotheses:\n{hypos}\n\nGoal:\n{goal}\n\n'\
                        .format(hypos='\n'.join([f'{k}: {v}' for k, v in goal['hypos'].items()]) if goal['hypos'] else 'None', goal=goal['goal'])
            return prompt

    def query(self, prompt: str):
        #print('querying')
        #exit()
        if MODEL.startswith('gpt'):
            return self.query_gpt(prompt)
        elif 'llama' in MODEL:
            return self.query_llama(prompt)
        elif 'suozhi' in MODEL:
            return query_ours(prompt,MODEL)
        


        
    def query_llama(self, prompt: str):
        length, prompt = trim_prompt(prompt, self.max_tokens-sum(self.length))
        self.messages.append({'role': 'user', 'content': prompt})
        self.length.append(length)

        input = {
            "max_tokens": 1000,
            "min_tokens": 0,
            "temperature": 0.75,
            "system_prompt": system,
            "prompt": prompt,
            "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou will be provided with a Coq proof state and related definitions and lemmas, your task is to give a proof.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "length_penalty": 1,
            "stop_sequences": "<|end_of_text|>,<|eot_id|>",
            "presence_penalty": 1.15,
            "log_performance_metrics": False
        }

        output = replicate.run(
            MODEL,
            input=input
        )
        output = ''.join(output)
        print(output)
        self.messages.append({'role': 'assistant', 'content': output})
        return process_response(output)


    def query_gpt(self, prompt: str):
        length, prompt = trim_prompt(prompt, self.max_tokens-sum(self.length))
        self.messages.append({'role': 'user', 'content': prompt})
        self.length.append(length)
        # gpt-4o-2024-05-13
        response = client.chat.completions.create(model=MODEL,
            messages=self.messages)
        role = response.choices[0].message.role
        content = response.choices[0].message.content
        self.messages.append({'role': role, 'content': content})
        return process_response(content)
    

    def batch_record(self, custom_id, prompt):
        length, prompt = trim_prompt(prompt, self.max_tokens-sum(self.length))
        self.messages.append({'role': 'user', 'content': prompt})
        record = {
            "custom_id": custom_id, 
            "method": "POST", 
            "url": "/v1/chat/completions", 
            "body": {
                "model": "gpt-3.5-turbo-0125", 
                "messages": self.messages,
                "max_tokens": 1000
                }
            }
        return record


    def log(self):
        return self.messages[1:]
        

if __name__ == '__main__':
    pass
    
    

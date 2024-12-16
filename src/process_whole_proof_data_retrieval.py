import os
import json
import hashlib
import pdb
import threading

import random
import time
from .utils import *
from .sft_data import prove_file_data

def files_in_rec(directory):
    all_files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            relative_path = os.path.relpath(full_path, directory)
            all_files.append(relative_path)
    return all_files

data_path='/home/suozhi/PALM/data'
files = [f for f in files_in_rec(os.path.join(data_path)) if f.endswith('.json') and f!='intersection.json' and  f!='path.json']


def extract_proof_data(file_name, max_tokens, length, trim_prompt, trim_theorem, prove_file_data):
    json_name=file_name.replace('.v','.json')
    file_data=prove_file_data(json_name)
    premises, defs, goals=file_data[1],file_data[2],file_data[4]
    
    goal = goals[0]
    prompt = 'Solve This Proof State:\n\n'
    prompt += 'Hypotheses:\n{hypos}\n\nGoal:\n{goal}\n\n'\
                .format(hypos='\n'.join([f'{k}: {v}' for k, v in goal['hypos'].items()]) if goal['hypos'] else 'None', goal=goal['goal'])
    if defs != [] or premises != []:
        prompt += 'Premises:'
    if defs != []:
        
        num_tokens = max_tokens - sum(length) - trim_prompt(prompt)[0]
        tokens_one = int(num_tokens/(len(defs)+len(premises)))
        prompt += '\n{defs}' \
            .format(defs= '\n'.join([trim_prompt(d, tokens_one)[1] for d in defs]))
    if premises != []:
        num_tokens = max_tokens - sum(length) - trim_prompt(prompt)[0]
        tokens_one = int(num_tokens/len(premises))
        prompt += '\n{lemmas}' \
            .format(lemmas= '\n'.join([trim_theorem(p, tokens_one)[1] for p in premises]))
    return prompt

def execute_with_timeout(func, args=(), kwargs={}, timeout_duration=60):
    result = [None]

    def target():
        result[0] = func(*args, **kwargs)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout_duration)
    if thread.is_alive():
        return None
    else:
        return result[0]

def process_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    sft_data = []
    total_tactic_lines = 0
    total_proofs = 0
    count=0
    for proof in data['proofs']:
        #print(len(data['proofs']))
        theorem_name = proof['name']
        file_name = f"{data['coq_project']}/{data['filename']}"
        print('file_name:',file_name)
        #exit()
        max_tokens=7168
        messages = []
        length=[trim_prompt(p['content'])[0] for p in messages]
        
        if file_name.replace('.v','.json') in files:
            count+=1
            try:
                time_interval = time.time()
                
                prompt = execute_with_timeout(extract_proof_data, args=(file_name, max_tokens, length, trim_prompt, trim_theorem, prove_file_data), timeout_duration=120)
                
                print("prompt:",prompt)
                #exit()
                time_interval = time.time() - time_interval
                #print("time_interval:",time_interval)
                #print(prompt)
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
                initial_goal_id = proof['steps'][0]['goal_ids']['fg'][0]
                initial_state = proof['goals'][str(initial_goal_id)]

                # 构建初始状态字符串
                initial_state_str = "Hypotheses:\n"
                for hyp in initial_state['hypotheses']:
                    initial_state_str += f"{', '.join(hyp['idents'])}: {hyp['type']}\n"
                initial_state_str += f"\nGoal:\n{initial_state['type']}"
                prompt = f"Solve This Proof State:\n\n{initial_state_str}\n\n"
                
            
            
            #return prompt
        #pdb.set_trace()
        #exit()
            
        else:
            # 获取初始状态
            initial_goal_id = proof['steps'][0]['goal_ids']['fg'][0]
            initial_state = proof['goals'][str(initial_goal_id)]

            # 构建初始状态字符串
            initial_state_str = "Hypotheses:\n"
            for hyp in initial_state['hypotheses']:
                initial_state_str += f"{', '.join(hyp['idents'])}: {hyp['type']}\n"
            initial_state_str += f"\nGoal:\n{initial_state['type']}"
            prompt = f"Solve This Proof State:\n\n{initial_state_str}\n\n"
            
            # 构建完整的证明
        full_proof = "```coq\n"
        tactic_lines = 0
        for step in proof['steps']:
            full_proof += step['command'][0] + "\n"
            tactic_lines += 1
        full_proof += "```"

        # 更新统计
        total_tactic_lines += tactic_lines
        total_proofs += 1

           
        print('total_proofs',total_proofs)
        if prompt is not None:
            sft_data.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": full_proof}
                ]
            })
    #print('count ',count)
    #exit()
    return sft_data, total_tactic_lines, total_proofs, count

def process_all_files(root_dir):
    all_sft_data = []
    total_tactic_lines = 0
    total_proofs = 0
    total_count =0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.json'):
                file_path = os.path.join(dirpath, filename)
                sft_data, file_tactic_lines, file_proofs, count = process_json_file(file_path)
                all_sft_data.extend(sft_data)
                total_tactic_lines += file_tactic_lines
                total_proofs += file_proofs
                total_count+=count
    print('total count:',total_count)
    #exit()
    return all_sft_data, total_tactic_lines, total_proofs

def deduplicate_data(data):
    unique_data = {}
    for item in data:
        # 创建一个唯一的键，基于消息内容
        key = hashlib.md5(json.dumps(item['messages'], sort_keys=True).encode()).hexdigest()
        unique_data[key] = item
    return list(unique_data.values())

def random_sample(data, sample_ratio=0.1):
    sample_size = int(len(data) * sample_ratio)
    return random.sample(data, sample_size)

if __name__ == '__main__':
    root_directory = '/home/suozhi/data/coq/data/coqgym/data'
    output_folder = '/home/suozhi/data/coq/data/whole_proof_sft_data'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    sft_dataset, total_tactic_lines, total_proofs = process_all_files(root_directory)

    # 去重
    deduplicated_dataset = deduplicate_data(sft_dataset)

    # 保存完整的去重数据集
    full_dataset_path = os.path.join(output_folder, 'coq_whole_proof_sft_dataset_full_new.json')
    with open(full_dataset_path, 'w') as f:
        json.dump(deduplicated_dataset, f, indent=2)

    print(f"处理完成。原始样本数: {len(sft_dataset)}，去重后样本数: {len(deduplicated_dataset)}。")
    print(f"完整数据集已保存在 '{full_dataset_path}' 文件中。")

    # 计算并输出平均tactic行数
    avg_tactic_lines = total_tactic_lines / total_proofs if total_proofs > 0 else 0
    print(f"平均每个证明的tactic行数: {avg_tactic_lines:.2f}")

    # 可选：生成一个采样数据集
    sampled_dataset = random_sample(deduplicated_dataset)
    sampled_file_path = os.path.join(output_folder, 'coq_whole_proof_sft_dataset_new_sampled.json')

    with open(sampled_file_path, 'w') as f:
        json.dump(sampled_dataset, f, indent=2)

    print(f"采样后样本数: {len(sampled_dataset)}。")
    print(f"采样数据已保存在 '{sampled_file_path}' 文件中。")
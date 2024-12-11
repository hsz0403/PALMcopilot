import os
import json
import hashlib
import random

def process_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    sft_data = []
    total_tactic_lines = 0
    total_proofs = 0

    for proof in data['proofs']:
        theorem_name = proof['name']
        file_name = f"{data['coq_project']}/{data['filename']}"

        # 获取初始状态
        initial_goal_id = proof['steps'][0]['goal_ids']['fg'][0]
        initial_state = proof['goals'][str(initial_goal_id)]

        # 构建初始状态字符串
        initial_state_str = "Hypotheses:\n"
        for hyp in initial_state['hypotheses']:
            initial_state_str += f"{', '.join(hyp['idents'])}: {hyp['type']}\n"
        initial_state_str += f"\nGoal:\n{initial_state['type']}"

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

        # 构建prompt
        prompt = f"Solve This Proof State:\n\n{initial_state_str}\n\n"

        sft_data.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": full_proof}
            ]
        })

    return sft_data, total_tactic_lines, total_proofs

def process_all_files(root_dir):
    all_sft_data = []
    total_tactic_lines = 0
    total_proofs = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.json'):
                file_path = os.path.join(dirpath, filename)
                sft_data, file_tactic_lines, file_proofs = process_json_file(file_path)
                all_sft_data.extend(sft_data)
                total_tactic_lines += file_tactic_lines
                total_proofs += file_proofs

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
    root_directory = '/cpfs01/shared/public/llm_math/wangjiayu/szh/coq/data/coqgym/data/'
    output_folder = '/cpfs01/shared/public/llm_math/wangjiayu/szh/coq/data/whole_proof_sft_data'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    sft_dataset, total_tactic_lines, total_proofs = process_all_files(root_directory)

    # 去重
    deduplicated_dataset = deduplicate_data(sft_dataset)

    # 保存完整的去重数据集
    full_dataset_path = os.path.join(output_folder, 'coq_whole_proof_sft_dataset_full.json')
    with open(full_dataset_path, 'w') as f:
        json.dump(deduplicated_dataset, f, indent=2)

    print(f"处理完成。原始样本数: {len(sft_dataset)}，去重后样本数: {len(deduplicated_dataset)}。")
    print(f"完整数据集已保存在 '{full_dataset_path}' 文件中。")

    # 计算并输出平均tactic行数
    avg_tactic_lines = total_tactic_lines / total_proofs if total_proofs > 0 else 0
    print(f"平均每个证明的tactic行数: {avg_tactic_lines:.2f}")

    # 可选：生成一个采样数据集
    sampled_dataset = random_sample(deduplicated_dataset)
    sampled_file_path = os.path.join(output_folder, 'coq_whole_proof_sft_dataset_sampled.json')

    with open(sampled_file_path, 'w') as f:
        json.dump(sampled_dataset, f, indent=2)

    print(f"采样后样本数: {len(sampled_dataset)}。")
    print(f"采样数据已保存在 '{sampled_file_path}' 文件中。")
1. palm.yml sexpdata==1.0.1 会报错得改成 1.0.2
2. coqllm8.11 改成 palm8.11 在(util 118行)里面改version

MKL_SERVICE_FORCE_INTEL=1 python -m src.main --proj="coqrel" --exp_name="test" --threads=1  -backtrack -intersect

额外文件 -intersect 可以不要

如果Hammer error: Dependency prediction failed.
Prediction command: predict /tmp/predict74bd43fea /tmp/predict74bd43dep /tmp/predict74bd43seq -n 64 -p knn 2>/dev/null < /tmp/predict74bd43conj > /tmp/coqhammer_outknn647679e3

重装 PALM repo:
newpalm 环境
./prepare.sh

# Build projects
./build.sh


llama factory:

conda activate palm 
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train /home/suozhi/LLaMA-Factory/examples/train_lora/prover_no_retrieval_5epoch.yaml

llamafactory-cli export examples/merge_lora/prover_no_retrieval.yaml

python -m src.success
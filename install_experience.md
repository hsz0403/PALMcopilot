1. palm.yml sexpdata==1.0.1 会报错得改成 1.0.2
2. coqllm8.11 改成 palm8.11 在(util 118行)里面改version

MKL_SERVICE_FORCE_INTEL=1 python -m src.main --proj="coqrel" --exp_name="test" --threads=1  -backtrack -intersect

额外文件 -intersect 可以不要
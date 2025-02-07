VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 CUDA_VISIBLE_DEVICES=1 vllm serve \
  /home/suozhi/raid_suozhi/saves/whole_proof_no_retr/full/sft \
  --trust-remote-code \
  --served-model-name coqcopilot7b \
  --max-model-len 16384 \

  
#conda activate palm
PATH=$PATH:/home/suozhi/old_PALM
PATH=$PATH:/home/suozhi/opam
#MKL_SERVICE_FORCE_INTEL=1 python -m src.main --proj="verdi" --exp_name="llm" --threads=1 -backtrack -intersect

CUDA_VISIBLE_DEVICES=0,1 MKL_SERVICE_FORCE_INTEL=1 python -m src.main --proj="zorns-lemma"  --exp_name="0204_full_no_retr" --threads=1 -backtrack
#threads=2 : None daemonic processes are not allowed to have children
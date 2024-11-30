conda activate palm
PATH=$PATH:/home/suozhi/old_PALM
PATH=$PATH:/home/suozhi/opam
MKL_SERVICE_FORCE_INTEL=1 python -m src.main --proj="verdi" --exp_name="llm" --threads=1 -backtrack -intersect
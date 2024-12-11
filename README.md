## Short Intro

PALM is a code base integrated with python and coq serapi to prove coqgym files.

How to run PALM code base?

see [PALMreadme](PALMreadme.md)


Additional files added: [process whole proof data with retrieval](src/process_whole_proof_data_retrieval.py)

To extract SFT format json files with retrieval premises, run

```
cd PALM
python -m src.process_whole_proof_data_retrieval
```

Data Format see Notion

To extract no retrieval SFT data, run 

```
python process_whole_proof_data_no_retrieval.py
```



Compare two models:

Change [config](src/config.py) file MODEL
```
python -m src.main --proj="hoare-tut" --exp_name="test" --threads=1 -backtrack

python -m src.main --proj="hoare-tut" --exp_name="llama8B_no_retrieval" --threads=1 -backtrack
```

## Current Problems

1. Extract SFT format json files with retrieval premises takes a few days

2. Test no retrieval first
(our trained model and llama 8B instruct)
But some projects have errors like 'angles', 'coq-procrastination'.
See error details in Notion
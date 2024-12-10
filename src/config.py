import os

OPENAI_API_KEY = ''
REPLICATE_API_TOKEN = ''

# Options: meta/meta-llama-3-8b-instruct ; meta/meta-llama-70b-instruct
MODEL = 'gpt-3.5-turbo-0125'
MODEL = '/home/suozhi/models/deepseek_prover_whole_1800iter/iter_1800_hf'

#llama 3 8B instruct

MODEL = '/home/suozhi/models/5f0b02c75b57c5855da9ae460ce51323ea669d8a'
# Path to opam installation, usually /home/username/.opam
# Replace username with your user name.
opam_path = '/home/suozhi/.opam' 

# Path to parent directory of your coq projects (used for evaluation),
# we will use os.path.join(projects_path, proj_name) to locate the project.
# For example, the project you want to use for evaluation is /home/username/coq_projects/proj_name,
# then this should be '/home/username/data/coq_projects'.
projects_path = '/home/suozhi/PALM/coq_projects'
data_path = './data'
#data_path ='/home/suozhi/data/coq/data/coqgym/data'
eval_path = './evaluation'
proof_path = os.path.join(eval_path, 'proof')

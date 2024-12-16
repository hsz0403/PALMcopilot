import os
import json
from pathlib import Path

from src.config import *
from src.utils import *


def run_proofs(proj, exp_name):
    try:
        print(f"Running proofs for project {proj} and experiment {exp_name}...")
        os.system(f"MKL_SERVICE_FORCE_INTEL=1 python -m src.main --proj={proj} --exp_name={exp_name} --threads=1 -backtrack")
    except Exception as e:
        print(f"Error running proofs for project {proj} and experiment {exp_name}: {e}")

if __name__ == '__main__':
    proj_all = 'weak-up-to buchberger jordan-curve-theorem dblib disel zchinese zfc dep-map chinese UnifySL hoare-tut huffman PolTac angles coq-procrastination coq-library-undecidability tree-automata coquelicot fermat4 demos coqoban goedel verdi-raft verdi zorns-lemma coqrel fundamental-arithmetics'
    proj_v11 = "weak-up-to buchberger dblib disel zchinese zfc dep-map chinese hoare-tut huffman PolTac \
      angles coq-procrastination fermat4 demos coqoban goedel verdi-raft zorns-lemma fundamental-arithmetics".split()
    projs = proj_all.split()
    projs = [p for p in projs if p != 'verdi']
    print(projs)
    
    for proj in projs:
        run_proofs(proj, 'llama8B_no_retrieval')
    
    
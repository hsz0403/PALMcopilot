def chat_template_to_prompt(prompt_list):
    result = ""
    total_step = len(prompt_list)
    for i, message in enumerate(prompt_list):
        result += ('<|im_start|>' + message['role'] +
                   '\n' + message['content'])
        if i+1 != total_step:
            result += '<|im_end|>\n'
        elif message['role'] == 'user':
            result += '<|im_end|>\n<|im_start|>assistant\n'
    return result

def query_ours(content,model_name):
    import vllm
    from transformers import AutoTokenizer, AutoModelForCausalLM
    def load_model(name):
        model = vllm.LLM(
            model=name,
            tensor_parallel_size=1,
            dtype='bfloat16',
            #max_num_batched_tokens=32768,
            trust_remote_code=True,
            enforce_eager=True
        )
        tokenizer = AutoTokenizer.from_pretrained(name,trust_remote_code=True)
        return model, tokenizer
    temperature=0.75
    max_tokens= 1000
    stop=['<|im_end|>',]
    
    model, tokenizer = load_model(model_name)
    #print(model)
    #exit()
    params = vllm.SamplingParams(
        n=5,
        temperature=temperature,
        #use_beam_search=temperature==0.0,
        max_tokens=max_tokens,
        stop=stop,
        repetition_penalty=1.0
    )
    outputs = model.generate(chat_template_to_prompt([{'role': 'user', 'content': content}]), params, use_tqdm=False)
    #print('outputs',outputs)
    if len(outputs) == 0:
        return [], []
    for output in outputs[0].outputs:
        text = output.text.replace(tokenizer.eos_token, '')
    
    print(content,text)
    #exit()
    return text

if __name__ == '__main__':
    #query_ours('Solve This Proof State:\n\nHypotheses:\np: nat\nHp: not (@eq nat p O)\n\nGoal:\nZp_invertible Zp_one\n\n','/home/suozhi/models/deepseek_prover_whole_1800iter/iter_1800_hf')
    #query_ours('Solve This Proof State:\n\nHypotheses:\np: nat\nHp: not (@eq nat p O)\n\nGoal:\nZp_invertible Zp_one\n\n','/home/suozhi/LLaMA-Factory/models/prover_no_retrieval_lora')
    #query_ours("Solve This Proof State:\n\nHypotheses:\nNone\n\nGoal:\nforall (T : Type) (F : (T -> Prop) -> Prop), Filter F -> forall P : T -> Prop, (forall x : T, P x) -> F P\n\nPremises:\nRecord Filter (T : Type) (F : (T -> Prop) -> Prop) : Prop := Build_Filter { filter_true : F (fun _ : T => True); filter_and : forall P Q : T -> Prop, F P -> F Q -> F (fun x : T => P x /\\ Q x); filter_imp : forall P Q : T -> Prop, (forall x : T, P x -> Q x) -> F P -> F Q } For Filter: Argument T is implicit and maximally inserted For Filter: Argument scopes are [type_scope function_scope] For Build_Filter: Argument scopes are [type_scope function_scope _ function_scope function_scope]\nClassicalFacts.F : forall A : Prop, ClassicalFacts.has_fixpoint A -> (A -> A) -> A\nBuild_ProperFilter : forall (T : Type) (F : (T -> Prop) -> Prop), (forall P : T -> Prop, F P -> exists x : T, P x) -> Filter F -> ProperFilter F\nBuild_Filter : forall (T : Type) (F : (T -> Prop) -> Prop), F (fun _ : T => True) -> (forall P Q : T -> Prop, F P -> F Q -> F (fun x : T => P x /\\ Q x)) -> (forall P Q : T -> Prop, (forall x : T, P x -> Q x) -> F P -> F Q) -> Filter F\nBuild_ProperFilter' : forall (T : Type) (F : (T -> Prop) -> Prop), ~ F (fun _ : T => False) -> Filter F -> ProperFilter' F\nRecord ProperFilter (T : Type) (F : (T -> Prop) -> Prop) : Prop := Build_ProperFilter { filter_ex : forall P : T -> Prop, F P -> exists x : T, P x; filter_filter : Filter F } For ProperFilter: Argument T is implicit and maximally inserted For ProperFilter: Argument scopes are [type_scope function_scope] For Build_ProperFilter: Argument scopes are [type_scope function_scope function_scope _]\nfilter_true = fun (T : Type) (F : (T -> Prop) -> Prop) (Filter0 : Filter F) => let (filter_true, _, _) := Filter0 in filter_true : forall (T : Type) (F : (T -> Prop) -> Prop), Filter F -> F (fun _ : T => True) Arguments T, F, Filter are implicit and maximally inserted Argument scopes are [type_scope function_scope _]\nfilter_filter' = fun (T : Type) (F : (T -> Prop) -> Prop) (ProperFilter'0 : ProperFilter' F) => let (_, filter_filter') := ProperFilter'0 in filter_filter' : forall (T : Type) (F : (T -> Prop) -> Prop), ProperFilter' F -> Filter F Arguments T, F, ProperFilter' are implicit and maximally inserted Argument scopes are [type_scope function_scope _]\nfilter_filter = fun (T : Type) (F : (T -> Prop) -> Prop) (ProperFilter0 : ProperFilter F) => let (_, filter_filter) := ProperFilter0 in filter_filter : forall (T : Type) (F : (T -> Prop) -> Prop), ProperFilter F -> Filter F Arguments T, F, ProperFilter are implicit and maximally inserted Argument scopes are [type_scope function_scope _]\nfilter_not_empty = fun (T : Type) (F : (T -> Prop) -> Prop) (ProperFilter'0 : ProperFilter' F) => let (filter_not_empty, _) := ProperFilter'0 in filter_not_empty : forall (T : Type) (F : (T -> Prop) -> Prop), ProperFilter' F -> ~ F (fun _ : T => False) Arguments T, F, ProperFilter' are implicit and maximally inserted Argument scopes are [type_scope function_scope _]\nfilter_imp : forall P Q : ?T -> Prop, (forall x : ?T, P x -> Q x) -> ?F P -> ?F Q where ?T : [ |- Type] ?F : [ |- (?T -> Prop) -> Prop] ?Filter : [ |- Filter ?F]\npositive_ind : forall P : positive -> Prop, (forall p : positive, P p -> P (p~1)%positive) -> (forall p : positive, P p -> P (p~0)%positive) -> P 1%positive -> forall p : positive, P p\nN.peano_ind : forall P : N -> Prop, P 0%N -> (forall n : N, P n -> P (N.succ n)) -> forall n : N, P n\nnat_ind : forall P : nat -> Prop, P 0%nat -> (forall n : nat, P n -> P (S n)) -> forall n : nat, P n\nfilter_and : forall P Q : ?T -> Prop, ?F P -> ?F Q -> ?F (fun x : ?T => P x /\\ Q x) where ?T : [ |- Type] ?F : [ |- (?T -> Prop) -> Prop] ?Filter : [ |- Filter ?F]\nRecord ProperFilter' (T : Type) (F : (T -> Prop) -> Prop) : Prop := Build_ProperFilter' { filter_not_empty : ~ F (fun _ : T => False); filter_filter' : Filter F } For ProperFilter': Argument T is implicit and maximally inserted For ProperFilter': Argument scopes are [type_scope function_scope] For Build_ProperFilter': Argument scopes are [type_scope function_scope _ _]\nN.peano_rect : forall P : N -> Type, P 0%N -> (forall n : N, P n -> P (N.succ n)) -> forall n : N, P n\nfilter_ex : forall P : ?T -> Prop, ?F P -> exists x : ?T, P x where ?T : [ |- Type] ?F : [ |- (?T -> Prop) -> Prop] ?ProperFilter : [ |- ProperFilter ?F]\nexist : forall (A : Type) (P : A -> Prop) (x : A), P x -> {x : A | P x}\nOmegaLemmas.fast_Zred_factor6 : forall (x : Z) (P : Z -> Prop), P (x + 0)%Z -> P x\nsigT_of_sig : forall (A : Type) (P : A -> Prop), {x : A | P x} -> {x : A & P x}\nexistT : forall (A : Type) (P : A -> Type) (x : A), P x -> {x : A & P x}\nOmegaLemmas.fast_Zopp_eq_mult_neg_1 : forall (x : Z) (P : Z -> Prop), P (x * -1)%Z -> P (- x)%Z\nproj1_sig : forall (A : Type) (P : A -> Prop), {x : A | P x} -> A\nproj2_sig : forall (A : Type) (P : A -> Prop) (e : {x : A | P x}), P (proj1_sig e)\nOmegaLemmas.fast_Zred_factor5 : forall (x y : Z) (P : Z -> Prop), P y -> P (x * 0 + y)%Z\nN.peano_rect_succ : forall (P : N -> Type) (a : P 0%N) (f : forall n : N, P n -> P (N.succ n)) (n : N), N.peano_rect P a f (N.succ n) = f n (N.peano_rect P a f n)\nInductive sig (A : Type) (P : A -> Prop) : Type := exist : forall x : A, P x -> {x : A | P x} For sig: Argument A is implicit For exist: Argument A is implicit For sig: Argument scopes are [type_scope type_scope] For exist: Argument scopes are [type_scope function_scope _ _]\nRadd_comm : forall (R : Type) (rO rI : R) (radd rmul rsub : R -> R -> R) (ropp : R -> R) (req : R -> R -> Prop), ring_theory rO rI radd rmul rsub ropp req -> forall x y : R, req (radd x y) (radd y x)\nInductive sigT (A : Type) (P : A -> Type) : Type := existT : forall x : A, P x -> {x : A & P x} For sigT: Argument A is implicit For existT: Argument A is implicit For sigT: Argument scopes are [type_scope type_scope] For existT: Argument scopes are [type_scope function_scope _ _]\nN.case_analysis : forall A : N -> Prop, Morphisms.Proper (Morphisms.respectful eq iff) A -> A 0%N -> (forall n : N, A (N.succ n)) -> forall n : N, A n\nN.induction : forall A : N -> Prop, Morphisms.Proper (Morphisms.respectful eq iff) A -> A 0%N -> (forall n : N, A n -> A (N.succ n)) -> forall n : N, A n",'/home/suozhi/LLaMA-Factory/models/prover_retrieval_lora')
    query_ours('Solve This Proof State:\n\nHypotheses:\np: nat\nHp: not (@eq nat p O)\n\nGoal:\nZp_invertible Zp_one\n\n','/home/suozhi/models/5f0b02c75b57c5855da9ae460ce51323ea669d8a')
    query_ours('Solve This Proof State:\n\nHypotheses:\np: nat\nHp: not (@eq nat p O)\n\nGoal:\nZp_invertible Zp_one\n\n','/home/suozhi/raid_suozhi/saves/whole_proof_no_retr/full/sft')
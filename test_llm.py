def query_ours(prompt,model_name):
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
    outputs = model.generate([prompt], params, use_tqdm=False)
    #print('outputs',outputs)
    if len(outputs) == 0:
        return [], []
    for output in outputs[0].outputs:
        text = output.text.replace(tokenizer.eos_token, '')
    
    print(prompt,text)
    #exit()
    return text

if __name__ == '__main__':
    query_ours('?','/home/suozhi/models/deepseek_prover_whole_1800iter/iter_1800_hf')

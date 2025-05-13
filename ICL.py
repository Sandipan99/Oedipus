from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
#from Cola_generate_promt import generate_prompt
#from MRPC_generate_prompt import generate_prompt, generate_prompt_binary
from tqdm import tqdm
import os
#from SST_generate_prompt import generate_prompt_template
import pickle
from huggingface_hub import login

login(token='<your token>')

quant_config = BitsAndBytesConfig(load_in_4bit=True)

#m = "tiiuae/falcon-7b-instruct"
#m = "meta-llama/Llama-2-13b-chat"
#m = "EleutherAI/gpt-j-6B"
#m = "meta-llama/Llama-3.1-8B-Instruct"
m = "meta-llama/Llama-3.1-70B-Instruct"
#m = "meta-llama/Llama-3.2-3B-Instruct"
#m = "mistralai/Mistral-7B-Instruct-v0.1"
#m = "microsoft/Phi-3-mini-4k-instruct"
#m = "google/gemma-2-2b-it"

model_name = 'llama3_1_70B'
#model_name = 'Gemma'
#model_name = 'LLama'
#model_name = 'Phi'


tokenizer = AutoTokenizer.from_pretrained(m)
#model = AutoModelForCausalLM.from_pretrained(m, device_map='auto')
model = AutoModelForCausalLM.from_pretrained(m, quantization_config=quant_config, device_map='auto')
tokenizer.pad_token_id = tokenizer.eos_token_id


def run():
    #task = path.split('/')[0]
    fnames = ['recentqa_ignore_correct_prompt', 'recentqa_ignore_masked_prompt', 'recentqa_ignore_noisy_prompt', 'recentqa_ignore_absurd_prompt']
    for fname in fnames:
        out_name = f"{model_name}_{fname}.pkl"

        all_results = []
        with open(f"{fname}_llama_3.pkl", 'rb') as fs:
            prompts = pickle.load(fs)

        for prompt in prompts:
            message = prompt[0]
            g_t = prompt[1]
            if fname=='recentqa_ignore_noisy_prompt':
                g_t_1 = prompt[2]
            input_ids = tokenizer.apply_chat_template(message, return_tensors="pt").to(model.device)

            output = model.generate(input_ids, max_new_tokens=20)
            response = output[0][input_ids.shape[-1]:]
            gen_text = tokenizer.decode(response)

            if fname=='recentqa_ignore_noisy_prompt':
                all_results.append((g_t, gen_text, g_t_1))

            else:
                all_results.append((g_t, gen_text))

        with open(f"{out_name}", 'wb') as fs:
            pickle.dump(all_results, fs)


if __name__ == '__main__':
    #paths = ['CoLA/new_prompts', 'MRPC/new_prompts', 'SST/new_prompts']
    #paths = ['Covid/all_prompts']
    #for path in paths:
    #    run(path=path)
    run()

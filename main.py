from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import pickle
from huggingface_hub import login

login(token='<your token>')

quant_config = BitsAndBytesConfig(load_in_4bit=True)

m = "meta-llama/Llama-3.1-70B-Instruct"

model_name = 'llama3_1_70B'

instruction_no_context = "Answer the question in one word or as precise as possible\n"

instruction_with_context = "Answer only using information in the given context. Answer \n    " \
                     "the question in one word or as precise as possible without any explanation. " \
                     "If the answer is not in the context, respond 'Not Found'\n"

instruction_ignore_context = "Answer the question by completely ignoring the context. Answer the question " \
                             "in one word or as precise as possible without any explanation."


tokenizer = AutoTokenizer.from_pretrained(m)
model = AutoModelForCausalLM.from_pretrained(m, quantization_config=quant_config, device_map='auto')
tokenizer.pad_token_id = tokenizer.eos_token_id


def create_prompt(question, context=None, ignore=False):
    if ignore:
        messages = [
            {"role": "system", "content": instruction_ignore_context},
            {"role": "user", "content": f"Question: {question}"}
        ]
    else:
        if context is None:
            messages = [
                {"role": "system", "content": instruction_no_context},
                {"role": "user", "content": f"Question: {question}"}
            ]
        else:
            messages = [
                {"role": "system", "content": instruction_with_context},
                {"role": "user", "content": f"Question: {question}\n Context: {context}"}
            ]

    return messages


def main(question, context=None, ignore=False):

    prompt = create_prompt(question, context, ignore)
    input_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(model.device)

    output = model.generate(input_ids, max_new_tokens=20)
    response = output[0][input_ids.shape[-1]:] # ignore the prompt text and only generate the output
    gen_text = tokenizer.decode(response)

    print(gen_text)


if __name__ == '__main__':
    question = "Who is the current president of Argentina?"
    correct_context = "General elections were held in Argentina on 22 " \
              "October 2023 to elect the president, vice president, " \
              "members of the National Congress, and the governors of most provinces. " \
              "As no presidential candidate won a majority in the first round, " \
              "a runoff was held on 19 November, in which Javier Milei defeated Sergio Massa."

    masked_context = "General elections were held in Argentina on 22 October 2023 to elect " \
                     "the president, vice president, members of the National Congress, and " \
                     "the governors of most provinces. As no presidential candidate won a majority " \
                     "in the first round, a runoff was held on 19 November."

    noisy_context = "General elections were held in Argentina on 22 October 2023 to elect the president, " \
                    "vice president, members of the National Congress, and the governors of most provinces. " \
                    "As no presidential candidate won a majority in the first round, a runoff was held on 19 " \
                    "November, in which Alberto Fernández defeated Mauricio Macri"

    absurd_context = "Real Madrid are the most successful club with 35 titles. Barcelona has won the Spanish " \
                     "version of the double the most times, having won the league and cup in the same year eight " \
                     "times in history, three more than Athletic Bilbao's five. Barcelona is one of two UEFA " \
                     "clubs (along with Bayern Munich who joined them in 2020) to have won the treble twice, " \
                     "after accomplishing this feat for a second time in 2015. The current champions are Real Madrid."


    main(question=question) # without context
    main(question=question, context=correct_context) # replace with masked_context, noisy_context, absurd_context #
    main(question=question, ignore=True) # ignore context

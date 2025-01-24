import os
import torch
import json
import transformers
import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, pipeline
import argparse
import re
import ast
import math

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--predict_file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--llama3_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=False)
    parser.add_argument("--conv-mode", type=str, required=False, default='video-chatgpt_v1')
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)

    return parser.parse_args()


def prepare_prompt(tokenizer:AutoTokenizer, messages:list):
    # tokens = []
    complete_message = []
    complete_message.append("<|begin_of_text|>")
    for messgae in messages:
        # tokens.append(tokenizer("<|start_header_id|>"))
        # tokens.append()
        complete_message.append("<|start_header_id|>")
        complete_message.append(messgae["role"])
        complete_message.append("<|end_header_id|>")
        complete_message.append("\n\n")
        complete_message.append(messgae["content"])
        complete_message.append("<|eot_id|>")
    
    # complete_message.append("<|eot_id|>")
    # complete_message.append("<|start_header_id|>")
    # complete_message.append("assistant")
    # complete_message.append("<|end_header_id|>")
    # complete_message.append("\n\n")
    
    
    complete_message = " ".join(complete_message)
    message_ids = tokenizer(complete_message)
    # print(message_ids)
    # message_token
    # print(complete_message)
    # assert 1==2
    # message_ids = torch.tensor(message_ids, dtype=torch.long)
    return complete_message, message_ids

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def main(args):
    llama_path = args.llama3_path
    predict_file = args.predict_file
    output_dir = args.output_dir
    output_name = args.output_name
    kwargs = {"device_map": "auto"}
   
    kwargs['torch_dtype'] = torch.float16
    
    
    print("loading llama3 model for eval ...")
    llama_config = LlamaConfig.from_pretrained(llama_path)
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_path, use_fast=True)
    llama_model = LlamaForCausalLM.from_pretrained(llama_path, low_cpu_mem_usage=True, config=llama_config, **kwargs)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    print(llama_model.device)
    print("llama3 load finish  !!")
    
    with open(predict_file) as f:
        new_pred_contents = json.load(f)
    # file = open(args.predict_file)
    # new_pred_contents = [eval(i.strip()) for i in file.readlines()]
    new_pred_contents = get_chunk(new_pred_contents, args.num_chunks, args.chunk_idx)
    answer_file = open(f"{output_dir}/{output_name}.json", "w")
    
    combined_contents = []
    # count = 0
    for pred in tqdm.tqdm(new_pred_contents, desc="Eval Video with LLaMA-3"):
        # count = count + 1
        # print(pred)
        question   = pred["question"]
        answer     = pred["label"]
        prediction = pred["predict"]
        
        messages=[
                    {
                        "role": "system",
                        "content":
                            "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                            "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                            "------"
                            "##INSTRUCTIONS: "
                            "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                            "- Consider synonyms or paraphrases as valid matches.\n"
                            "- Evaluate the correctness of the prediction compared to the answer."
                    },
                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following video-based question-answer pair:\n\n"
                            f"Question: {question}\n"
                            f"Correct Answer: {answer}\n"
                            f"Predicted Answer: {prediction}\n\n"
                            "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                            "Please generate the response in the form of a Python dictionary string with keys 'llama_pred' and 'score', where value of 'llama_pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                            "For example, your response should look like this: {'llama_pred': 'yes', 'score': 4.8}."
                    }
                ]
        complete_message, message_ids = prepare_prompt(llama_tokenizer, messages)
        ids = torch.tensor(message_ids['input_ids'], dtype=torch.long).to(llama_model.device)
        attention_mask = torch.tensor(message_ids['attention_mask']).to(llama_model.device).unsqueeze(0)
        
        embeddings = llama_model.model.embed_tokens(ids).unsqueeze(0).to(dtype=torch.float16)
        
        # print(embeddings.shape)
        # print(attention_mask.shape)
        with torch.inference_mode():
            output_ids = llama_model.generate(
                inputs_embeds = embeddings,
                attention_mask = attention_mask,
                pad_token_id=llama_tokenizer.eos_token_id,
                # do_sample=True if args.temperature > 0 else False,
                # temperature=args.temperature,
                # top_p=args.top_p,
                # max_new_tokens=128,
                # use_cache=True
            )
        # print(answer)
        out_text = llama_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        # print("prediction:{}".format(out_text))
        matches = re.findall(r'\{.*?\}', out_text)
        for match in matches:
            result_dict = ast.literal_eval(match)
        # result_qa_pair = [result_dict, pred]
        pred.update(result_dict)
        combined_contents.append(pred)
        # Save the question-answer pairs to a json file.
        answer_file.write(json.dumps(pred) + "\n")
        # if count > 10:
        #     break
    # with open(f"{output_dir}/{output_name}.json", "w") as f:
    #     json.dump(combined_contents, f, indent=4)
        # assert 1==2
    print("Prediction complete")    
    
    # # Calculate average score and accuracy
    # score_sum = 0
    # count = 0
    # yes_count = 0
    # no_count = 0
    # for result in tqdm.tqdm(combined_contents):
    #     try:
    #         # Computing score
    #         count += 1
    #         score_match = result[0]['score']
    #         score = int(score_match)
    #         score_sum += score

    #         # Computing accuracy
    #         pred = result[0]['pred']
    #         if "yes" in pred.lower():
    #             yes_count += 1
    #         elif "no" in pred.lower():
    #             no_count += 1
    #     except:
    #         print(result)

    # average_score = score_sum / count
    # accuracy = yes_count / (yes_count + no_count)
    # print("Yes count:", yes_count)
    # print("No count:", no_count)
    # print("Accuracy:", accuracy)
    # print("Average score:", average_score)
        
if __name__ == "__main__":
    args = parse_args()
    main(args)
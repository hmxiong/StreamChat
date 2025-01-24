import os
import torch
import json
# import jsonl
import transformers
import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, pipeline
import argparse
import re
import ast
import math
import json_lines
import argparse
import requests
import pandas as pd

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    # parser.add_argument('--s', help='Directory to save the model results JSON.', required=True)
    parser.add_argument("--model_name", type=str, required=False)
    parser.add_argument("--conv-mode", type=str, required=False, default='video-chatgpt_v1')
    parser.add_argument("--data_set", type=str, required=False, default='msvd')
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)

    return parser.parse_args()

def eval_intent(args):
    output_dir = args.output_dir
    output_name = args.output_name
    
    num_valids = 0
    num_corrects = 0
    count = 0
    
    with open(f"{output_dir}/{output_name}.jsonl", "rb") as f:
        for result in tqdm.tqdm(json_lines.reader(f)):
    
            count = count + 1
            if result['correct_answer'] == -1:
                continue
            num_valids += 1
            if result['truth'] == result['correct_answer']:
                num_corrects += 1 
    
    stat = {
        'num_total': count,
        'num_valids': num_valids,
        'num_corrects': num_corrects,
        'acc': num_corrects / count,
    }
    
    print(stat)

def eval_next(args):
    '''
    This function was adapted from https://github.com/doc-doc/NExT-QA/blob/main/eval_mc.py
    '''
    output_dir = args.output_dir
    output_name = args.output_name
    preds = {}
    
    anno_file_path = "/13390024681/All_Data/nextqa/val.csv"
    map_name = {'CW': 'Why', 'CH': 'How', 'TN': 'Bef&Aft', 'TC': 'When', 'DC': 'Cnt', 'DL': 'Loc', 'DO': 'Other', 'C': 'Acc_C', 'T': 'Acc_T', 'D': 'Acc_D'}
    sample_list = pd.read_csv(anno_file_path)
    group = {'CW':[], 'CH':[], 'TN':[], 'TC':[], 'DC':[], 'DL':[], 'DO':[]}
    
    with open(f"{output_dir}/{output_name}.jsonl", "rb") as f:
        for result in tqdm.tqdm(json_lines.reader(f)):
            preds.update({result['id']:{'truth':result['truth'], 'pred':result['correct_answer']}})
            
    for id, row in sample_list.iterrows():
        qns_id = str(row['video']) + '_' + str(row['qid'])
        if qns_id not in preds:
            continue
        qtype = str(row['type'])
        #(combine temporal qns of previous and next as 'TN')
        if qtype == 'TP': qtype = 'TN'
        group[qtype].append(qns_id)

    group_acc = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0, 'DC': 0, 'DL': 0, 'DO': 0}
    group_cnt = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0, 'DC': 0, 'DL': 0, 'DO': 0}
    overall_acc = {'C':0, 'T':0, 'D':0}
    overall_cnt = {'C':0, 'T':0, 'D':0}
    all_acc = 0
    all_cnt = 0
    for qtype, qns_ids in group.items():
        cnt = 0
        acc = 0
        for qid in qns_ids:

            cnt += 1
            answer = preds[qid]['truth']
            pred = preds[qid]['pred']

            if answer == pred: 
                acc += 1

        group_cnt[qtype] = cnt
        group_acc[qtype] += acc
        overall_acc[qtype[0]] += acc
        overall_cnt[qtype[0]] += cnt
        all_acc += acc
        all_cnt += cnt

    for qtype, value in overall_acc.items():
        group_acc[qtype] = value
        group_cnt[qtype] = overall_cnt[qtype]

    stat = {}
    for qtype in group_acc:
        print(map_name[qtype], end='\t')
    print('')
    for qtype, acc in group_acc.items():
        if group_cnt[qtype] == 0:
            stat[qtype] = 0
            print('{:.2f}'.format(0), end ='\t')
        else:
            stat[qtype] = acc*100.0/group_cnt[qtype]
            print('{:.2f}'.format(acc*100.0/group_cnt[qtype]), end ='\t')
    stat['Acc'] = all_acc*100.0/all_cnt
    print('')
    print('Acc: {:.2f}'.format(all_acc*100.0/all_cnt))
    # stat['data'] = preds
    
    print(stat)
    
def eval_ego(args):
    output_dir = args.output_dir
    output_name = args.output_name
    
    num_valids = 0
    num_corrects = 0
    
    with open(f"{output_dir}/{output_name}.jsonl", "rb") as f:
        for result in tqdm.tqdm(json_lines.reader(f)):
            if result['pred'] == -1:
                continue
            num_valids += 1
            if result['truth'] == result['pred']:
                num_corrects += 1 
    stat = {
        'num_total': len(f),
        'num_valids': num_valids,
        'num_corrects': num_corrects,
        'acc': num_corrects / len(f),
    }
    print(stat)
    # pass

def main(args):
    output_dir = args.output_dir
    output_name = args.output_name
    
    res_dict = []
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    
    with open(f"{output_dir}/{output_name}.jsonl", "rb") as f:
        # combined_contents = json_lines.reader(f)
        for result in tqdm.tqdm(json_lines.reader(f)):
    
    # Calculate average score and accuracy
    # for result in tqdm.tqdm(combined_contents):
            try:
                # Computing score
                count += 1
                score_match = result['score']
                score = int(score_match)
                score_sum += score

                # Computing accuracy
                pred = result['llama_pred']
                if "yes" in pred.lower():
                    yes_count += 1
                elif "no" in pred.lower():
                    no_count += 1
            except:
                print(result)

    average_score = score_sum / count
    accuracy = yes_count / (yes_count + no_count)
    print("Yes count:", yes_count)
    print("No count:", no_count)
    print("Accuracy:", accuracy)
    print("Average score:", average_score)
    
    res_dict.append({"Yes count":yes_count})
    res_dict.append({"No count":no_count})
    res_dict.append({"Accuracy":accuracy})
    res_dict.append({"Average score":average_score})
    
    with open(f"{output_dir}/{output_name}_res.json","w") as f:
        json.dump(res_dict, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    if args.data_set in ['msvd', 'msrvtt']:
        main(args)
    elif args.data_set in ['intent-qa']:
        eval_intent(args)
    elif args.data_set in ['egoschema']:
        eval_ego(args)
    elif args.data_set in ['next-qa']:
        eval_next(args)
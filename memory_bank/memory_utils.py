import os, shutil, datetime, time, json
import gradio as gr
import sys
import os
# from llama_index import GPTSimpleVectorIndex
bank_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../memory_bank')
sys.path.append(bank_path)
# from build_memory_index import build_memory_index
memory_bank_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../memory_bank')
sys.path.append(memory_bank_path)
from summarize_memory import summarize_memory
from transformers import Trainer, HfArgumentParser
from dataclasses import dataclass, field

@dataclass
class DataArguments:
    memory_search_top_k: int = field(default=2)
    memory_basic_dir: str = field(default='/llama/EfficientVideo/Ours/memory_bank/memories')
    memory_file: str = field(default='update_memory_0512_eng.json')
    language: str = field(default='en')
    max_history: int = field(default=7,metadata={"help": "maximum number for keeping current history"},)
    enable_forget_mechanism: bool = field(default=False)
@dataclass
class ModelArguments:
    model_type: str = field(
        default="chatglm",
        metadata={"help": "model type: chatglm / belle"},
    )
    base_model: str = field(
        default="THUDM/chatglm-6b",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    adapter_model: str = field(
        default=None,
        metadata={"help": "Path to lora adapter model"},
    )
    ptuning_checkpoint: str = field(
        default=None,
        metadata={"help": "Path to pretrained prefix embedding of ptuning"},
    )
    

    # prompt_column
    # train_file: str = field(default="/home/t-qiga/azurewanjun/SiliconGirlfriend/data/merge_data/only_mental_0426.json")

# data_args,model_args = HfArgumentParser(
#     (DataArguments,ModelArguments)
# ).parse_args_into_dataclasses()


def summarize_memory_event_personality(data_args, memory, user_name, llm_client):
    if isinstance(data_args,gr.State):
        data_args = data_args.value
    if isinstance(memory,gr.State):
        memory = memory.value
    memory_dir = os.path.join(data_args.memory_basic_dir,data_args.memory_file)
    memory = summarize_memory(memory_dir,llm_client, user_name,language=data_args.language)
    user_memory = memory[user_name] if user_name in memory.keys() else {}
    return user_memory#, user_memory_index 

def enter_name(name, memory,local_memory_qa,data_args,update_memory_index=True):
    cur_date = datetime.date.today().strftime("%Y-%m-%d")
    user_memory_index = None
    if isinstance(data_args,gr.State):
        data_args = data_args.value
    if isinstance(memory,gr.State):
        memory = memory.value
    if isinstance(local_memory_qa,gr.State):
        local_memory_qa=local_memory_qa.value
    memory_dir = os.path.join(data_args.memory_basic_dir, data_args.memory_file)
    if name in memory.keys():
        user_memory = memory[name]
        memory_index_path = os.path.join(data_args.memory_basic_dir,f'memory_index/{name}_index')
        
        os.makedirs(os.path.dirname(memory_index_path), exist_ok=True)
        if (not os.path.exists(memory_index_path)) or update_memory_index:
            print(f'Initializing memory index {memory_index_path}...')

            if os.path.exists(memory_index_path):
                shutil.rmtree(memory_index_path)
            memory_index_path, _ = local_memory_qa.init_memory_vector_store(filepath=memory_dir,vs_path=memory_index_path,user_name=name,cur_date=cur_date)                      
        
        user_memory_index = local_memory_qa.load_memory_index(memory_index_path) if memory_index_path else None
        msg = f"欢迎回来，{name}！" if data_args.language=='cn' else f"Wellcome Back, {name}！"
        return msg,user_memory,memory, name,user_memory_index
    else:
        memory[name] = {}
        memory[name].update({"name":name}) 
        msg = f"欢迎新用户{name}！我会记住你的名字，下次见面就能叫你的名字啦！" if data_args.language == 'cn' else f'Welcome, new user {name}! I will remember your name, so next time we meet, I\'ll be able to call you by your name!'
        return msg,memory[name],memory,name,user_memory_index

def update_memory():
    pass

def save_local_memory(memory,b,user_name,data_args):
    if isinstance(data_args,gr.State):
        data_args = data_args.value
    if isinstance(memory,gr.State):
        memory = memory.value
    memory_dir = os.path.join(data_args.memory_basic_dir,data_args.memory_file)
    date = time.strftime("%Y-%m-%d", time.localtime())
    if memory[user_name].get("history") is None:
        memory[user_name].update({"history":{}})
    
    if memory[user_name]['history'].get(date) is None:
        memory[user_name]['history'][date] = []
    # date = len(memory[user_name]['history'])
    memory[user_name]['history'][date].append({'query':b[-1][0],'response':b[-1][1]})
    json.dump(memory,open(memory_dir,"w",encoding="utf-8"),ensure_ascii=False, indent=4)
    return memory
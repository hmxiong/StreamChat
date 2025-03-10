from llama_index.core import SimpleDirectoryReader, Document
# from llama_index.core import GPTTreeIndex, GPTSimpleVectorIndex
# from llama_index.indices.composability import ComposableGraph
# import json, openai
# from llama_index.core import LLMPredictor, GPTSimpleVectorIndex, PromptHelper, ServiceContext
# from langchain import OpenAI, AzureOpenAI
from llama_index.legacy.llms import (HuggingFaceLLM, CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata)
from llama_index.core.llms.callbacks import llm_completion_callback
from langchain.llms import AzureOpenAI,OpenAIChat
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from typing import Optional, List, Mapping, Any
import os
import torch
# language = 'en'
# openai.api_key = os.environ["OPENAI_API_KEY"]
# os.environ["OPENAI_API_BASE"] = openai.api_base
# define LLM
# llm_predictor = LLMPredictor(llm=OpenAIChat(model_name="gpt-3.5-turbo"))

# define prompt helper
# set maximum input size
max_input_size = 2048
# set number of output tokens
num_output = 256
# set maximum chunk overlap 
max_chunk_overlap = 20

class LLaMA3(CustomLLM):
    def __init__(self, model_path: str):
        print("Init LLaMA3 model for predictor and memorizing !!")
        kwargs = {"device_map": "auto"}
        kwargs['torch_dtype'] = torch.float16
        llama_config = LlamaConfig.from_pretrained(model_path)
        self.llama_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.llama_model = LlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=llama_config, **kwargs)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        
    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=max_input_size,
            num_output=num_output,
            model_name="llama3-8b",
        )
        
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # 使用分词器将输入文本编码为token
        inputs = self.llama_tokenizer(prompt, return_tensors='pt')

        # 将输入张量移动到模型所在的设备（例如GPU）上
        inputs = {key: val.to(self.llama_model.device) for key, val in inputs.items()}

        # 使用模型生成文本
        output = self.llama_model.generate(**inputs, max_length=kwargs.get("max_length", 256))

        # 解码生成的token为文本
        text = self.llama_tokenizer.decode(output[0], skip_special_tokens=True)

        return CompletionResponse(text=text)
    # @llm_completion_callback()
    # def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
    #     prompt_length = len(prompt)

    #     # only return newly generated tokens
    #     text,_ = self.llama_model.chat(self.llama_tokenizer, prompt, history=[])
    #     return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError()


def generate_memory_docs(data,language):
    # data = json.load(open(memory_path,'r',encoding='utf8'))
    all_user_memories = {}
    for user_name, user_memory in data.items():
        # print(user_memory)
        all_user_memories[user_name] = []
        if 'history' not in user_memory.keys():
            continue
        for date, content in user_memory['history'].items():
            memory_str = f'日期{date}的对话内容为：' if language=='cn' else f'Conversation on {date}：'
            for dialog in content:
                query = dialog['query']
                response = dialog['response']
                memory_str += f'\n{user_name}：{query.strip()}'
                memory_str += f'\nAI：{response.strip()}'
            memory_str += '\n'
            if 'summary' in user_memory.keys():
                if date in user_memory['summary'].keys():
                    summary = f'时间{date}的对话总结为：{user_memory["summary"][date]}' if language=='cn' else f'The summary of the conversation on {date} is: {user_memory["summary"][date]}'
                    memory_str += summary
            # if 'personality' in user_memory.keys():
            #     if date in user_memory['personality'].keys():
            #         memory_str += f'日期{date}的对话分析为：{user_memory["personality"][date]}'
            # print(memory_str)
            all_user_memories[user_name].append(Document(memory_str))
    return all_user_memories
            
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS # type: ignore
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
# from langchain import HuggingFacePipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains.retrieval_qa.base import RetrievalQA
# from langchain.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from rag_memory import RAGChain
import argparse
import logging
import warnings

warnings.filterwarnings("ignore")
# Default constants for the script
DEFAULT_CHAT_MODEL_ID = "/13390024681/All_Model_Zoo/llama3-8b-instruct-hf"
DEFAULT_EMBED_MODEL_ID = "/13390024681/All_Model_Zoo/mxbai-embed-large-v1"
DEFAULT_K = 4
DEFAULT_TOP_K = 2
DEFAULT_TOP_P = 0.6
DEFAULT_TEMPERATURE = 0.6
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_HF_TOKEN = None

def test_embedding():
    # 指定数据集名称和包含内容的列
    dataset_name = "/13390024681/All_Data/databricks-dolly-15k"
    page_content_column = "context"  # 或者其他您感兴趣的列

    # 创建加载器实例
    loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)

    # 加载数据
    data = loader.load()

    # 显示前15个条目
    data[:2]

    # 使用特定参数创建RecursiveCharacterTextSplitter类的实例。
    # 它将文本分成每个1000个字符的块，每个块有150个字符的重叠。
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    # 'data'包含您想要拆分的文本，使用文本拆分器将文本拆分为文档。
    docs = text_splitter.split_documents(data)
    print(docs[0])    
    
    # 定义要使用的预训练模型的路径
    modelPath = "/13390024681/All_Model_Zoo/all-MiniLM-L6-v2"

    # 创建一个包含模型配置选项的字典，指定使用CPU进行计算
    model_kwargs = {'device':'cpu'}

    # 创建一个包含编码选项的字典，具体设置 'normalize_embeddings' 为 False
    encode_kwargs = {'normalize_embeddings': False}

    # 使用指定的参数初始化HuggingFaceEmbeddings的实例
    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,     # 提供预训练模型的路径
        model_kwargs=model_kwargs, # 传递模型配置选项
        encode_kwargs=encode_kwargs # 传递编码选项
    )
    text = "This is a test document."
    query_result = embeddings.embed_query(text)
    print(query_result[:3])
    print("Building FAISS index in :{}".format(model_kwargs['device']))
    db = FAISS.from_documents(docs, embeddings) # 对文本数据进行编码
    
    question = "What is cheesemaking?"
    searchDocs = db.similarity_search(question)
    print(searchDocs[0].page_content)
    
    #<<<<<<<<<<<<<<<<<building language model>>>>>>>>>>>>>>>>>>>
    
    # 通过加载预训练的"Intel/dynamic_tinybert"标记器创建标记器对象。
    tokenizer = AutoTokenizer.from_pretrained("Intel/dynamic_tinybert")

    # 通过加载预训练的"Intel/dynamic_tinybert"模型创建问答模型对象。
    model = AutoModelForQuestionAnswering.from_pretrained("Intel/dynamic_tinybert")
    
    # 指定要使用的模型名称
    model_name = "Intel/dynamic_tinybert"

    # 加载与指定模型关联的标记器
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512)

    # 使用模型和标记器定义一个问答管道
    question_answerer = pipeline(
        "question-answering", 
        model=model_name, 
        tokenizer=tokenizer,
        return_tensors='pt'
    )

    # 创建HuggingFacePipeline的实例，该实例包装了问答管道，并带有额外的模型特定参数（温度和max_length）
    llm = HuggingFacePipeline(
        pipeline=question_answerer,
        model_kwargs={"temperature": 0.7, "max_length": 512},
    )
    
    retriever = db.as_retriever()
    
    docs = retriever.get_relevant_documents("What is Cheesemaking?")
    print(docs[0].page_content)
    
    # 从'db'创建一个带有搜索配置的检索器对象，其中检索最多4个相关的拆分/文档。
    retriever = db.as_retriever(search_kwargs={"k": 4})

    # 使用RetrievalQA类创建一个带有检索器、链类型“refine”和不返回源文档选项的问答实例（qa）。
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever, return_source_documents=False)
    
    question = "Who is Thomas Jefferson?"
    result = qa.run({"query": question})
    print(result["result"])

def test_RAG_QA():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset
    
    SYS_PROMPT = """You are an assistant for answering questions.
    You are given the extracted parts of a long document and a question. Provide a conversational answer.
    If you don't know the answer, just say "I do not know." Don't make up an answer."""
    
    model_id = "/13390024681/All_Model_Zoo/llama3-8b-instruct-hf"
    
    ST = SentenceTransformer("/13390024681/All_Model_Zoo/mxbai-embed-large-v1")

    dataset = load_dataset("/13390024681/All_Data/wikipedia_embedded",revision = "embedded")

    data = dataset["train"]
    data = data.add_faiss_index("embeddings") # column name that has the embeddings of the dataset

    def search(query: str, k: int = 3 ):
        """a function that embeds a new query and returns the most probable results"""
        embedded_query = ST.encode(query) # embed new query
        scores, retrieved_examples = data.get_nearest_examples( # retrieve results
            "embeddings", embedded_query, # compare our new embedded query with the dataset embeddings
            k=k # get only top k results
        )
        return scores, retrieved_examples


    # use quantization to lower GPU usage
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config
    )
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    def format_prompt(prompt,retrieved_documents,k):
        """using the retrieved documents we will prompt the model to generate our responses"""
        PROMPT = f"Question:{prompt}\nContext:"
        for idx in range(k) :
            PROMPT+= f"{retrieved_documents['text'][idx]}\n"
        return PROMPT

    def generate(formatted_prompt):
        formatted_prompt = formatted_prompt[:2000] # to avoid GPU OOM
        messages = [{"role":"system","content":SYS_PROMPT},{"role":"user","content":formatted_prompt}]
        # tell the model to generate
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        outputs = model.generate(
            input_ids,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        return tokenizer.decode(response, skip_special_tokens=True)

    def rag_chatbot(prompt:str,k:int=2):
        scores , retrieved_documents = search(prompt, k)
        formatted_prompt = format_prompt(prompt,retrieved_documents,k)
        return generate(formatted_prompt)

    response = rag_chatbot("what's anarchy ?", k = 2)
    
    print(response)

def test_memory():
    hf = HuggingFacePipeline.from_model_id(
        model_id="/13390024681/All_Model_Zoo/llama3-8b-instruct-hf",
        task="text-generation",
        device=0,
        pipeline_kwargs={"max_new_tokens": 100},
    )

    template = """Question: {question}

    Answer: Let's think step by step."""
    prompt = PromptTemplate.from_template(template)

    chain = prompt | hf

    question = "Who is Cristiano Ronaldo?"

    print(chain.invoke({"question": question}))

def test_RAG_Chain():
    """
    Main function to run the Bangla Retrieval-Augmented Generation (RAG) System.
    It parses command-line arguments, loads the RAG model, and processes user queries in an interactive loop.
    """
    # Argument parser for command-line options, arguments and sub-commands
    parser = argparse.ArgumentParser(
        description="Bangla Retrieval-Augmented Generation System"
    )
    parser.add_argument(
        "--chat_model",
        type=str,
        default=DEFAULT_CHAT_MODEL_ID,
        help="The Hugging Face model ID of the chat model",
    )
    parser.add_argument(
        "--embed_model",
        type=str,
        default=DEFAULT_EMBED_MODEL_ID,
        help="The Hugging Face model ID of the embedding model",
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K, help="The number of documents to retrieve"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help="The top_k parameter for the chat model",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=DEFAULT_TOP_P,
        help="The top_p parameter for the chat model",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="The temperature parameter for the chat model",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="The maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="The chunk size for text splitting",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="The chunk overlap for text splitting",
    )
    parser.add_argument(
        "--text_path",
        type=str,
        default='/13390024681/llama/EfficientVideo/Ours/rag_memory/memory.txt',
        help="The txt file path to the text file",
    )
    parser.add_argument(
        "--show_context",
        action="store_true",
        help="Whether to show the retrieved context or not.",
    )
    parser.add_argument(
        "--quantization",
        action="store_true",
        help="Whether to enable quantization(4bit) or not.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=DEFAULT_HF_TOKEN,
        help="Your Hugging Face API token",
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        # Initialize and load the RAG model
        rag_chain = RAGChain()
        rag_chain.load(
            chat_model_id=args.chat_model,
            embed_model_id=args.embed_model,
            text_path=args.text_path,
            k=args.k,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            hf_token=args.hf_token,
            max_new_tokens=args.max_new_tokens,
            quantization=args.quantization,
        )
        logging.info(
            f"RAG model loaded successfully: chat_model={args.chat_model}, embed_model={args.embed_model}"
        )

        # Interactive loop for user queries
        while True:
            query = input("your question: ")
            if query.lower() in ["exit", "quit"]:
                print("See you again, thanks!")
                break
            try:
                answer, context = rag_chain.get_response(query)
                if args.show_context:
                    print(f"Context {context}\n------------------------\n")
                print(f"the answer: {answer}")
            except Exception as e:
                logging.error(f"Couldn't generate an answer: {e}")
                print("Try again!")

    except Exception as e:
        logging.critical(f"Fatal error: {e}")
        print("Error occurred, please check logs for details.")

if __name__ == '__main__':
    # test_memory()
    # test_embedding()
    # test_RAG_QA()
    test_RAG_Chain()
    
    



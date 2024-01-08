import os
from dotenv import load_dotenv
import time

# from langchain.document_loaders import TextLoader, PyPDFLoader
from PyPDF2 import PdfReader
from io import StringIO
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceBgeEmbeddings

import chromadb
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
import together
# from langchain.schema import prompt
from pydantic import Extra, root_validator

from typing import Dict, Any

from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env

from htmlTemplates import css, bot_template, user_template
# import textwrap

import streamlit as st

import glob

file_paths = glob.glob("./docs_retrieval/test/*.*")
# file_paths = glob.glob("/content/retrieval_docs/*.*")
vectordb_path = "./chromadb/"
# vectordb_path = "/content/chromadb"

model_name = "BAAI/bge-base-en"
encode_kwargs = {"normalize_embeddings": True} ## cosine similarity
model_norm = HuggingFaceBgeEmbeddings(
    model_name = model_name,
    encode_kwargs = encode_kwargs
)

def create_embeddings(file_docs):
    # docs = []
    text = ""
    for file in file_docs:
        file_name = str(file.name)
        if file_name.endswith(".pdf"):
            pdf = PdfReader(file)
            for page in pdf.pages:
                text += page.extract_text()
            # docs.extend(text)

        elif file_name.endswith(".txt"):
            content = StringIO(file.getvalue().decode("utf-8")).read()
            text += content

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, 
                                                    chunk_overlap = 200)
    texts = text_splitter.split_text(text)

    client = chromadb.PersistentClient(path = vectordb_path)

    vector_db = Chroma.from_texts(texts = texts,
                          embedding = model_norm,
                          client = client)

    return vector_db

# vector_db = create_embeddings(file_paths)

vector_db = Chroma(persist_directory = vectordb_path, embedding_function = model_norm)

retriever = vector_db.as_retriever(search_kwargs = {"k" : 3})

## llama prompt style

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "SYS", "/SYS"
DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assisstant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

"""

def get_system_prompt(NEW_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + NEW_SYSTEM_PROMPT + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + E_INST
    return(prompt_template)

SYS_PROMPT = """
You are a helpful, respectful and honest assisstant. Always answer as helpfully as possible using the context provided. You should give the answer once and stop after it. Do not generate any human like question.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, If you don't know the answer to a question, reply "Please provide more information".
CONTEXT: \n\n {context} \n
"""
instructions = """Question: {question}"""

system_prompt_template = get_system_prompt(SYS_PROMPT)
messages = [
    SystemMessagePromptTemplate.from_template(system_prompt_template),
    HumanMessagePromptTemplate.from_template(instructions)
]
qa_prompt = ChatPromptTemplate.from_messages(messages)
combine_docs_chain_kwargs = {"prompt" : qa_prompt}


os.environ["TOGETHER_API_KEY"] = "19e7b9519392abfe4ab32388df8546aea9a808f8d96ec922e0b0d5aa8c6bc25e"

together.api_key = os.environ["TOGETHER_API_KEY"]

models = together.Models.list()

together.Models.start("togethercomputer/llama-2-7b-chat")

class TogetherLLM(LLM):
    """Together Large Language Models"""

    """Model endpoint to use"""
    model: str = "togethercomputer/llama-2-7b-chat"

    """Together API key"""
    together_api_key: str = os.environ["TOGETHER_API_KEY"]

    """sampling temperature to use"""
    temperature: float = 0.7

    """Number of maximum tokens to generate"""
    max_tokens: int = 256

    class Config:
        extra = Extra.forbid

    @root_validator(allow_reuse = True)
    def validate_environment(cls, values: Dict) -> Dict:
        "Validate that the API key is set"
        api_key = get_from_dict_or_env(
            values, "together_api_key", "TOGETHER_API_KEY"
        )
        values["together_api_key"] = api_key
        return values

    @property
    def _llm_type(self) -> str:
        return "together"

    def _call(self, prompt: str, **kwargs: Any) -> str:
        "Call to together endpoint"

        together.api_key = self.together_api_key
        output = together.Complete.create(prompt,
                                          model = self.model,
                                          max_tokens = self.max_tokens,
                                          temperature = self.temperature,
                                          )
        text = output["output"]["choices"][0]["text"]
        return text

llm = TogetherLLM(
    model = "togethercomputer/llama-2-7b-chat",
    temperature = 0.1,
    max_tokens = 300
)

def get_conversation_chain(vector_db):
    memory = ConversationBufferMemory(
        memory_key = "chat_history",
        return_messages = True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        memory = memory,
        combine_docs_chain_kwargs = combine_docs_chain_kwargs
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html = True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html = True)
            

def main():
    load_dotenv()
    st.set_page_config(page_title = "Retrieval Augmented Generation",
                       page_icon = ":books:")
    st.write(css, unsafe_allow_html = True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Retrieval Augmented Generation :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        time.sleep(1)
        handle_user_input(user_question)
        
        
    with st.sidebar:
        st.subheader("Your Documents")
        file_docs = st.file_uploader(
            "Upload your text or pdf files here and click on 'Process' to create Embeddings", accept_multiple_files = True
        )
        if st.button("Process"):
            with st.spinner("Creating vector embeddings..."):
                vector_db = create_embeddings(file_docs)

                st.session_state.conversation = get_conversation_chain(vector_db)

if __name__ == "__main__":
    main()
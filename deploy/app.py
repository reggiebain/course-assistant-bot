import streamlit as st
import pdfplumber
import markdown
import faiss
import numpy as np

from tqdm.auto import tqdm
import pandas as pd
from typing import Optional, List, Tuple
import json
import datasets
import os
import re
import torch

from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.llms import HuggingFaceHub
from ragatouille import RAGPretrainedModel
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models.llms import LLM
from langchain.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, UnstructuredHTMLLoader  # Assumes both loaders exist
from langchain.schema import Document
from langchain_core.language_models import BaseChatModel

######################## GLOBAL STUFF ########################
DEFAULT_QUESTIONS = [
    "What topics are covered in this course?",
    "What are the prerequisites for this course?",
    "What is the grading policy?",
    "What is the attendance policy?",
    "What are the lab requirements?",
    "What is the policy on late assignments?",
    "Are there any required textbooks?",
    "How can students contact the instructor?"
]
HF_TOKEN = st.secrets['api_keys']['HF_TOKEN']
CHUNK_SIZE = 200
RAG_PROMPT_TEMPLATE = """
<|system|>
Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.</s>
<|user|>
Context:
{context}
---
Now here is the question you need to answer.

Question: {question}
</s>
<|assistant|>
"""
READER_MODEL_PARAMS = {"max_new_tokens": 512, "top_k": 30,"temperature": 0.1,"repetition_penalty": 1.03,}

######################### RAG PIPELINE FUNCTIONS ######################
# Function to extract text from PDF
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages)
    return text

# Function to extract text from Markdown
def extract_text_from_markdown(file):
    content = file.read().decode("utf-8")
    text = markdown.markdown(content)
    return text

# Takes in streamlit loaded file with extension -> documents
def load_documents(file_name):
    if file_name.endswith('.pdf'):
        loader = PyPDFLoader(file_name)
        print("Loading PDF document...")
    elif file_name.endswith('.md'):
        loader = UnstructuredMarkdownLoader(file_name)
        print("Loading Markdown document...")
    elif file_name.endswith('.html'):
        loader = UnstructuredHTMLLoader(file_name)
        print("Loading HTML file...")
    else:
        raise ValueError("Unsupported file format. Please provide a PDF or Markdown file.")
    
    documents = loader.load()
    return documents

# Function to get the selected LLM
def get_llm(selected_model):
    if selected_model == "Zephyr-7b":
        repo_id = "HuggingFaceH4/zephyr-7b-beta"  # Replace with your OpenAI API key in `.env`
    elif selected_model == "Llama-3.1-8B":
        repo_id =  'meta-llama/Llama-3.1-8B-Instruct'
    else:
        st.error("Unsupported model selected.")
        return None
    reader_llm = HuggingFaceHub(
        repo_id=repo_id,
        task="text-generation", 
        huggingfacehub_api_token=HF_TOKEN,
        model_kwargs = READER_MODEL_PARAMS,
    )
    return reader_llm

# Implement splitting of docs via text_splitter
def split_documents(chunk_size: int, knowledge_base: List[Document],tokenizer_name: str,) -> List[Document]:

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique

# Method to load embeddins and create vectore store
def load_embeddings(langchain_docs, chunk_size, embedding_model_name: Optional[str] = "thenlper/gte-small",) -> FAISS:

    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        multi_process=True,
        #model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # set True to compute cosine similarity
    )

    docs_processed = split_documents(
        chunk_size,
        langchain_docs,
        embedding_model_name,
    )
    knowledge_index = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE,
        #allow_dangerous_deserialization=True
    )
    return knowledge_index

def answer_with_rag(question: str, llm: LLM, knowledge_index: VectorStore, 
                    reranker: Optional[RAGPretrainedModel] = None, num_retrieved_docs: int = 30,
                    num_docs_final: int = 7,) -> Tuple[str, List[Document]]:
    
    # Gather documents with retriever
    relevant_docs = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
    relevant_docs = [doc.page_content for doc in relevant_docs]  # keep only the text

    # Optionally rerank results using RAGatoulli ColBERT model
    if reranker:
        relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
        relevant_docs = [doc["content"] for doc in relevant_docs]

    relevant_docs = relevant_docs[:num_docs_final]

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

    # Redact an answer
    answer = llm(final_prompt)
    #answer = llm.invoke(final_prompt)

    return answer, relevant_docs

def parse_question_answer(rag_output):
    # Split the output into lines for easier parsing
    lines = rag_output.splitlines()

    # Initialize variables
    question = None
    answer = None

    # Look for the question and answer markers
    for line in lines:
        line = line.strip()
        if line.startswith("Question:"):
            # Extract the question text
            question = line.replace("Question:", "").strip()
        elif line.startswith("<|assistant|>"):
            # Answer follows <|assistant|>, so join subsequent lines
            answer_start_index = lines.index(line) + 1
            answer = " ".join(l.strip() for l in lines[answer_start_index:])
            break

    # Return the parsed question and answer
    #return {"question": question, "answer": answer}
    return answer

# Run RAG for each question 
def process_questions_with_answers(questions, llm, knowledge_index):
    results = []
    for question in questions:
        if question:  # Ensure there is a valid question
            raw_answer =  answer_with_rag(question, llm, knowledge_index)
            answer = parse_question_answer(raw_answer)
            results.append({'question': question, 'answer': answer})
    return results     

def score_syllabus(results):
    answered = sum(1 for result in results if result['answer'])
    score = (answered / len(questions)) * 100
    return score

###################### STREAMLIT APP CODE ######################
# Streamlit app
st.title("Syllabus Evaluation App")

uploaded_file = st.file_uploader("Upload your syllabus (PDF or Markdown):", type=["pdf", "md", "html"])

print(f"Uploaded file type: {type(uploaded_file)}")
# LLM model selection
selected_model = st.selectbox(
    "Choose an LLM for questioning your syllabus:",
    ["Zephyr-7b", "Llama-3.1-8B"]
)
# see and edit questions
questions = st.text_area(
    "View and/or edit questions (one per line):",
    value="\n".join(DEFAULT_QUESTIONS),
    height=150,
).split("\n")

# Button to display questions
if st.button("Save Questions"):
    st.success("Questions updated!")

if uploaded_file and st.button("Upload and Process Syllabus"):
    with st.spinner('Processing syllabus...'):
    # Process input file in langchain
        with open(uploaded_file.name, mode='wb') as w:
            w.write(uploaded_file.getvalue())
        docs_processed = load_documents(uploaded_file.name)
        # Split into chunks, load embeddings
        knowledge_index = load_embeddings(docs_processed, chunk_size=CHUNK_SIZE)
        st.success("File uploaded and processed!")

    if st.button("Evaluate Syllabus"):
        with st.spinner("Analyzing syllabus..."):
            results = process_questions_with_answers(questions, selected_model, knowledge_index)
            score = score_syllabus(results)
        
        st.subheader(f"Results: {score:.2f}% of questions answered")
        for result in results:
            st.write(f"**Q:** {result['question']}")
            st.write(f"**A:** {result['answer']}")
            st.write("---")

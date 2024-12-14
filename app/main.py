import os
import groq
import sys
import yaml
import logging

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import  ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from typing import Dict

from datetime import datetime, timedelta
from fastapi_utils.tasks import repeat_every

sys.path.append('../..')
_ = load_dotenv(find_dotenv()) # read local .env file
groq.api_key  = os.getenv('GROQ_API_KEY')


class MessageRequest(BaseModel):
    message: str
    session_id: str

class MessageResponse(BaseModel):
    response: str
    session_id: str



def load_config(config_file):
    with open(config_file, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

groq_api_key = os.environ['GROQ_API_KEY']
config_file="config/config.yaml"
config = load_config(config_file)
model_name = config['model_name']
temperature = config['temperature']
max_tokens = config['max_tokens']
vectorstore_path = config['vectorstore']['path']
embedding_model = config['vectorstore']['embedding_model']
search_type = config['retriever']['search_type']
search_kwargs = config['retriever']['search_kwargs']
memory_k = config['memory']['k']
chain_type=config['qa']['chain_type']
output_key=config['qa']['output_key']
system_prompt=config['system_prompt']
static_dir=static_dir = os.path.join(os.path.dirname(__file__), "app", "static")
session_histories: Dict[str, list] = {}
session_timestamps: Dict[str, datetime] = {}


app=FastAPI()
app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can also specify the URL of your frontend here for security)
    allow_credentials=True,
    allow_methods=["POST"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)
    
def initialize_llm():
    return ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens)
    

def make_retriever(embedding_model,vectorstore_path,search_type,search_kwargs):
        embedding_function=HuggingFaceEmbeddings(model_name=embedding_model)
        vs = Chroma(persist_directory=vectorstore_path, embedding_function=embedding_function)
        retriever = vs.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
        return retriever


def create_prompt(question,chat_history):
    return PromptTemplate(
           input_variables=["chat_history", "question", "context"],
           template=(f"System: {system_prompt}\nChat History:\n{chat_history}\n"
                "Retrieved Context: {context}\n"f"User Input: {question}\n"
                 "Answer concisely and informatively:"))

def setup_memory(memory_k,return_messages=True):
        return ConversationBufferWindowMemory(
        k=memory_k, 
        memory_key="chat_history", 
        return_messages=return_messages)


def qa_conv_chain(prompt, output_key=output_key):
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt, "document_variable_name": "context"},
        output_key=output_key,
        get_chat_history=lambda h: h
    )
    
    

def querying(question,chat_history,session_id):
    # Append the current question to chat history
    chat_history.append((question, None)) # Placeholder for response

    # Build a prompt including the full chat history
    formatted_history = "\n".join(
        [f"User: {q}\nAssistant: {a}" for q, a in chat_history if a is not None]
    )
    prompt = create_prompt(question, formatted_history)  

    # Create the conversation chain
    qa_conv_chain_=qa_conv_chain(prompt)
    response=qa_conv_chain_({"question":question,"chat_history":chat_history})
    
    # Update the latest response in chat history
    answer = response[output_key].strip()
    chat_history.append((question, answer))
    session_histories[session_id] = chat_history

    return {"response": answer, "session_id": session_id}




'''    def chat_w_llm():
        while True:
            question = input("You: ")

            if question.lower() in ['exit', 'quit','bye']:
                print("\nTH-Rosenheim Assistant: Goodbye! Have a nice day. I hope I was of help to you!")
                break

            # if user input given
            if question.strip():        
                # Generate and display the chatbot's response
                response = querying(question,chat_history)
                print("\nTH-Rosenheim Assistant:", response)
                chat_history.append((question, response)) '''
        
    
    
retriever=make_retriever(embedding_model,vectorstore_path,search_type,search_kwargs)
memory=setup_memory(memory_k)
llm=initialize_llm()



# Define a route for the root to serve the index.html
@app.get("/", response_class=HTMLResponse)
async def get_index():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "static", "index.html")
    with open(file_path, "r") as f:
        return HTMLResponse(content=f.read())



@app.post("/chat", response_model=MessageResponse)
async def chat(request: MessageRequest):
    session_id = request.session_id
    question = request.message
    
    if not question:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if session_id not in session_histories:
        raise HTTPException(status_code=404, detail="Session expired or does not exist. Please start a new session.")

    # Check if the session exists; if not, initialize it
    if session_id not in session_histories:
        session_histories[session_id] = []

    # Retrieve the chat history for this session
    chat_history = session_histories[session_id]

    if question.lower() in ['exit', 'quit', 'bye']:
        return {"response": "Goodbye! Have a nice day. I hope I was of help to you!", "session_id": session_id}

    answer = querying(question, chat_history, session_id)
    chat_history.append((question, answer))
    session_histories[session_id] = chat_history
    session_timestamps[session_id] = datetime.now()

    return {"response": answer, "session_id": session_id}

@app.get("/new_session")
async def new_session():
    """Generate a new unique session ID."""
    new_session_id = str(uuid4())
    session_timestamps[new_session_id] = datetime.now()
    return {"session_id": new_session_id, "message": "New session created successfully."}


@app.on_event("startup")
@repeat_every(seconds=3600)  # Run every hour
def cleanup_sessions():
    now = datetime.now()
    for session_id, timestamp in list(session_timestamps.items()):
        if now - timestamp > timedelta(hours=1):  # Remove sessions older than 1 hour
            del session_histories[session_id]
            del session_timestamps[session_id]
            logger.info(f"Cleaned up session {session_id}")
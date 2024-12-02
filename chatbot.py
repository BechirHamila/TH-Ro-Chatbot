
# %%
import os
import groq
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

groq.api_key  = os.getenv('GROQ_API_KEY')
# %%
! pip install groq
! pip install "langchain==0.1.16" 
! pip install "langchain-ibm==0.1.4"
! pip install "huggingface == 0.0.1"
! pip install "huggingface-hub == 0.23.4"
! pip install "sentence-transformers == 2.5.1"
! pip install "chromadb == 0.4.24"
# %%
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
# %%

#Joining my Context text documents together
def joinfiles(file1,file2,outfile):
    filenames = [file1, file2]
    with open(outfile, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    return outfile

# %%
data='data.txt'
file1='studierenwerk_scraped.txt'
file2='thRo_scraped.txt'
outfile=joinfiles(file1,file2,outfile)
# %%

# Loading my data
from langchain.text_splitter import RecursiveCharacterTextSplitter

data='data.txt'
loader=TextLoader(data)
documents=loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n\n","\n","=================================================="]
)
texts=text_splitter.split_documents(documents)
print(len(texts))
type(texts)
texts[:10]

# %%
!rm -rf ./docs/chroma  # remove old database files if any
# %%

# Making the Vector Embeddings

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

model_name = "sentence-transformers/all-MiniLM-L6-v2"
persist_directory = 'docs/chroma/'
embedding = HuggingFaceEmbeddings(model_name=model_name)
vectordb= Chroma.from_documents(documents=texts,
                 embedding=embedding,
                 persist_directory=persist_directory,)
# %%

#Testing the functionality of the DB

question = "What english level do i need to study AAI bachelor "
docs = vectordb.similarity_search(question,k=3)
len(docs)
docs

# %%
# %%

# making my Chatbot

import os

from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq


def main():
    """
    This function is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.
    """

    # Get Groq API key
    groq_api_key = os.environ['GROQ_API_KEY']
    model = 'llama3-8b-8192'
    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name=model
    )
    
    print("Hello there! I'm your smart university assistant chatbot. I’m here to help you with anything about the university—like finding today’s Mensa menu, upcoming events, or answers about your courses. Ask away!")

    # A refined and engaging system prompt
    system_prompt = """
    You are a highly intelligent and empathetic university assistant chatbot. Your purpose is to help students, staff, and visitors with questions about the university. 
    You provide clear and accurate information about any information about the university.
    No hallucinations of wrong factuial information generated. If you don't know something say i na friendly manner that you dont know.
    You are friendly, professional, and concise in your responses.
    """

    # Define the number of previous messages to remember
    conversational_memory_length = 10  # Increased for richer context retention

    # Initialize conversational memory
    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    # Start the conversation loop
    while True:
        user_question = input("You: ")

        # If user provides a question
        if user_question.strip():
            # Construct a detailed and responsive chat prompt
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content=system_prompt
                    ),  # Core instructions for the chatbot
                    MessagesPlaceholder(
                        variable_name="chat_history"
                    ),  # Inserts past conversation for context
                    HumanMessagePromptTemplate.from_template(
                        "Student asked: {human_input}. Provide a clear and helpful response."
                    ),  # Frame user input as a request for specific assistance
                ]
            )

            # Create the conversation chain
            conversation = LLMChain(
                llm=groq_chat,  # Your LLM object
                prompt=prompt,  # Custom prompt template
                verbose=False,  # Set to True for debugging
                memory=memory,  # Memory for tracking chat history
            )

            # Generate and display the chatbot's response
            response = conversation.predict(human_input=user_question)
            print("TH-Rosenheim Assistant:", response)

if __name__ == "__main__":
    main()
# %%

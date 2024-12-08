
import os
import groq
import sys
import yaml

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import  ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv

sys.path.append('../..')
_ = load_dotenv(find_dotenv()) # read local .env file
groq.api_key  = os.getenv('GROQ_API_KEY')


def load_config(config_file):
    with open(config_file, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

groq_api_key = os.environ['GROQ_API_KEY']
config_file="config\config.yaml"
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
quit_msg=config['quit_msg']


def main():
    
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
    

    def qa_chain(prompt):
        return RetrievalQA.from_chain_type(
            llm,
            retriever,
            memory,
            chain_type_kwargs={"prompt": prompt})
    

    def querying(question,chat_history):
        chat_history= "\n".join([f"User: {question}\nAssistant: {answer}" for question, answer in chat_history])
        prompt = create_prompt(system_prompt, chat_history)  

        # Create the conversation chain
        qa_conv_chain_=qa_conv_chain(prompt)
        response=qa_conv_chain_({"question":question,"chat_history":chat_history})

        return response['answer'].strip()


    def chat_w_llm():
        chat_history=[]
        # Start the conversation loop
        while True:
            question = input("You: ")

            if question.lower() in ['exit', 'quit','bye']:
                # Getting a personalised Goodbye message
                quit_msg_=quit_msg
                response=querying(quit_msg_,chat_history)
                print("\nTH-Rosenheim Assistant:", response)
                break

            # if user input given
            if question.strip():        
                # Generate and display the chatbot's response
                response = querying(question,chat_history)
                print("\nTH-Rosenheim Assistant:", response)
                chat_history.append((question, response))
    
    
    
    retriever=make_retriever(embedding_model,vectorstore_path,search_type,search_kwargs)
    memory=setup_memory(memory_k)
    llm=initialize_llm()
    chat_w_llm()

        
if __name__ == "__main__":
    main()
    


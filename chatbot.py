
# %%
import os
import groq
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

groq.api_key  = os.getenv('GROQ_API_KEY')
# %%
#! pip install groq
#! pip install "langchain==0.1.16" 
#! pip install "langchain-ibm==0.1.4"
#! pip install "huggingface == 0.0.1"
#! pip install "huggingface-hub == 0.23.4"
#! pip install "sentence-transformers == 2.5.1"
#! pip install "chromadb == 0.4.24"

# %%
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain,LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate,MessagesPlaceholder,AIMessagePromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import DocArrayInMemorySearch

print("Imports successful!")
# %%

# making my Chatbot


# Groq-specific imports
from langchain_groq import ChatGroq
from groq import Groq


def main():
    """
    This function is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.
    """
    # Get Groq API key
    groq_api_key = os.environ['GROQ_API_KEY']
    model = 'llama3-8b-8192'
    # Initialize Groq Langchain chat object and conversation
    llm = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name=model,
            temperature=0.3,
            max_tokens=None
    )
    chat_history=[]
    
    print("Hello there! I'm your smart university assistant chatbot. I’m here to help you with anything about the university. Ask away!")

    system_prompt = """
    system: You are a highly intelligent and empathetic university assistant chatbot. Your purpose is to help students, staff, and visitors with questions about the university. 
    You provide clear and accurate information about any information about the university.
    No hallucinations of wrong factuial information generated. If you don't know something say i na friendly manner that you dont know.
    You are friendly, professional, and concise in your responses.
    """
    
    prompt = PromptTemplate(
            input_variables=["chat_history", "user_input"],
            template=f"""
            System: {system_prompt}

            Chat History:
            {{chat_history}}

            User Input:
            {{user_input}}

            Answer concisely and informatively:
            """
            )
    
    def querying(user_input,chat_history,prompt):


        # Initialize conversational memory
        memory = ConversationBufferWindowMemory(k=10, memory_key="chat_history", return_messages=True)


        #importing Vector Embeddings
        vectorstore_path='\\docs\\chroma'
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vs = Chroma(persist_directory=vectorstore_path, embedding_function=embedding_function)
        retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
                
    
        # Create the conversation chain
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            output_key='answer',
            get_chat_history=lambda h : h,
            verbose=True
        )

        result=qa({"user_input":user_input,"chat_history":chat_history})
        return result['answer'].strip()



        # Start the conversation loop
    
    while True:
        user_input = input("You: ")

        if user_input.lower() in ['exit', 'quit','bye']:
            print("Goodbye!")
            sys.exit()

        # if user input given
        if user_input.strip():        
            # Generate and display the chatbot's response
            response = querying(user_input,chat_history,prompt)
            print("\nTH-Rosenheim Assistant:", response)
            chat_history.append((user_input, response))

            

if __name__ == "__main__":
    main()
# %%

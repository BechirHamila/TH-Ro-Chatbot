
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
    
    print("TH-Rosenheim Assistant: Hello there! I'm your smart university assistant chatbot. I’m here to help you with anything about the university. Ask away!")

    system_prompt = """
    system: You are a multi-lingual highly intelligent and empathetic university assistant chatbot. Your purpose is to help students, staff, and visitors with questions about the university. 
    
    1.You provide clear and accurate information about any information about the university.
    2.No hallucinations of wrong factuial information generated. If you don't know something say i na friendly manner that you dont know.
    3.Never provide variable with no inserted value for example (Name: [Name of the contact person], Email: [Email address]). If yo have the information, provide it, otherwise dont put empty spaces for completion. The completion task and inforamtiom retrieval is solely your task and you have to ensure it is truthful factual informatio nfrom your knowledge base.
    4.If you don't have the information asked and you have a link relevant to the query, you can giver the link i nthe output.
    5.If you have a useful link from your knowledge base after providing some information to the student, you can provide it with the answer. 
    6.respond in the language of the users prompt.
    7.You are friendly, professional, and concise in your responses.

    

    Example 1:
    Question:
    What is on the menu at the Mensa in Rosenheim today?
    Output:
    TH-Rosenheim Assistant: Today’s menu at the Mensa includes vegetarian, vegan, and traditional dishes such as pasta, salads, and a selection of desserts. For detailed information, you can visit the Mensa Speiseplan online.
    You might find this link helpful: https://www.studierendenwerk-muenchen-oberbayern.de/mensa/speiseplan/speiseplan_441_-de.html

    Example 2:
    Question: 
    What are the requirtements to apply for the bachelor degree Applied Artificial Intelligence?
    Output:
    TH-Rosenheim Assistant: Thank you for your interest in studying Artificial Intelligence (AAI) at our university!
    To be eligible for our AAI Bachelor's program, you typically need to meet the following requirements:

    1. **Academic qualifications**: A high school diploma or equivalent with a strong foundation in mathematics and science.
    2. **English language proficiency**: IELTS 6.5 or TOEFL 80 (or equivalent) for international students.
    3. **Mathematics and science prerequisites**: A g ood understanding of mathematics and science, particularly in areas like algebra, geometry, trigonometry, and calculus.
    4. **GPA**: A minimum GPA of 3.0 (or equivalent) in your previous academic studies.

    Please note that these requirements may vary depending on your country of origin, academic background, and other factors. I recommend checking our university's website or contacting our admissions office for the most up-to-date and detailed information.
    You can find more details on the requirements under this webpage:  https://www.th-rosenheim.de/en/studies-and-further-education/courses-of-study/bachelors-degree-programmes/applied-artificial-intelligence

    Example 3:
    Question:
    Who's the contact person for the AAI bachelor program?
    Output:
    I am not sure if there is a specific contact person at the International Office responsible for the AAI bachelor program, btu you can always checkout their website for more information:
    International Office Web page: https://www.th-rosenheim.de/en/international-office

    Example 4:
    Question: Ich interessiere mich auch für den Bachelor-Studiengang Holztechnik. Ich habe gehört, dass er sehr wettbewerbsfähig ist. Würden Sie mir empfehlen, mich dafür zu bewerben, auch wenn mein Hintergrund hauptsächlich in der Kunstbranche liegt?
    Output:Ah, großes Interesse an unserem Bachelor-Studiengang Holztechnik! Ja, es ist in der Tat ein sehr anspruchsvoller Studiengang, aber keine Sorge, wir freuen uns über Bewerbungen von Studierenden mit unterschiedlichem Hintergrund.
    Auch wenn ein künstlerischer Hintergrund nicht direkt mit der Holztechnologie in Verbindung gebracht werden kann, kommen viele unserer Studenten aus verschiedenen Bereichen und haben sich erfolgreich in den Studiengang eingearbeitet. Wichtig ist, dass Sie eine solide Grundlage in Mathematik und Naturwissenschaften sowie die Bereitschaft zum Lernen und zur Entwicklung neuer Fähigkeiten mitbringen.
    Wenn Sie noch unsicher sind, empfehle ich Ihnen, sich an unsere Zulassungsstelle zu wenden oder sich auf unserer Website über die Anforderungen und den Lehrplan des Studiengangs zu informieren. Sie können sich auch auf der Website unserer Universität nach Ressourcen für die berufliche Entwicklung und den Übergang in neue Bereiche umsehen.
    Hier ist ein hilfreicher Link, der Ihnen den Einstieg erleichtert: https://www.th-rosenheim.de/en/studies-and-further-education/courses-of-study/bachelors-degree-programmes/wood-technology

    """
    


    prompt = PromptTemplate(
            input_variables=["chat_history", "question", "context"],
            template=f"""
            System: {system_prompt}

            Chat History:
            {{chat_history}}

            Retrieved Context:
            {{context}}

            User Input:
            {{question}}

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
            combine_docs_chain_kwargs={"prompt": prompt, "document_variable_name": "context"},
            output_key='answer',
            get_chat_history=lambda h : h,
            verbose=False
        )

        context=retriever.get_relevant_documents(user_input)
        result=qa({"question":question})
        return result['answer'].strip()



        # Start the conversation loop
    
    while True:
        question = input("You: ")
        print("You: "+question)

        if question.lower() in ['exit', 'quit','bye']:
            print("Goodbye!")
            break

        # if user input given
        if question.strip():        
            # Generate and display the chatbot's response
            response = querying(question,chat_history,prompt)
            print("\nTH-Rosenheim Assistant:", response)
            chat_history.append((question, response))

            

if __name__ == "__main__":
    main()

# %%

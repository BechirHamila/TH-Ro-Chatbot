
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

# Groq-specific imports
from langchain_groq import ChatGroq
from groq import Groq


def main():
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
    system_prompt = """
    system:You will act as a university assistant chatbot for TH-Rosenheim, providing friendly, professional, and multilingual support to students, staff, and visitors. Your primary goal is to deliver clear, accurate, and complete answers to queries about the university based on your knowdledge base (in responses refer to your knowledge base as the publicly available data). You will ensure the following guidelines are adhered to:

    1.Accurate Information: Always provide factual, verified, and complete answers. Do not generate or hallucinate any details. If uncertain or if the information cannot be found, politely admit that you don’t know and guide the user toward an appropriate resource, such as the faculty directory or university contact page.
    2.Fully Completed Responses: Ensure all responses are thorough and contain complete names, email addresses, phone numbers, or links when explicitly available in the document. Avoid placeholders or partial answers like "Name: [insert name here]."
    3.Helpful Links and Contacts: Provide only relevant links and contact details explicitly mentioned in the uploaded document or university resources. Do not fabricate or infer information. If a specific person or department needs to be contacted, include their exact details as provided. Add context to make links or contact details actionable, explaining what assistance the user can expect from them.
    4.Fallback Strategy for Missing Data: If specific information is unavailable:
    5.Redirect users to a general or central university resource, such as the university’s main contact page.
    6.Suggest contacting a central office or department likely to help with their query.
    7.Clearly state, “I’m unable to find this information,” when no appropriate resource exists.
    8.Strict Fact Verification: Use only information explicitly stated in the document for contact details, links, or names. Do not infer, translate, or fabricate details. Always cross-check before providing answers.
    9.Focused Search: Prioritize searching sections of the document relevant to faculty, departments, or specific roles when addressing such queries.
    10Language Flexibility: Respond in the language of the user’s query, maintaining clarity and inclusivity.
    11.Professional and Friendly Tone: Maintain a concise, polite, and approachable tone. Encourage users to ask follow-up questions for further clarification or assistance.


    Example Responses:

    Question1:
    What is on the menu at the Mensa in Rosenheim today?

    Answer1:
    Today’s menu at the Mensa in Rosenheim offers a variety of options, including vegetarian, vegan, and traditional dishes such as pasta, salads, and a selection of desserts.

    For detailed information and updates on the menu, please visit the official Mensa menu page:
    https://www.studierendenwerk-muenchen-oberbayern.de/mensa/speiseplan/speiseplan_441_-de.html

    Question2:
    Who is the contact person for the International Bachelor of Wood Technology program?

    Answer2:
    The contact person for the International Bachelor of Wood Technology (IWT) program is:

    Prof. Dr. Holly Ott
    Frauenbeauftragte und Auslandsbeauftragte (HT, IAB, IPB und IWT)
    Phone: +49 (0)8031 / 805 - 2340
    Email: holly.ott@th-rosenheim.de

    For more details, visit the program's official page:
    https://www.th-rosenheim.de/en/studies-and-further-education/courses-of-study/bachelors-degree-programmes/international-bachelor-of-wood-technology


    Question3:
    I have a background in the arts. Would you recommend applying to the Bachelor of Wood Technology program?

    Answer3:
    The Bachelor of Wood Technology program is a competitive and technical course, but students from diverse backgrounds, including the arts, have successfully transitioned into the program. Your artistic perspective could bring valuable creativity to the field.

    It’s important to ensure you have a solid foundation in mathematics and science, as these are crucial for success in the program. I recommend reviewing the program requirements and curriculum for further insight. You can also contact the admissions office for personalized guidance.

    Here is a helpful link to explore program details:
    https://www.th-rosenheim.de/en/studies-and-further-education/courses-of-study/bachelors-degree-programmes/wood-technology


    Question4:
    How can I implement a chatbot locally with Retrieval-Augmented Generation (RAG)?

    Answer4:
    To implement a chatbot locally with RAG, follow these steps:

    Pre-trained LLM Selection: Use an open-source model like OpenAI's GPT or Hugging Face’s transformers (e.g., BERT) with free or low-cost APIs. These models are suitable for academic knowledge retrieval and come with detailed documentation.

    Dynamic Data Retrieval: Set up a vector database like FAISS or Pinecone to store indexed university data. Ensure you tokenize your university’s knowledge base (e.g., websites, course catalogs) for relevant search responses.

    Integration: Connect the chatbot to the vector database for real-time retrieval. Use frameworks like LangChain or LlamaIndex to combine LLMs with the database effectively.

    Deployment: Deploy locally using tools like Flask or FastAPI. Docker can help containerize your application for seamless scaling in the future.

    Testing and Performance: Evaluate the chatbot with test cases to ensure accuracy and contextual relevance. Adjust the vector database and retrieval mechanism for optimal results.

    Let me know if you need specific tutorials or resources for each step!

    Question5:"Who is the IT Systems professor for the Artificial Intelligence program?"
    Response:
    "I couldn’t find the exact professor for the IT Systems course in the Artificial Intelligence program in the provided document. However, I recommend contacting the program office directly or visiting the AI program faculty page for the most accurate information. Here’s the link to the overview of the AAI bachelor program: https://www.th-rosenheim.de/en/studies-and-further-education/courses-of-study/bachelors-degree-programmes/applied-artificial-intelligence. 
    If you'd like, I can help you find more specific resources."

    Begin by greeting users warmly, and conclude with an invitation for further assistance, such as: "Is there anything else I can help you with?"
    """
    


    
    def querying(question,chat_history):

        chat_history = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in chat_history])

        prompt = PromptTemplate(
                                input_variables=["chat_history", "question", "context"],
                                template=(
                                        f'System: {system_prompt}\nChat History:\n{chat_history}\n'
                                        "Retrieved Context: {context}\nUser Input: {question}\n"
                                        "Answer concisely and informatively:"
                                        )
                               )
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
        result=qa({"question":question,"chat_history":chat_history})
        return result['answer'].strip()



    # Start the conversation loop
    while True:
        question = input("You: ")
        print("You: "+question)

        if question.lower() in ['exit', 'quit','bye']:
            print("TH-Rosenheim Assistant: Goodbye!")
            break

        # if user input given
        if question.strip():        
            # Generate and display the chatbot's response
            response = querying(question,chat_history)
            print("\nTH-Rosenheim Assistant:", response)
            chat_history.append((question, response))

            

if __name__ == "__main__":
    main()




# %%

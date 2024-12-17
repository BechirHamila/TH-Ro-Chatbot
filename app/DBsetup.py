# %%
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

import yaml
import os
import shutil


def load_config(config_file):
    with open(config_file, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    
config_file="../config/config.yaml"
config = load_config(config_file)


#Joining my Context text documents together
def joinfiles(file1,file2,outfile):
    filenames = [file1, file2]
    with open(outfile, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    return outfile


'''
data='data.txt'
file1='studierenwerk_scraped.txt'
file2='thRo_scraped.txt'
outfile=joinfiles(file1,file2,outfile)'''



# Loading my data
def split_data_to_chunks(data):
    loader=TextLoader(data)
    documents=loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["=================================================="]
    )
    texts=text_splitter.split_documents(documents)
    print(len(texts))

    return texts





'''#Test
print(len(texts))
type(texts)
texts[:10]'''



# Making the Vector Store
data_path=config['data_path']
embedding_model = config['vectorstore']['embedding_model']
vectorstore_path_FAISS=config['vectorstore']['path_FAISS']
vectorstore_path_chroma=config['vectorstore']['path_chroma']

def setup_vs(vs_path,texts):
    embedding_function=HuggingFaceEmbeddings(model_name=embedding_model)
    vs = FAISS.from_documents(documents=texts,embedding=embedding_function)
    vs.save_local(vs_path)
    


vectorstore_path_FAISS = config['vectorstore']['path_FAISS']  # Replace with your vectorstore path
vectorstore_path_chroma= config['vectorstore']['path_chroma'] 

# Delete the existing FAISS vectorstore directory
if os.path.exists(vectorstore_path_FAISS):
    shutil.rmtree(vectorstore_path_FAISS)
    print("FAISS vectorstore has been reset.")


data=config['data_path']
texts=split_data_to_chunks(data)
vs=setup_vs(vectorstore_path_FAISS,texts)





# %%
#Testing the functionality of the VS
question = "What english level do i need to study AAI bachelor "
docs = vs.similarity_search(question,k=3)
len(docs)
print(docs)


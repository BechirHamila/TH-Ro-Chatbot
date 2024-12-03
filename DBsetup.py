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
! rm -rf ./docs/chroma  # remove old database files if any
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
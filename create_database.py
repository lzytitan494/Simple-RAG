from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

CHROMA_PATH = "./chroma"
DATA_PATH = "./sample_data"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3:8b"

#---------------------------------------------------------------------------------

# Ensure Ollama Models are Downloaded
def ensure_ollama_model(model_name):
    try:
        Ollama(model=model_name)  # Attempt to load the model
        print(f"Ollama model '{model_name}' found.")
    except Exception as e:
        raise Exception(f"Ollama model '{model_name}' not found. Please download it using 'ollama pull {model_name}'")

ensure_ollama_model(EMBEDDING_MODEL)
ensure_ollama_model(LLM_MODEL)

#---------------------------------------------------------------------------------

# Set embeddings
# !ollama pull nomic-embed-text --> Run this before
embed = OllamaEmbeddings(model=EMBEDDING_MODEL)

#---------------------------------------------------------------------------------

# Load documents
docs = DirectoryLoader(DATA_PATH, glob="*.txt", show_progress=True).load()

#---------------------------------------------------------------------------------

# Rephrase the documents
# !ollama pull llama3:8B --> Run this before

llm = Ollama(model=LLM_MODEL)
prompt = PromptTemplate(
    template="""For the following paragraph give me a paraphrase of the same in high quality English language as 
    in sentences on Wikipedia.
    
    ## Paragraph:
    {doc}

    ## Paraphrased paragraph:
    """,
    input_variables=["doc"]
)

chain = prompt | llm

print('Rephrasing:\n')
for i in range(len(docs)):
    docs[i].page_content = chain.invoke({"doc": docs[i].page_content})
    print("-> ",i+1, "\n", docs[i].page_content, "\n\n")


#---------------------------------------------------------------------------------

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 512, chunk_overlap = 0
)

doc_splits = text_splitter.split_documents(docs)

#---------------------------------------------------------------------------------

print('Storing in Database:')
# Add to vectorstore
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    embedding=embed,
    collection_name="RAG_data",
    persist_directory=CHROMA_PATH
)
print('Database created successfully!')

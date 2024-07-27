from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

CHROMA_PATH = "./chroma"
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

# Set embedding and load the database
# !ollama pull nomic-embed-text --> Run this before
embed = OllamaEmbeddings(model="nomic-embed-text")

try:
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embed)
except Exception as e:
    print(f"Error loading Chroma database: {e}")

#---------------------------------------------------------------------------------

# query
query = "After graphic driver update, I'm facing flickering issue while playing games?"
print("Query: ", query)
# Retrieve documents
retrieved_docs = vectorstore.similarity_search(query, k=5)

# Display the retrieved documents
print("Retrieved Documents:")
for i, doc in enumerate(retrieved_docs):
    print(f"Document {i+1}:")
    print(doc.page_content)
    print("\n")

#---------------------------------------------------------------------------------
if retrieved_docs != []:
    # llm
    # !ollama pull llama3:8B --> Run this before
    llm = Ollama(model='llama3:8b')

    # prompt
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = PromptTemplate(
        template="""Using the following context, answer the question.

        ## Context:
        {context}

        ## Question:
        {query}

        ## Answer:
        """,
        input_variables=["context", "query"]
    )

    chain = prompt | llm

    # generate response
    response = chain.invoke({"context": context, "query": query})
    print("Generated Response:\n", response)
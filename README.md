## Laptop RAG Pipeline with Ollama and ChromaDB

This project implements a Retrieval Augmented Generation (RAG) pipeline for answering questions about laptop issues. It leverages the Laptop Wiki community as a knowledge base and uses powerful open-source language models from Ollama for information extraction and question answering.

<p align="center">
<img src="https://github.com/lzytitan494/Simple-RAG/blob/main/RAG.png" alt="RAG Pipeline Diagram" width="600"/>
</p>

### How it Works

1. **Data Extraction and Refinement:**
    * Data is extracted from the Laptop Wiki community.
    * The extracted data is rephrased and refined using the `llama3:8b` LLM from Ollama to ensure high-quality language and consistency.

2. **Vector Database Creation:**
    * The refined data is split into chunks and embedded using the `nomic-embed-text` model from Ollama.
    * These embeddings are stored in a ChromaDB vector database for efficient similarity search.

3. **Question Answering:**
    * User queries are embedded using the same `nomic-embed-text` model.
    * The ChromaDB database is queried for the most relevant chunks based on similarity to the query embedding.
    * The retrieved chunks, along with the original query, are fed into the `llama3:8b` LLM with a specific prompt to generate a comprehensive and helpful answer.

### Files

* **`create_database.py`:**  This script handles the entire pipeline for creating the vector database:
    * Downloads the required Ollama models (`llama3:8b`, `nomic-embed-text`).
    * Loads data from text files in the `sample_data` directory.
    * Rephrases the loaded data using the `llama3:8b` LLM.
    * Splits the data into chunks, embeds them, and stores them in the ChromaDB database at the specified path.

* **`query_database.py`:**  This script loads the created database and answers user queries:
    * Downloads the required Ollama models.
    * Loads the ChromaDB database from the specified path.
    * Takes a user query as input, retrieves relevant information from the database, and feeds it to the `llama3:8b` LLM to generate a response. 

### Instructions for Running

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
2. **Download Ollama Models:**
   ```bash
   ollama pull llama3:8b
   ollama pull nomic-embed-text 
   ```

3. **Prepare your data:**
   * Place your Laptop Wiki data in text files within the `sample_data` directory. 

4. **Create the Database:**
   ```bash
   python create_database.py
   ```

5. **Run the Question Answering System:**
   ```bash
   python query_database.py
   ```
   You can then input your laptop-related queries.

### Notes:

* Ensure that the paths to your data (`DATA_PATH`) and database (`CHROMA_PATH`) are correctly set in the scripts.
* ChromaDB persistency can sometimes be finicky. If you encounter issues, try deleting the existing database directory and recreating it.
* This project is a starting point, and you can further customize it by:
    * Adding more data sources.
    * Fine-tuning the LLMs for your specific use case. 
    * Experimenting with different embedding models and prompt engineering for better results. 

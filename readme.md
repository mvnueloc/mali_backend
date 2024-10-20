### MALI Backend - Architecture Overview

This document provides an overview of the backend architecture for the application MALI, describing the flow of data, key components, and interactions within the system.

![image](https://github.com/user-attachments/assets/25340607-1860-4c77-904a-cccc4ace36b5)

## **Architecture Components:**

1. **Open Government Data (Datos abiertos del gobierno):**
   - **Input:** The system starts by gathering open data provided by government sources. These datasets come in various formats such as CSV, JSON, or PDF files.
   - **Processing:** This raw data is processed and transformed into usable formats for further steps in the pipeline.

2. **Information Processing (Información):**
   - The collected open data is transformed into embeddings, which are vector representations of the data that capture the semantic meaning of the content. These embeddings are created based on the features and structure of the input data.

3. **Embeddings:**
   - **Role:** Embeddings represent data points in a multidimensional space, allowing for similar data points to be grouped together. In the diagram, different clusters represent different categories (e.g., ROCK, FOLK, REGGAE, POP) based on similarities in the dataset.

4. **Vector Databases (BBDD vectoriales):**
   - **Function:** This is a database specifically designed to store and query vectorized data (embeddings). It allows for efficient similarity searches and clustering operations, facilitating tasks like recommendation or information retrieval.
   
5. **RAG (Retrieval-Augmented Generation):**
   - **Role:** RAG is an architecture that combines a retrieval mechanism with a generative model. In this case, it retrieves relevant information from the vector database and passes it to the next component for further processing.
   
6. **LLaMA:**
   - **Function:** LLaMA is a large language model responsible for processing natural language inputs and generating responses. It interacts with the vector database through the RAG mechanism, allowing the model to retrieve relevant information from the stored embeddings and provide accurate answers.
   
7. **Server (Servidor):**
   - **Technologies:** The backend server is built using Python and Flask. It is responsible for handling requests from the frontend (e.g., the web page), interacting with the LLaMA model, managing the RAG process, and communicating with the vector database.
   
8. **Web Page (Página Web):**
   - **Interface:** The web page is the frontend of the MALI application, where users can interact with the system. They can input queries or requests, which are sent to the server. The server, in turn, interacts with the language model and other components to retrieve the relevant data and display it back on the page.
   - The web page provides an interface to access personal information, which is retrieved through the entire backend pipeline.

## **Data Flow Summary:**

1. **Data Input:** The pipeline starts with government open data being ingested and processed into embeddings.
2. **Embedding Storage:** The embeddings are stored in a vector database, where data similarity can be queried.
3. **RAG Interaction:** When a user query is made via the web page, the server sends the request to the LLaMA model.
4. **Information Retrieval:** The LLaMA model uses the RAG mechanism to pull relevant data from the vector database.
5. **Response Generation:** The processed data is sent back to the server and displayed on the web page for the user.

## **Technological Stack:**

- **Data Sources:** CSV, JSON, PDF (Government Open Data)
- **Embedding Creation:** Feature extraction and data representation into vectors
- **Vector Storage:** Vector databases for efficient similarity searching
- **RAG Model:** Combines retrieval and generative model for enhanced performance
- **Language Model:** LLaMA for natural language processing and generation
- **Backend Server:** Python and Flask to handle API requests and model interaction
- **Frontend:** Web interface whit React.JS.

This architecture allows MALI to handle complex natural language queries, retrieve relevant government data efficiently, and present the results to users in an intuitive interface.

## Run Locally

Download Ollama

https://ollama.com/library/llama3.1

Clone the project

```bash
  git clone https://github.com/mvnueloc/mali.git
```

Go to the project directory

```bash
  cd rag-ollama
```

Install dependences

```bash
  conda create -n rag-ollama python=3.10.15
```


```bash
  conda activate rag-ollama
```


```bash
  pip install -r requirements.txt
```


Run the script

```bash
  python main.py
```

## Add new document

Save the document you want to add to the knowledge base in the /documents folder. 
Supported files: ".pdf", ".csv", ".txt", ".json", ".docx", ".md"


```bash
  python local_rag.py
```




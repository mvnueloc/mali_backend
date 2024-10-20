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




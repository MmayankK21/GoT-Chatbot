

Game of Thrones Trivia Q&A Chatbot

An interactive **Game of Thrones** trivia chatbot that uses a Large Language Model (LLM) from **Ollama**, **LangChain** for RAG (Retrieval-Augmented Generation), and **ChromaDB** for vector search. It fetches content from Wikipedia to create a knowledge base and provides accurate, context-aware answers to user queries with an intuitive **Streamlit UI**.

![image](https://github.com/user-attachments/assets/2e7cc0be-ca18-42c9-ab86-5bef9fa7bd14)

---

## Features

- ⚡️ Ask trivia questions related to *Game of Thrones*
- 🔍 Uses Retrieval-Augmented Generation to improve accuracy and reduce hallucinations
- 📚 Context built using Wikipedia API + ChromaDB
- 🧠 Powered by `llama3.2` model from Ollama
- 📊 Evaluation metrics with ROUGE and BERTScore
- 💬 Clean and interactive Streamlit chat UI

---

## Tech Stack

| Component         | Technology                          |
|------------------|--------------------------------------|
| UI               | Streamlit                            |
| Backend LLM      | Ollama (`llama3.2:latest`)           |
| Embeddings       | `OllamaEmbeddings`                   |
| RAG Framework    | LangChain                            |
| Vector Store     | Chroma                               |
| Knowledge Source | Wikipedia (via `wikipedia-api`)      |
| Evaluation       | ROUGE, BERTScore                     |

---

## Project Structure

```
.
├── app.py                  # Streamlit UI and main chatbot logic
├── build_knowledge_base.py # Script to build Chroma vector DB using Wikipedia
├── test_questions.json     # Sample test trivia questions
├── got_db_llama3.2/        # Persisted vector database directory
├── README.md               # You are here!
```

---

## How It Works

1. **Knowledge Base Creation (`build_knowledge_base.py`):**
   - Scrapes summary and sections from the *Game of Thrones* Wikipedia page.
   - Splits content into chunks using `RecursiveCharacterTextSplitter`.
   - Generates vector embeddings using `OllamaEmbeddings`.
   - Stores chunks in **Chroma** vector database.

2. **Chat Application (`app.py`):**
   - Loads the vector DB and sets up a LangChain `RetrievalQA` chain.
   - Accepts user input via Streamlit chat.
   - Retrieves relevant chunks from the DB and queries the `llama3.2` LLM.
   - Displays the answer along with optional source and evaluation scores.

3. **Evaluation (Optional):**
   - If `rouge-score` and `bert-score` are installed, model responses are evaluated and displayed in real-time.

---

## Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/got-trivia-bot.git
cd got-trivia-bot
```

### 2. Install Dependencies

Make sure you have Python 3.8+ and [Ollama](https://ollama.com) installed.

```bash
pip install -r requirements.txt
```

<details>
<summary> Example <code>requirements.txt</code></summary>

```txt
streamlit
langchain
langchain-community
langchain-core
langchain-ollama
chromadb
wikipedia-api
rouge-score
bert-score
pandas
```
</details>

### 3. Start Ollama and Pull Model

```bash
ollama run llama3
```

Make sure Ollama is running locally and `llama3.2` model is available.

### 4. Build the Knowledge Base

```bash
python build_knowledge_base.py
```

This will create the Chroma DB and save sample trivia questions.

### 5. Run the App

```bash
streamlit run app.py
```

---

## Evaluation Metrics

If `rouge-score` and `bert-score` are installed, the app computes:

- **ROUGE-1 / ROUGE-2 / ROUGE-L** F1-scores
- **BERTScore Precision / Recall / F1**

Evaluation results are shown under the **Evaluation Report** tab.

---

## Vector DB Architecture

- **Source**: Wikipedia API content
- **Splitter**: RecursiveCharacterTextSplitter (800 tokens, 150 overlap)
- **Embedding Model**: `OllamaEmbeddings` (llama3.2)
- **Storage**: ChromaDB (persisted in `got_db_llama3.2/`)
- **Retrieval Strategy**: Top-k (k=2) vector similarity

```text
Wikipedia Text → Chunk Splitter → Ollama Embeddings → Chroma Vector DB
                                                ↘ Retrieval → LLM → Answer
```

---

## Possible Improvements

- Add character-specific subpages to improve answer depth.
- Add multilingual support for non-English fans.
- Integrate speech input/output for accessibility.
- Deploy with Docker or Streamlit Cloud.


# 🧑‍💻 **Codebase Chatbot with Retrieval-Augmented Generation (RAG)**  

An AI-powered chatbot that allows users to interact with and understand codebases by leveraging **Retrieval-Augmented Generation (RAG)**. This tool embeds the content of code repositories, stores them in a vector database, and uses Large Language Models (LLMs) to answer queries contextually.

---

## 🚀 **Features**  

- **Chat with a Codebase**: Understand the structure, purpose, and potential improvements of any codebase.  
- **Preloaded Repositories**: Seamlessly switch between preloaded repositories to explore different projects.  
- **Accurate Contextual Answers**: Powered by LLMs, providing insights based on embedded code content.  
- **Future Enhancements**: Plan to allow dynamic uploads of any GitHub repository for embedding and querying.

---

## 🛠️ **Tech Stack**  

- **Python**: Core language for implementation.  
- **Hugging Face Transformers**: Used for generating embeddings with the `sentence-transformers/all-mpnet-base-v2` model.  
- **Pinecone**: A vector database to store and retrieve code embeddings.  
- **Streamlit**: Frontend framework to provide an interactive and user-friendly UI.  
- **OpenAI LLMs**: For generating accurate, context-aware responses.  

---

## ⚙️ **How it Works**

- **Create Vector Embeddings**: Using Hugging Face model, create embeddings of relevant information like function definitions, comments, and documentation from the codebase.
- **Store the embeddings**: Used Pinecone Vector Database
- **Query the Codebase**: When a query is made, relevant pieces of the codebase are retrieved from Pinecone and augmented with the query before being sent to the LLM for a response.
- **Interactive Chat**: Users can ask questions through a Streamlit-based UI, select from preloaded repositories, and receive responses in real time.

---

## 📋 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/codebase-rag.git
cd codebase-rag
```
### 2.  Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Pinecone
- Create a Pinecone account at Pinecone.io.
- Get your Pinecone API key and index name.
- Configure your .env file with Pinecone credentials.

### 4. Run the Application
```bash
streamlit run app.py
```
---

## 🌟 Next Steps

- Allow users to upload custom GitHub repositories for embedding.
- Enhance the UI for better interactivity.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for discussion.

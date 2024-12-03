import streamlit as st
import os
from github import Github
from git import Repo
from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from api_utils import list_repositiories, get_api_response
import time

# Initialize Pinecone
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
pinecone_index = pc.Index("codebase-rag")

# Initialize OpenAI client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=st.secrets["GROQ_API_KEY"]
)

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)

# Perform rag
def perform_rag(query, namespace):
   # Embed the query
   raw_query_embedding = get_huggingface_embeddings(query)

   # Find the top_matches
   top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k=5, include_metadata=True, namespace=namespace)
 
   # Get the list of retrieved texts
   contexts = [item['metadata']['text'] for item in top_matches['matches']]

   # Augment the query with contexts retrieved
   augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[ : 10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

   # Modify the prompt below as needed to improve the response quality
   system_prompt = f"""You are a Senior Software Engineer, specializing in Typescript and Python.


   Answer any questions I have about the codebase, based on the code provided. Always consider all of the context provided when forming a response.

   If there are points, format it and make sure each point starts at a new line.
   """


   llm_response = client.chat.completions.create(
       model="llama-3.1-8b-instant",
       messages=[
           {"role": "system", "content": system_prompt},
           {"role": "user", "content": augmented_query}
       ]
   )


   return llm_response.choices[0].message.content


# Streamed response emulator
def response_generator(prompt, repo):
    response = perform_rag(prompt, repo)
    for word in response.split("\n"):
        yield word + "\n"
        time.sleep(0.05)


st.title("🤖 CodeSage 🤖")

# Sidebar
st.sidebar.title("💡 About")
st.sidebar.info(
    "CodeSage answers your questions on a specific codebase using RAG (Retrieval Augmented Generation)."
)
st.sidebar.title("Select Github Repo")

selected_repo = "https://github.com/CoderAgent/SecureAgent"

st.write(f"You have selected the repository: {selected_repo}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt, selected_repo))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
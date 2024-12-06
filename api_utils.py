import streamlit as st
import requests

API_BASE_URL = "http://localhost:8501"

# Query the API to get a response
def get_api_response(question, repo):
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    data = {"query": question, "repo": repo}

    try:
        response = requests.post(f"{API_BASE_URL}/query", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API request failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def list_repositiories():
    try: 
        response = requests.get(f"{API_BASE_URL}/repos")
        if response.status_code == 200:
            return response.json().get("repos", [])
        else:
            st.error(f"Failed to fetch repositories. Error: {response.status_code} - {response.text}") 
            return []
    except Exception as e:
        st.error(f"An error occurred while fetching repositories: {str(e)}.")    
        return []


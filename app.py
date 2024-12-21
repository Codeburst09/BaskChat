import streamlit as st
from langchain_groq import ChatGroq
import chromadb
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import uuid

# Ollama Llama 3.2 Configuration
API_BASE = "http://localhost:11434"
MODEL = "ollama/llama3.2"

# Create a ChatGroq instance
llm = ChatGroq(
    temperature=0,
    groq_api_key='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',
    model_name="llama-3.1-70b-versatile"
)

# Initialize Chroma client
client = chromadb.Client()

# Ensure unique collection handling
if "my_collection" in [col.name for col in client.list_collections()]:
    collection = client.get_collection(name="my_collection")
else:
    collection = client.create_collection(name="my_collection")

# Function for adding documents to the collection
def add_document_to_collection(doc, doc_id):
    collection.add(
        documents=[doc],
        ids=[doc_id]
    )
    st.write(f"Document added with ID: {doc_id}")
    all_docs = collection.get()
    st.write("Current documents in collection:", all_docs)

# Function for querying the collection
def query_collection(query_text):
    results = collection.query(
        query_texts=[query_text]
    )
    return results

# Function to format the services in a user-friendly way
def format_services(services):
    formatted_services = "\n".join([f"{key}: {value}" for key, value in services.items()])
    return f"Here are the services we provide:\n{formatted_services}"

# Function to handle large input sizes
def split_text_into_chunks(text, chunk_size=1000):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Streamlit UI
st.title("Customer Support Chatbot")
st.subheader("Powered by Bask Energy Systems")

user_input = st.text_input("How can I assist you today?", "")

if user_input:
    if "services" in user_input.lower():
        # Load a website page to extract services
        loader = WebBaseLoader("https://www.energetica-india.net/powerful-thoughts/online/chandresh-jain")
        page_data = loader.load().pop().page_content
        st.write("Page content loaded. Extracting services...")

        # Reduce input size by chunking if needed
        chunks = split_text_into_chunks(page_data)
        extracted_services = {}

        for i, chunk in enumerate(chunks):
            st.write(f"Processing chunk {i + 1}/{len(chunks)}...")
            prompt_extract = PromptTemplate.from_template(
                """
                Extract Company name, Deatils from the following text in JSON format with keys `Company name`, `Desscription`,.
                ### TEXT:
                {chunk}
                """
            )
            chain_extract = prompt_extract | llm
            res = chain_extract.invoke(input={'chunk': chunk})
            json_parser = JsonOutputParser()

            try:
                json_res = json_parser.parse(res.content)
                extracted_services.update(json_res)
            except Exception as e:
                st.error(f"Error parsing response for chunk {i + 1}: {e}")

        # Format and display the extracted services
        formatted_services = format_services(extracted_services)
        st.write(formatted_services)

        # Add the extracted services to the collection
        document = str(extracted_services)
        doc_id = f"service_{uuid.uuid4()}"
        add_document_to_collection(document, doc_id)

    elif "query" in user_input.lower():
        query_text = st.text_input("Enter your query:", "")
        if query_text:
            results = query_collection(query_text)
            st.write(f"Query Results: {results}")

    else:
        # LLM generates a response for unexpected inputs
        st.write("Let me check that for you...")
        response = llm.invoke(user_input)

        # Prepend the focused message to LLM response
        focused_response = "I am centered only to help regarding Bask Energy. " + response.content
        st.write(f"LLM Response: {focused_response}")

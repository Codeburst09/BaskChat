from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import uuid

# FastAPI app
app = FastAPI()

# Initialize Chroma client
client = chromadb.Client()

# Ensure unique collection handling
if "my_collection" in [col.name for col in client.list_collections()]:
    collection = client.get_collection(name="my_collection")
else:
    collection = client.create_collection(name="my_collection")

# Ollama Llama 3.2 Configuration
llm = ChatGroq(
    temperature=0,
    groq_api_key='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',
    model_name="llama-3.1-70b-versatile"
)

# Request model
class QueryRequest(BaseModel):
    query: str

# Function for adding documents to the collection
def add_document_to_collection(doc, doc_id):
    collection.add(
        documents=[doc],
        ids=[doc_id]
    )

# Function for querying the collection
def query_collection(query_text):
    results = collection.query(query_texts=[query_text])
    return results

# Function to handle large input sizes
def split_text_into_chunks(text, chunk_size=1000):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Chatbot endpoint
@app.post("/chatbot")
async def chatbot(request: QueryRequest):
    user_input = request.query

    if "services" in user_input.lower():
        # Load website data and extract services
        loader = WebBaseLoader("https://economictimes.indiatimes.com/company/bask-energy-systems-india-privatelimited/U74999TG2016PTC109659")
        try:
            page_data = loader.load().pop().page_content
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error loading web data")
        
        # Process and extract services
        chunks = split_text_into_chunks(page_data)
        extracted_services = {}

        for chunk in chunks:
            prompt_extract = PromptTemplate.from_template(
                """
                Extract Company name and Details from the following text in JSON format with keys `Company name`, `Description`.
                ### TEXT:
                {chunk}
                """
            )
            chain_extract = prompt_extract | llm
            response = chain_extract.invoke(input={'chunk': chunk})
            json_parser = JsonOutputParser()
            try:
                json_res = json_parser.parse(response.content)
                extracted_services.update(json_res)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error parsing response: {e}")

        # Add extracted services to the collection
        document = str(extracted_services)
        doc_id = f"service_{uuid.uuid4()}"
        add_document_to_collection(document, doc_id)

        return {"services": extracted_services}

    elif "query" in user_input.lower():
        # Query the collection
        query_results = query_collection(user_input)
        return {"query_results": query_results}

    else:
        # General LLM response
        response = llm.invoke(user_input)
        return {"response": response.content}

# Run the server using Uvicorn (if not using a production server like Gunicorn)
# Run this script with the command:
# uvicorn app_name:app --host 0.0.0.0 --port 8000

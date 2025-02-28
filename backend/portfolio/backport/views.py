import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rest_framework.response import Response
from rest_framework.decorators import api_view

# Get the absolute path of the current script (views.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  
DETAILS_FILE = os.path.join(CURRENT_DIR, "details.json")  

# Load the model
model = SentenceTransformer("all-MiniLM-L12-v2")

# Open details.json with the correct path
with open(DETAILS_FILE, "r") as f:
    data = json.load(f)

# Load FAISS embeddings
embeddings = np.load(os.path.join(CURRENT_DIR, "portfolio_embeddings.npy")).astype('float32')

# Initialize FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def func(query):
    """Search for the most relevant response based on FAISS index."""
    query_embedding = model.encode([query]).astype('float32')
    D, I = index.search(query_embedding, 3)  # Get top 3 matches

    for idx in I[0]:  
        if "tags" in data[idx] and any(tag in query.lower() for tag in data[idx]["tags"]):
            return data[idx]["content"]

    return data[I[0][0]]["content"]  # Fallback to the first match

@api_view(['POST'])
def handle_message(request):
    """Django view to process user query and return a response."""
    text = request.data.get("text")
    if text:
        response_text = func(text)
        return Response({"message": response_text}, status=200)
    return Response({"error": "No text provided"}, status=400)

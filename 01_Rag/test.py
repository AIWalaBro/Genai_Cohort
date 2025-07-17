from euriai import EuriaiClient
from dotenv import load_dotenv
import os
load_dotenv()

euri = os.getenv("EURI")

client = EuriaiClient(
    api_key= euri,
    model="gpt-4.1-nano"  # You can also try: "gemini-2.0-flash-001", "llama-4-maverick", etc.
)

response = client.generate_completion(
    prompt="Write a short poem about artificial intelligence.",
    temperature=0.7,
    max_tokens=300
)

# print(response)

from euriai.langchain_embed import EuriaiEmbeddings

embedding_model = EuriaiEmbeddings(api_key=euri)
print(embedding_model.embed_query("What's AI?")[:5])  # Print first 5 dimensionscls
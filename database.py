import chromadb
import redis
import redis
import hashlib
import json
from kafka import KafkaProducer
from sentence_transformers import SentenceTransformer


# Initialize Shared Models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="master_sheet")


def get_db_resources():
    return embed_model, collection

def get_cache_resources():
    # 1. Redis Connection
    r = redis.Redis(host='remote', password="password", port=6379, db=0, decode_responses=True)
    
    # 2. Kafka Producer (optional)
    producer = None
    try:
        producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    except Exception as e:
        print(f"Warning: Kafka not available: {e}. Proceeding without Kafka.")
    
    # 3. Dummy placeholder for your existing embedding/collection logic
    # embed_model, collection = your_chroma_setup()
    return r, producer

def generate_hash(data: bytes or str):
    """Generates a unique key for Redis."""
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()

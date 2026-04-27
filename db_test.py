import chromadb
import redis

chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="master_sheet"
)
all_data = collection.get()
print(all_data)
print(all_data["documents"])
print(all_data["ids"])

# Deleting the Vector DB documents
all_data = collection.get()
all_ids = all_data["ids"]

if all_ids:
    collection.delete(ids=all_ids)
    print("All documents deleted from 'master_sheet'.")
else:
    print("Collection is already empty.")



r = redis.Redis(host='remote', password="", port=6379, db=0, decode_responses=True)
r.flushdb()


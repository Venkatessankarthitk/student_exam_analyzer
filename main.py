import os
import json
import time
# Suppress Hugging Face Hub warnings and disable telemetry
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HOME'] = './hf_cache'

from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import ollama

# Try importing agents, database, and sync modules
try:
    from agents import create_agentic_api
    from database import get_db_resources, get_cache_resources, generate_hash
except ImportError as e:
    print(f"Warning: Could not import optional modules: {e}")
    print("API will run in reduced mode")

app = FastAPI(title="AI Student Tutor API", version="1.0")

# Initialize agent app safely
try:
    agent_app = create_agentic_api()
except Exception as e:
    print(f"Error initializing agent: {e}")
    agent_app = None

@app.post("/analyze")
async def analyze_homework(question: UploadFile = File(...), answer: UploadFile = File(...)):
    if agent_app is None:
        raise HTTPException(status_code=503, detail="Agent not initialized. Check server logs.")
    start_time = time.time()
    q_data = await question.read()
    a_data = await answer.read()

    redis_client, kafka_prod = get_cache_resources()
    analysis_key = f"analysis:{generate_hash(q_data + a_data)}"
    print(q_data)
    cached_analysis = redis_client.get(analysis_key)
    
    if cached_analysis:
        return {
            "source": "Redis_Cache",
            "elapsed": f"{time.time() - start_time:.4f}s",
            "data": json.loads(cached_analysis)
        }

    db_context = "Formula: (a+b)^2 = a^2 + 2ab + b^2" 

    # inputs = {
    #     "q_img": q_data, "a_img": a_data,
    #     "student_text": "", "question_text": "",
    #     "teacher_remarks": "", "subject": "", 
    #     "db_context": db_context, "final_observation": ""
    # }

    result = agent_app.invoke({"q_img": q_data, "a_img": a_data})
    
    analysis_output = {
        "subject_detected": result["subject"],
        "observation": result["final_observation"],
        "teacher_notes": result.get("teacher_remarks", "")
    }

    # 1. Save to Redis (Requirement 1)
    redis_client.setex(analysis_key, 86400, json.dumps(analysis_output))
    
    # 2. Push to Kafka (For asynchronous processing/logging) if available
    if kafka_prod:
        kafka_prod.send('student_analysis_topic', analysis_output)

    return {
            "source": "Original AI",
            "elapsed": f"{time.time() - start_time:.4f}s",
            "data": analysis_output
        }

@app.post("/upload-lesson")
async def upload_lesson(file: UploadFile = File(...)):
    """
    API to upload a textbook image, describe it using VLM, 
    and store it in the RAG (ChromaDB).
    """
    try:
        # 1. Read the uploaded image
        img_bytes = await file.read()
        
        # 2. Get DB resources
        embed_model, collection = get_db_resources()

        # 3. Generate visual description using the VLM
        # This acts as the "searchable text" for the lesson
        response = ollama.generate(
            model='qwen2.5vl:3b',
            prompt="Describe this textbook lesson or example in detail. Extract all math formulas, English grammar rules, and key facts for reference.",
            images=[img_bytes]
        )
        description = response['response']

        # 4. Create a unique ID (using filename)
        file_id = file.filename

        # 5. Store in ChromaDB
        collection.add(
            documents=[description],
            embeddings=[embed_model.encode(description).tolist()],
            ids=[file_id],
            metadatas=[{"filename": file.filename, "type": "lesson_reference"}]
        )

        return {
            "status": "success",
            "message": f"Lesson '{file.filename}' uploaded and indexed.",
            "description_preview": description[:100] + "..."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "agent": "initialized" if agent_app else "not_initialized",
        "service": "AI Student Tutor API"
    }

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)



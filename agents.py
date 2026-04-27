from typing import TypedDict, Annotated, List
import ollama
import base64
from database import get_db_resources
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    q_img: bytes
    a_img: bytes
    student_text: str
    question_text: str
    teacher_remarks: str
    subject: str # 'math' or 'english'
    db_context: str
    final_observation: str

def vision_router_node(state: AgentState):
    # Encode images to base64
    q_img_b64 = base64.b64encode(state['q_img']).decode('utf-8')
    a_img_b64 = base64.b64encode(state['a_img']).decode('utf-8')
    
    prompt = """You are analyzing student homework. Ignore red ink marks completely.
    Image 1: Question | Image 2: Student's Answer
    
    Extract ONLY what is written in BLACK/BLUE INK (ignore red marks):
    1. SUBJECT: Is this Math or English?
    2. QUESTION_TEXT: Transcribe EXACTLY what the question asks, preserving all mathematical symbols (x, +, -, *, /, =, parentheses, fractions)
    3. STUDENT_ANSWER: Transcribe EXACTLY all work shown in black ink line by line, preserving all mathematical notation and equal signs
    
    IMPORTANT: For math expressions:
    - Use "/" for division
    - Include all operations exactly as written
    - Preserve line breaks to show calculation steps
    - Do NOT simplify or interpret, just transcribe
    
    Format EXACTLY as:
    SUBJECT: [Math/English]
    QUESTION: [exact question transcription]
    STUDENT_ANSWER: [exact student work line by line]
    """
    
    try:
        res = ollama.generate(
            model='qwen2.5vl:3b', 
            prompt=prompt, 
            images=[q_img_b64, a_img_b64]
        )['response']
    except Exception as e:
        return {
            "subject": "math",
            "question_text": "Error reading question",
            "student_text": "Error reading answer",
            "teacher_remarks": ""
        }
    
    # Parse the response
    try:
        subj = "math" if "math" in res.lower() else "english"
        question = res.split("QUESTION:")[1].split("STUDENT_ANSWER:")[0].strip() if "QUESTION:" in res else ""
        student = res.split("STUDENT_ANSWER:")[1].strip() if "STUDENT_ANSWER:" in res else ""
    except:
        subj = "math"
        question = res[:200]
        student = res[200:]
    
    return {
        "subject": subj,
        "question_text": question,
        "student_text": student,
        "teacher_remarks": ""
    }

def retrieval_node(state: AgentState):
    # Generate embedding for the question to search the DB
    embed_model, collection = get_db_resources()
    query_emb = embed_model.encode(state['question_text']).tolist()
    
    results = collection.query(query_embeddings=[query_emb], n_results=1)
    context = results['documents'][0][0] if results['documents'] else "No specific lesson found."
    print("===========================")
    print("retrieval")
    print(context)
    return {"db_context": context}

def math_expert_node(state: AgentState):
    prompt = f"""
    You are a Math Tutor. Analyze the student's work based on the provided Lesson Context.

    CONTEXT/FORMULA: {state['db_context']}
    QUESTION: {state['question_text']}
    STUDENT'S TRANSCRIPTION: {state['student_text']}

    INSTRUCTIONS:
    1. Identify the 'Given' values (e.g., what is x?).
    2. Calculate the CORRECT step-by-step solution based on those values.
    3. Compare your correct steps to the student's "STUDENT WROTE" text.
    4. Specifically check for:
       - Substitution errors (did they use the right x?).
       - Calculation errors (multiplication/addition).
       - Common denominator errors in fractions.

    OUTPUT FORMAT:
    CORRECT_SOLUTION: [Your step-by-step calculation]
    MISTAKE_ANALYSIS: [Identify exactly where the student deviated]
    FEEDBACK: [A friendly note for the student]
    """
    
    ans = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
    response_text = ans['message']['content']
    
    # # Parse to clean format
    # try:
    #     correct = response_text.split("CORRECT_PART:")[1].split("MISTAKE:")[0].strip() if "CORRECT_PART:" in response_text else ""
    #     mistake = response_text.split("MISTAKE:")[1].strip() if "MISTAKE:" in response_text else ""
    #     observation = f"✓ {correct}\n✗ {mistake}"
    # except:
    #     observation = response_text
    
    # return {"final_observation": observation}
    print("===========================")
    print("math_expert_node")
    print(response_text)
    return {"final_observation": response_text}


def english_expert_node(state: AgentState):
    prompt = f"""Compare Student Answer against the Lesson Context.
    LESSON: {state['db_context']}
    QUESTION: {state['question_text']}
    STUDENT WROTE: {state['student_text']}
    
    YOUR TASK (IGNORE RED INK - look only at black ink):
    1. Check if student answered the question correctly
    2. Find spelling mistakes
    3. Check grammar mistakes
    
    Output EXACTLY in this format:
    CORRECT_PART: [What the student did right]
    MISTAKE: [Specific mistake - e.g. "spelling error in 'someword'" or "didn't answer the question"]
    """
    
    ans = ollama.chat(model='qwen2.5vl:3b', messages=[{'role': 'user', 'content': prompt}])
    response_text = ans['message']['content']
    
    # Parse to clean format
    # try:
    #     correct = response_text.split("CORRECT_PART:")[1].split("MISTAKE:")[0].strip() if "CORRECT_PART:" in response_text else ""
    #     mistake = response_text.split("MISTAKE:")[1].strip() if "MISTAKE:" in response_text else ""
    #     observation = f"✓ {correct}\n✗ {mistake}"
    # except:
    #     observation = response_text
    print("english")
    print("========================================")
    print(response_text)
    return {"final_observation": response_text}


def create_agentic_api():
    workflow = StateGraph(AgentState)
    workflow.add_node("router", vision_router_node)
    workflow.add_node("retriever", retrieval_node)
    workflow.add_node("math_agent", math_expert_node)
    workflow.add_node("english_agent", english_expert_node)

    # Routing Logic
    workflow.set_entry_point("router")
    workflow.add_edge("router", "retriever")
    workflow.add_conditional_edges(
    "retriever",
    lambda x: x["subject"],
    {"math": "math_agent", "english": "english_agent"}
    )
    workflow.add_edge("math_agent", END)
    workflow.add_edge("english_agent", END)
    return workflow.compile()
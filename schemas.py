from pydantic import BaseModel
from typing import Optional

class AnalysisRequest(BaseModel):
    # Base64 strings are better for JSON APIs than raw bytes
    question_image: str 
    answer_image: str

class AnalysisResponse(BaseModel):
    tutor_observations: str
    teacher_remark: str
    subject: str
"""
Chat routes — handles text-based conversation logic.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from app.model.ai_model import text_model

router = APIRouter()

# === Schema ===
class ChatRequest(BaseModel):
    prompt: str


# === Route ===
@router.post("/chat", summary="Trao đổi hội thoại")
async def chat(request: ChatRequest):
    """
    Receive a user's prompt and return a friendly Vietnamese chatbot response.
    """
    try:
        prompt = f"""
        Bạn là một chatbot thân thiện, trả lời bằng tiếng Việt một cách tự nhiên và hữu ích.
        Bạn là chatbot của OhBầu, ứng dụng quản lý sức khoẻ thai nhi.
        Câu hỏi: {request.prompt}
        """

        response = text_model.generate_content(prompt)
        return {"message": response.text}

    except Exception as e:
        return {"message": f"Lỗi: {str(e)}"}

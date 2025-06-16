# Import thư viện Gemini AI
import google.generativeai as genai

# Import thư viện làm việc với hệ điều hành và biến môi trường
import os
from dotenv import load_dotenv

# Import FastAPI và Pydantic
from fastapi import FastAPI
from pydantic import BaseModel

# Tải biến môi trường
load_dotenv()

# Cấu hình API key cho Gemini
genai.configure(api_key=os.getenv("API_KEY"))

# Khởi tạo mô hình Gemini
model = genai.GenerativeModel("gemini-1.5-flash")  # Sử dụng phiên bản mới nhất

# Định nghĩa schema dữ liệu cho request
class ChatRequest(BaseModel):
    prompt: str

# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="Simple Chatbot API", description="Chatbot sử dụng Gemini AI", version="1.0")

# Endpoint POST /chat
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Tạo prompt với ngữ cảnh
        prompt = f"""
        Bạn là một chatbot thân thiện, trả lời bằng tiếng Việt một cách tự nhiên và hữu ích.
        Bạn là chatbot của OhBầu, ứng dụng quản lý sức khoẻ thai nhi
        Câu hỏi: {request.prompt}
        """
        # Gửi yêu cầu đến Gemini
        response = model.generate_content(prompt)
        return {"message": response.text}
    except Exception as e:
        return {"message": f"Lỗi: {str(e)}"}

# Chạy ứng dụng
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))  # Lấy PORT từ môi trường, mặc định 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
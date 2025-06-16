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
model = genai.GenerativeModel("gemini-1.5-flash")

# Định nghĩa schema dữ liệu cho request
class ChatRequest(BaseModel):
    prompt: str

# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="Simple Chatbot API", description="Chatbot sử dụng Gemini AI", version="1.0", docs_url="/docs")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Endpoint GET /
@app.get("/")
async def read_root():
    return {"message": "Welcome to OhBầu Chatbot! Visit /docs to see API documentation and try /chat with POST."}

# Endpoint POST /chat
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Tạo prompt với ngữ cảnh
        prompt = f"""
        Bạn là một chatbot thân thiện, trả lời bằng tiếng Việt một cách tự nhiên và hữu ích.
        Bạn là chatbot của OhBầu, ứng dụng quản lý sức khoẻ thai nhi.
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
    port = 8000  # Mặc định cổng
    try:
        port_str = os.environ.get("PORT", "8000").strip()
        print(f"Raw PORT value: {port_str}")
        if port_str.endswith("."):
            port_str = port_str[:-1]
        port = int(port_str)
        if not (1 <= port <= 65535):
            raise ValueError(f"Port {port} is out of valid range (1-65535)")
    except (ValueError, TypeError) as e:
        print(f"Invalid port value: {e}. Falling back to port 8000.")
        port = 8000
    print(f"Starting uvicorn on port: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
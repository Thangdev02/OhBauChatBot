# ==============================================
#  OhBầu - Baby Face Generator API
#  Version: 2.0 (Gemini 2.0 Flash + Hugging Face + ImgBB)
# ==============================================

import os
import uuid
import base64
import requests
import traceback
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import google.generativeai as genai
from PIL import Image
from pydantic import BaseModel

# ===== API Keys =====
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# ===== Configure Gemini =====
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

app = FastAPI(title="OhBầu Baby Generator", version="2.0")

class ChatRequest(BaseModel):
    prompt: str


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Trò chuyện với chatbot OhBầu.
    """
    try:
        prompt = f"""
        Bạn là OhBầu Chatbot - một trợ lý thân thiện, nói tiếng Việt tự nhiên.
        Ứng dụng OhBầu giúp mẹ bầu theo dõi sức khoẻ thai kỳ, nhận tư vấn dinh dưỡng và tâm lý.

        Người dùng hỏi: {request.prompt}
        """
        response = gemini_model.generate_content(prompt)
        return {"success": True, "message": response.text}

    except Exception as e:
        traceback.print_exc()
        return {"success": False, "message": f"Lỗi khi phản hồi: {str(e)}"}



@app.post("/generate-baby")
async def generate_baby(mother: UploadFile = File(...), father: UploadFile = File(...)):
    """
    Tạo ảnh em bé từ ảnh cha mẹ:
    1. Phân tích ảnh cha mẹ bằng Gemini 2.0 Flash
    2. Generate ảnh em bé bằng Hugging Face
    3. Upload lên ImgBB
    4. Trả về link public
    """
    try:
        # --- Lưu file tạm ---
        os.makedirs("temp", exist_ok=True)
        mother_path = f"temp/mother_{uuid.uuid4().hex}.png"
        father_path = f"temp/father_{uuid.uuid4().hex}.png"

        with open(mother_path, "wb") as f:
            f.write(await mother.read())
        with open(father_path, "wb") as f:
            f.write(await father.read())

        print("[v0] Ảnh cha mẹ đã lưu tạm")

        print("[v0] Bắt đầu phân tích ảnh cha mẹ...")
        model = genai.GenerativeModel("gemini-2.0-flash")

        analysis_prompt = """Phân tích kỹ lưỡng hai ảnh này (cha và mẹ) và mô tả chi tiết:
        - Hình dáng khuôn mặt
        - Màu mắt
        - Màu tóc
        - Miệng
        - Đặc điểm nổi bật

        Sau đó, tạo một prompt chi tiết để generate ảnh em bé (1 tuổi, dễ thương, chân thực) 
        Chỉ 1 ảnh
        Tạo ảnh cho thật đáng yêu, má phúng phính
        kết hợp nét của cả hai người này. Prompt phải bằng tiếng Anh và chi tiết."""

        analysis_response = model.generate_content(
            [
                {"mime_type": "image/png", "data": open(mother_path, "rb").read()},
                {"mime_type": "image/png", "data": open(father_path, "rb").read()},
                analysis_prompt
            ],
            stream=False
        )

        # Lấy text từ Gemini response
        analysis_text = analysis_response.text
        print(f"[v0] Phân tích từ Gemini:\n{analysis_text}")

        generation_prompt = f"""Create a realistic and adorable baby photo (1 year old) that combines features from both parents.

        Analysis: {analysis_text}

        Requirements:
        - Realistic and natural looking
        - Cute and adorable expression
        - Professional photo quality
        - Clear face visible
        - Warm lighting"""

        # --- BƯỚC 2: Generate ảnh bằng Hugging Face ---
        print("[v0] Bắt đầu generate ảnh em bé bằng Hugging Face...")

        hf_api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        payload = {
            "inputs": generation_prompt,
            "options": {"wait_for_model": True}
        }

        hf_response = requests.post(hf_api_url, headers=headers, json=payload)

        if hf_response.status_code != 200:
            print(f"[v0] Hugging Face Error: {hf_response.text}")
            raise Exception(f"Hugging Face API error: {hf_response.text}")

        # Lấy ảnh từ Hugging Face
        image_bytes = hf_response.content
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        print("[v0] Ảnh em bé đã được generate thành công")

        # --- BƯỚC 3: Upload lên ImgBB ---
        print("[v0] Bắt đầu upload lên ImgBB...")

        upload_url = "https://api.imgbb.com/1/upload"
        payload = {"key": IMGBB_API_KEY, "image": image_base64}

        imgbb_response = requests.post(upload_url, data=payload)
        imgbb_data = imgbb_response.json()

        if not imgbb_response.ok or "data" not in imgbb_data:
            print(f"[v0] ImgBB Error: {imgbb_data}")
            raise Exception(imgbb_data.get("error", {}).get("message", "Upload ImgBB thất bại"))

        image_url = imgbb_data["data"]["url"]
        print(f"[v0] Upload ImgBB thành công: {image_url}")

        # --- Xoá file tạm ---
        os.remove(mother_path)
        os.remove(father_path)
        print("[v0] Đã xoá file tạm")

        return JSONResponse({
            "success": True,
            "message": "Tạo ảnh em bé thành công!",
            "image_url": image_url,
            "analysis": analysis_text
        })

    except Exception as e:
        print("[v0] Lỗi chi tiết:")
        traceback.print_exc()

        # Xoá file tạm nếu có lỗi
        try:
            if os.path.exists(mother_path):
                os.remove(mother_path)
            if os.path.exists(father_path):
                os.remove(father_path)
        except:
            pass

        return JSONResponse(
            status_code=500,
            content={"success": False, "detail": f"Lỗi khi sinh ảnh: {str(e)}"}
        )


@app.get("/health")
async def health_check():
    """Kiểm tra API có hoạt động không"""
    return JSONResponse({"status": "OK", "version": "2.0"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

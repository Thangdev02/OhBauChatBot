# ==============================================
#  OhBầu - Baby Face Generator API
#  Version: 2.6 (Fix 422 string_too_long + Truncate Prompt + Replicate Provider 2025)
# ==============================================

import os
import uuid
import base64
import traceback
import io  # Để xử lý PIL Image
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import google.generativeai as genai
from pydantic import BaseModel
from dotenv import load_dotenv
from huggingface_hub import InferenceClient  # pip install huggingface_hub
from PIL import Image  # pip install pillow

# Load .env trước khi lấy biến môi trường
load_dotenv()

# ===== API Keys =====
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Kiểm tra key (tùy chọn, để debug)
if not all([GOOGLE_API_KEY, IMGBB_API_KEY, HUGGINGFACE_API_KEY]):
    print("⚠️  Cảnh báo: Thiếu một trong các API key! Kiểm tra file .env")

# ===== Configure Gemini (dùng 2.0-flash như mày muốn) =====
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# ===== Hugging Face Client (dùng REPLICATE provider - free + limit cao cho FLUX) =====
hf_client = InferenceClient(
    api_key=HUGGINGFACE_API_KEY,
    provider="replicate"  # FREE cho FLUX.1-schnell, limit prompt ~4000 chars (cao hơn nebius)
)

app = FastAPI(title="OhBầu Baby Generator", version="2.6")

class ChatRequest(BaseModel):
    prompt: str

# ==============================================
# 1. Chatbot bình thường (giữ nguyên 100%)
# ==============================================
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        prompt = f"""
        Bạn là OhBầu Chatbot - trợ lý thân thiện, vui vẻ, nói tiếng Việt tự nhiên.
        Ứng dụng OhBầu giúp mẹ bầu theo dõi thai kỳ, tư vấn dinh dưỡng và tâm lý.

        Người dùng hỏi: {request.prompt}
        """
        response = gemini_model.generate_content(prompt)
        return {"success": True, "message": response.text}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "message": f"Lỗi khi phản hồi: {str(e)}"}

# ==============================================
# 2. Tạo ảnh em bé (TRUNCATE PROMPT + EXTRA_BODY MAX_LENGTH)
# ==============================================
@app.post("/generate-baby")
async def generate_baby(mother: UploadFile = File(...), father: UploadFile = File(...)):
    mother_path = father_path = None
    try:
        # --- Lưu file tạm ---
        os.makedirs("temp", exist_ok=True)
        mother_path = f"temp/mother_{uuid.uuid4().hex}.jpg"
        father_path = f"temp/father_{uuid.uuid4().hex}.jpg"

        with open(mother_path, "wb") as f:
            f.write(await mother.read())
        with open(father_path, "wb") as f:
            f.write(await father.read())

        print("Ảnh cha mẹ đã lưu tạm")

        # --- Phân tích bằng Gemini ---
        print("Đang phân tích khuôn mặt cha mẹ bằng Gemini...")
        analysis_prompt = """Phân tích thật kỹ hai bức ảnh này và mô tả CHI TIẾT bằng tiếng Anh:
        - Face shape (oval, round, square, heart...)
        - Eye color & shape
        - Hair color & style
        - Nose shape
        - Lips & smile
        - Skin tone
        - Đặc điểm nổi bật của cha và mẹ

        Sau đó tạo 1 prompt cực kỳ chi tiết bằng tiếng Anh để generate ảnh em bé 1 tuổi dễ thương, 
        má phúng phính, kết hợp đặc điểm của cả cha và mẹ."""

        analysis_response = gemini_model.generate_content([
            {"mime_type": "image/jpeg", "data": open(mother_path, "rb").read()},
            {"mime_type": "image/jpeg", "data": open(father_path, "rb").read()},
            analysis_prompt
        ])
        analysis_text = analysis_response.text
        print(f"Phân tích Gemini (full):\n{analysis_text}")

        # --- FIX CHÍNH: Truncate analysis_text để tránh prompt quá dài (giữ ~800 chars, chỉ features + prompt chính) ---
        # Tìm vị trí bắt đầu prompt (sau "**Prompt for AI Image Generation**") và cắt phần giải thích dài
        if "**Prompt for AI Image Generation**" in analysis_text:
            prompt_start = analysis_text.find("Here's a highly detailed prompt")  # Bắt đầu prompt thực
            if prompt_start != -1:
                analysis_text = analysis_text[:prompt_start + 2000]  # Cắt prompt chính + một ít context
            else:
                # Fallback: Cắt thủ công sau phần features
                features_end = analysis_text.find("**Prompt for AI Image Generation of a 1-Year-Old Baby:**")
                if features_end != -1:
                    analysis_text = analysis_text[:features_end + 1500]  # Giữ features + prompt ngắn
        # Truncate tổng thể xuống 800 chars để an toàn
        analysis_text = analysis_text[:800] + " ... (combined features for baby generation)"
        print(f"Phân tích Gemini (truncated):\n{analysis_text}")

        # --- Prompt sinh ảnh bé (ngắn gọn hơn) ---
        generation_prompt = f"""
        Adorable 1 year old baby, chubby cheeks, big bright eyes, soft lighting, professional studio photo, 
        ultra realistic, cinematic, highly detailed, cute smile, warm tone.
        Combine features from both parents: {analysis_text}
        """

        # Truncate generation_prompt xuống 2000 chars để tránh 422
        if len(generation_prompt) > 2000:
            generation_prompt = generation_prompt[:2000] + " ... (detailed baby features combined)"
        print(f"Final generation prompt length: {len(generation_prompt)} chars")

        # --- Gọi Hugging Face QUA INFERENCECLIENT (REPLICATE + EXTRA_BODY MAX_LENGTH) ---
        print("Đang tạo ảnh em bé bằng AI... (có thể mất 10-30 giây)")

        # Model FREE 100% + SIÊU NHANH (FLUX.1-schnell qua replicate)
        image_pil = hf_client.text_to_image(
            prompt=generation_prompt,
            model="black-forest-labs/FLUX.1-schnell",  # FREE qua replicate
            num_inference_steps=4,     # Chỉ 4 bước để nhanh
            guidance_scale=0.0,        # Không cần cho schnell
            width=512,
            height=512,
            extra_body={"max_length": 2000}  # Enforce limit để tránh 422
        )

        # Chuyển PIL Image thành bytes
        img_buffer = io.BytesIO()
        image_pil.save(img_buffer, format='PNG')
        image_bytes = img_buffer.getvalue()

        if not image_bytes or len(image_bytes) < 100:
            raise Exception("Hugging Face trả về ảnh rỗng (thử lại sau 1 phút)")

        print("Đã tạo xong ảnh em bé!")

        # --- Upload lên ImgBB ---
        print("Đang upload lên ImgBB...")
        image_base64_str = base64.b64encode(image_bytes).decode('utf-8')
        upload_url = "https://api.imgbb.com/1/upload"
        payload = {
            "key": IMGBB_API_KEY,
            "image": image_base64_str,
            "name": "ohbau_baby_" + uuid.uuid4().hex[:8]
        }

        imgbb_res = requests.post(upload_url, data=payload, timeout=60)
        imgbb_json = imgbb_res.json()

        if not imgbb_res.ok or imgbb_json.get("success") is not True:
            raise Exception(f"ImgBB lỗi: {imgbb_json}")

        final_url = imgbb_json["data"]["url"]
        print(f"Thành công! Link ảnh: {final_url}")

        # Xóa file tạm
        for p in [mother_path, father_path]:
            if p and os.path.exists(p):
                os.remove(p)

        return JSONResponse({
            "success": True,
            "message": "Tạo ảnh em bé thành công! Bé đáng yêu lắm nè!",
            "image_url": final_url,
            "analysis": analysis_text.strip()
        })

    except Exception as e:
        print("LỖI TOÀN BỘ:")
        traceback.print_exc()

        # Dọn dẹp file tạm nếu lỗi
        for p in [mother_path, father_path]:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except:
                pass

        return JSONResponse(
            status_code=500,
            content={"success": False, "detail": f"Lỗi khi sinh ảnh: {str(e)}"}
        )

# ==============================================
# Health check
# ==============================================
@app.get("/health")
async def health_check():
    return {"status": "OK", "version": "2.6", "message": "OhBầu đang rất khỏe, sẵn sàng sinh bé phúng phính!"}

# ==============================================
# Run server
# ==============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
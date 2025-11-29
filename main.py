import os
import uuid
import base64
import io
import httpx
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageEnhance
from dotenv import load_dotenv
import google.generativeai as genai
from huggingface_hub import InferenceClient
from pydantic import BaseModel

# ===============================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not all([GOOGLE_API_KEY, IMGBB_API_KEY, HUGGINGFACE_API_KEY]):
    raise RuntimeError("Thiếu API key! Check file .env")

genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")
hf_client = InferenceClient(token=HUGGINGFACE_API_KEY)

app = FastAPI(
    title="OhBầu v6.0 - Bé Siêu Thật + Chat JSON",
    version="6.0",
    description="Gen bé đẹp như thật (SDXL free mãi mãi) + Chat hỗ trợ mẹ bầu"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===============================
async def read_and_convert_image(file: UploadFile) -> bytes:
    contents = await file.read()
    if len(contents) == 0:
        raise ValueError("File rỗng!")
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img.thumbnail((720, 720), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


class ChatRequest(BaseModel):
    prompt: str


# ===============================
# UPLOAD IMGBB SIÊU ỔN ĐỊNH (có retry + giảm size)
# ===============================
async def safe_upload_imgbb(b64_str: str) -> str:
    url = "https://api.imgbb.com/1/upload"
    payload = {
        "key": IMGBB_API_KEY,
        "image": b64_str,
        "name": "ohbau_" + uuid.uuid4().hex[:10],
        "expiration": 15552000  # 180 ngày
    }

    for _ in range(3):  # thử tối đa 3 lần
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(url, data=payload, timeout=30.0)
                data = r.json()
                if data.get("success"):
                    return data["data"]["url"]
                else:
                    print(f"ImgBB lỗi: {data.get('error', 'Unknown')}")
        except Exception as e:
            print(f"Upload retry lỗi: {e}")

    raise Exception("Upload ImgBB thất bại sau 3 lần thử")


# ===============================
# GEN BÉ – DỰA HOÀN TOÀN TRÊN V5.0 CỦA BẠN (CHẠY NGON NHẤT)
# ===============================
@app.post("/generate-baby", tags=["Baby Generator"])
async def generate_baby(mother: UploadFile = File(...), father: UploadFile = File(...)):
    try:
        mom = await read_and_convert_image(mother)
        dad = await read_and_convert_image(father)

        analysis = gemini_model.generate_content([
            {"mime_type": "image/jpeg", "data": mom},
            {"mime_type": "image/jpeg", "data": dad},
            "Analyze both parents in English. Describe face shape, eyes, nose, lips, hair, skin tone, unique traits. Then write ONE short prompt (max 280 chars) for a cute 1-year-old baby combining their best features."
        ])
        traits = analysis.text.strip()

        prompt = (
            f"professional studio portrait of a cute 1 year old Vietnamese baby, "
            f"chubby cheeks, big sparkling eyes, soft baby skin, natural blush, "
            f"tiny sweet smile, wearing white onesie, soft window light, "
            f"extremely realistic, photorealistic, sharp details, Canon EOS R5, 85mm, f1.8, "
            f"warm tone, looks like real photo, ultra detailed baby skin --ar 1:1 --v 5 --q 2"
        )

        image = hf_client.text_to_image(
            model="stabilityai/stable-diffusion-xl-base-1.0",  # FREE MÃI MÃI
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=30,
            guidance_scale=7.5,
        )

        # Hậu kỳ nhẹ
        img = ImageEnhance.Brightness(image).enhance(1.05)
        img = ImageEnhance.Contrast(img).enhance(1.03)
        final = img.resize((384, 384), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        final.save(buf, format="JPEG", quality=88, optimize=True)  # size < 500KB → ImgBB luôn nhận
        b64 = base64.b64encode(buf.getvalue()).decode()

        image_url = await safe_upload_imgbb(b64)

        return {
            "success": True,
            "message": "Bé yêu đã chào đời xinh như thiên thần luôn nè!!!",
            "image_url": image_url,
            "analysis": traits
        }

    except Exception as e:
        error_msg = str(e)
        print("LỖI CHI TIẾT:", error_msg)  # để bạn thấy thật trên terminal
        if any(x in error_msg.lower() for x in ["rate limit", "quota", "timeout"]):
            error_msg = "Hệ thống đang đông khách, thử lại sau 10 giây nha mẹ iu!"
        return JSONResponse(status_code=500, content={"success": False, "error": error_msg})


# ===============================
# CHAT – NHẬN JSON BODY { "prompt": "..." }
# ===============================
@app.post("/chat", tags=["Chat Support"])
async def chat(request: ChatRequest):
    try:
        message = request.prompt.strip()
        msg_lower = message.lower()

        # Rule bắt buộc chuyển hướng bác sĩ
        doctor_keywords = ["liên hệ", "hỗ trợ", "bác sĩ", "tư vấn", "zalo", "điện thoại", "gọi", "hotline", "hạnh",
                           "đặng hạnh", "bs hạnh"]
        if any(k in msg_lower for k in doctor_keywords):
            return {
                "success": True,
                "reply": (
                    "Dạ mẹ ơi, để được hỗ trợ chính xác và tận tâm nhất về thai kỳ, dinh dưỡng hay tâm lý,\n"
                    "mẹ liên hệ trực tiếp **Bác sĩ Đặng Hạnh** nha ạ ❤️\n\n"
                    "Zalo/Điện thoại: **038 424 8930**\n"
                    "Bác sĩ hỗ trợ 24/7 luôn ạ!"
                )
            }

        system_prompt = (
            "Bạn là OhBầu Chatbot – trợ lý dễ thương, nói tiếng Việt tự nhiên, "
            "chuyên hỗ trợ mẹ bầu về thai kỳ, dinh dưỡng, tâm lý. "
            "Trả lời ngắn gọn, ấm áp, dùng nhiều emoji đáng yêu."
        )

        response = gemini_model.generate_content([
            system_prompt,
            f"Mẹ bầu hỏi: {message}\nTrả lời thật dễ thương và ngắn gọn thôi nha:"
        ])

        return {"success": True, "reply": response.text.strip()}

    except Exception as e:
        return {"success": False, "reply": "Oops, OhBầu đang hơi mệt xíu, mẹ hỏi lại sau vài giây nha"}


# ===============================
@app.get("/health")
async def health():
    return {
        "status": "OK",
        "version": "6.0-SDXL-Chat",
        "message": "OhBầu đang khỏe mạnh, sinh bé mượt + chat ngon lành!"
    }


# ===============================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
"""
Baby image generation routes — uploads father & mother photos,
then generates a predicted baby image using AI.
"""

import base64
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.model.ai_model import image_model
from app.utils.image_tools import validate_image_bytes, save_image_bytes

router = APIRouter()


@router.post("/generate-baby", summary="Tạo ảnh em bé từ ảnh cha và mẹ")
async def generate_baby(father: UploadFile = File(...), mother: UploadFile = File(...)):
    try:
        father_bytes = await father.read()
        mother_bytes = await mother.read()

        if not validate_image_bytes(father_bytes):
            raise HTTPException(status_code=400, detail="File father không phải ảnh hợp lệ.")
        if not validate_image_bytes(mother_bytes):
            raise HTTPException(status_code=400, detail="File mother không phải ảnh hợp lệ.")

        # Save uploaded files
        upload_id = str(uuid.uuid4())
        father_path = f"uploads/{upload_id}_father.jpg"
        mother_path = f"uploads/{upload_id}_mother.jpg"
        save_image_bytes(father_bytes, father_path)
        save_image_bytes(mother_bytes, mother_path)

        # Convert to base64
        father_b64 = base64.b64encode(father_bytes).decode("utf-8")
        mother_b64 = base64.b64encode(mother_bytes).decode("utf-8")

        prompt_text = (
            "Dự đoán khuôn mặt của em bé dựa trên đặc điểm khuôn mặt của cha và mẹ. "
            "Tạo một bức ảnh chân thực, phong cách ảnh chân dung, độ phân giải vừa phải."
        )

        # Request AI generation
        response = image_model.generate_content([
            prompt_text,
            {"mime_type": "image/jpeg", "data": father_b64},
            {"mime_type": "image/jpeg", "data": mother_b64},
        ])

        baby_b64 = getattr(response, "text", None)
        if not baby_b64:
            raise RuntimeError("Không thể tìm thấy ảnh trả về từ model.")

        baby_bytes = base64.b64decode(baby_b64)
        baby_path = f"uploads/{upload_id}_baby.jpg"
        save_image_bytes(baby_bytes, baby_path)

        return JSONResponse({
            "baby_image_base64": baby_b64,
            "baby_image_path": baby_path,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi sinh ảnh: {str(e)}")

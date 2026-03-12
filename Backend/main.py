import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from simple_lama_inpainting import SimpleLama
from PIL import Image
import io
import uvicorn
import cv2
import numpy as np

app = FastAPI(title="Doctor Codiey - Pro Magic Eraser")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading LaMa AI Model (Photoshop-Grade) on CPU...")
lama = SimpleLama()
print("LaMa AI Model loaded successfully! ✨")

@app.get("/")
def home():
    return {"message": "Pro Object Remover API is running!"}

@app.post("/remove-object")
async def remove_object(image: UploadFile = File(...), mask: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        mask_bytes = await mask.read()
        
        # 1. LOAD ORIGINAL IMAGES (Keeping Full Resolution Intact)
        original_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        mask_pil = Image.open(io.BytesIO(mask_bytes)).convert("L")
        
        orig_w, orig_h = original_pil.size
        
        # Ensure mask strictly matches original size
        if mask_pil.size != original_pil.size:
            mask_pil = mask_pil.resize(original_pil.size, Image.Resampling.NEAREST)

        # Convert to OpenCV format for advanced matrix math
        original_cv = np.array(original_pil)
        mask_cv = np.array(mask_pil)
        _, binary_mask = cv2.threshold(mask_cv, 127, 255, cv2.THRESH_BINARY)

        # 2. SMART DILATION (Expands selection dynamically based on image size)
        dilation_size = max(10, int(max(orig_w, orig_h) * 0.015))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
        dilated_mask_cv = cv2.dilate(binary_mask, kernel, iterations=1)
        
        # 3. CPU OPTIMIZATION FOR AI ENGINE
        # Scale down ONLY for the AI generation step to save memory and time
        max_ai_size = 1200
        ratio = 1.0
        if max(orig_w, orig_h) > max_ai_size:
            ratio = max_ai_size / max(orig_w, orig_h)
            new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
            ai_input_img = original_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            ai_input_mask = Image.fromarray(dilated_mask_cv).resize((new_w, new_h), Image.Resampling.NEAREST)
        else:
            ai_input_img = original_pil
            ai_input_mask = Image.fromarray(dilated_mask_cv)

        print(f"Running LaMa AI Engine on optimized size: {ai_input_img.size}...")
        
        # Inpaint with LaMa
        inpainted_ai_result = lama(ai_input_img, ai_input_mask)

        # 4. PHOTOSHOP-STYLE ALPHA COMPOSITING
        # Scale the AI-generated patch back up to the ORIGINAL resolution
        if ratio != 1.0:
            inpainted_full_res = inpainted_ai_result.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
        else:
            inpainted_full_res = inpainted_ai_result

        inpainted_cv = np.array(inpainted_full_res)

        # 5. MASK FEATHERING
        # Create a soft gradient on the mask edges for a seamless, invisible blend
        feather_amount = max(5, int(dilation_size * 0.8))
        if feather_amount % 2 == 0: feather_amount += 1 # Gaussian Blur kernel must be odd
        
        soft_mask_cv = cv2.GaussianBlur(dilated_mask_cv, (feather_amount, feather_amount), 0)
        
        # Normalize mask to 0.0 - 1.0 multiplier
        alpha = soft_mask_cv.astype(float) / 255.0
        alpha = np.expand_dims(alpha, axis=2) # Broadcast shape to (H, W, 1)

        # 6. FINAL BLEND: (AI Pixel * Alpha) + (Original Pixel * (1 - Alpha))
        # This guarantees 100% untouched areas remain perfectly sharp and original!
        final_composite = (inpainted_cv * alpha) + (original_cv * (1.0 - alpha))
        final_composite = np.clip(final_composite, 0, 255).astype(np.uint8)

        final_img_pil = Image.fromarray(final_composite)
        
        # Prepare output
        img_io = io.BytesIO()
        final_img_pil.save(img_io, format="PNG")
        img_bytes = img_io.getvalue()
        
        print("✨ Pro Photoshop-level Result Generated!")
        return Response(content=img_bytes, media_type="image/png")
    
    except Exception as e:
        import traceback
        traceback.print_exc() # Prints exact line of error in terminal
        print("Backend Error:", str(e))
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

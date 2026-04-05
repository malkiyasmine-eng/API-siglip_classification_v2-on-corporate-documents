# app.py
# FastAPI application — SigLIP document classifier


# ── Section 1: Imports ────────────────────────────────────────────────────────
import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

import classifier  # our refactored notebook logic (AI model module)


# ── Section 2: Lifespan (startup / shutdown) ──────────────────────────────────
# This runs load_model() ONCE when the server starts.
# All requests after that reuse the already-loaded model and text embeddings.
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up — loading SigLIP model …")
    classifier.load_model()#This loads: SigLIP model, processor, text embeddings (VERY heavy step)
    #Without this:model would reload for every request
    print("✅ API is ready to accept requests.")
    yield
    # (shutdown logic could go here if needed)
    print("🛑 Shutting down.")


# ── Section 3: FastAPI app instance ──────────────────────────────────────────
app = FastAPI(
    title="SigLIP Document Classifier",
    description=(
        "Classifies corporate document images using google/siglip-base-patch16-256-multilingual.\n\n"
        "Supported categories: NIF_certificate, NIS_certificate, certificat_existence, "
        "tax_declaration_form, residence_certificate, legal_contract, balance_sheet, "
        "RC_front, RC_inside_activities, RC_inside_2, "
        "driving_license_front, driving_license_back, driving_license_frontback."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
#This creates the API server with metadata:
# title
#  description 
# version 
# lifespan (startup logic)
#  This shows in Swagger UI: http://localhost:8000/docs


# ── Section 4: POST /classify ─────────────────────────────────────────────────
@app.post(
    "/classify",
    summary="Classify a document image",
    response_description="Predicted label, confidence score, and top-3 predictions",
)#the core API
async def classify(
    file: UploadFile = File(..., description="Image file to classify (JPEG, PNG, WEBP, BMP, TIFF)"),
    top_k: int = 3, #how many predictions to return
):
    """
    Upload a document image and receive:
    - **label**: the predicted document category
    - **confidence**: probability score between 0 and 1
    - **top3**: list of top-3 predictions with their confidence scores
    """

    # ── 4a: Validate MIME type (fast pre-check before reading bytes) ──────────
    ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"} #to reject unknown formats
    if file.content_type and file.content_type not in ALLOWED_MIME:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Please upload a JPEG, PNG, WEBP, BMP, or TIFF image.",
        )

    # ── 4b: Read bytes ────────────────────────────────────────────────────────
    #converts uploaded file into raw bytes
    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read uploaded file.")

    # ── 4c: Decode bytes → PIL Image (catches corrupt / non-image files) ──────
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=422,
            detail="The uploaded file could not be decoded as an image. "
                   "Make sure it is a valid image file (not a PDF, text file, etc.).",
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image decoding error: {str(e)}")
    # ensures image is valid
    # forces RGB format

    # ── 4d: Run inference ─────────────────────────────────────────────────────
    try:
        result = classifier.classify_single_image(pil_image, top_k=top_k) #This calls  SigLIP model
    except RuntimeError as e:
        # Model not loaded — should not happen in normal flow
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # ── 4e: Return JSON response ──────────────────────────────────────────────
    return JSONResponse(
        content={
            "label":      result["label"],
            "confidence": result["confidence"],
            "top3":       result["top3"],
        }
    )


# ── Section 5: GET /health ────────────────────────────────────────────────────
@app.get("/health", summary="Health check")
async def health():
    """Returns 200 OK if the API is running and the model is loaded."""
    model_ready = classifier.model is not None
    return JSONResponse(
        content={
            "status":      "ok" if model_ready else "model_not_loaded",
            "model":       classifier.MODEL_ID,
            "device":      classifier.DEVICE,
            "num_classes": len(classifier.CLASS_NAMES),
        }
    )


# ── Section 6: Run directly with `python app.py` (optional) ──────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

# SigLIP Document Classifier API

A FastAPI-based REST API for classifying Algerian corporate document images using
[google/siglip-base-patch16-256-multilingual](https://huggingface.co/google/siglip-base-patch16-256-multilingual).

---

## How it works

SigLIP is a vision-language model. Instead of training a traditional classifier,
we describe each document category in natural language (prompts), encode both the
image and the text into the same embedding space, and pick the label whosedescription is most similar to the image.

```
Image → SigLIP image encoder ──┐
                                ├──► cosine similarity ──► predicted label
Prompts → SigLIP text encoder ─┘
```

Text embeddings are pre-computed once at startup and reused for every request,so inference is fast.

---

## Supported document categories

| Label                       | Description                                          |
| --------------------------- | ---------------------------------------------------- |
| `NIF_certificate`           | Algerian DGI tax identification certificate          |
| `NIS_certificate`           | Algerian ONS statistical identification notice       |
| `certificat_existence`      | DGI existence certificate (Série C)                  |
| `tax_declaration_form`      | Tax declaration form (Série G 8)                     |
| `residence_certificate`     | Municipal residence certificate                      |
| `legal_contract`            | Notarial legal contract (Statuts)                    |
| `balance_sheet`             | French-language financial statement (Liasse Fiscale) |
| `RC_front`                  | CNRC commercial register cover page                  |
| `RC_inside_activities`      | CNRC activity codes page                             |
| `RC_inside_2`               | CNRC legal/penalties page                            |
| `driving_license_front`     | Algerian driving license — front side                |
| `driving_license_back`      | Algerian driving license — back side                 |
| `driving_license_frontback` | Algerian driving license — both sides stacked        |

---

## Project structure

```
my_api/
├── app.py            # FastAPI app — endpoints, startup, error handling
├── classifier.py     # Inference logic — model loading, prompts, classification
├── requirements.txt  # Python dependencies
├── test_api.py       # Python test script
└── README.md
```

---

## Installation

### 1. Clone or copy the project folder

```bash
cd my_api
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU note:** The `requirements.txt` installs the CPU version of PyTorch by default.
> If you have a CUDA GPU, install PyTorch separately first:
>
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```
>
> The API will automatically use the GPU if available.

---

## Running the API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

On first startup the model downloads from Hugging Face (~400 MB) and text embeddings are pre-computed. This takes 1–3 minutes once. Subsequent startups are faster if the model is cached locally.

Expected terminal output:

```
🚀 Starting up — loading SigLIP model …
⚡ Device: cuda
🔄 Loading model: google/siglip-base-patch16-256-multilingual …
✅ Model loaded — pre-computing text embeddings …
✅ Text embeddings ready — 13 classes
✅ API is ready to accept requests.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Development mode (auto-reload on code changes)

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## API endpoints

### `POST /classify`

Classify a document image.

**Request**

| Field   | Type              | Required | Description                                      |
| ------- | ----------------- | -------- | ------------------------------------------------ |
| `file`  | image file        | ✅       | JPEG, PNG, WEBP, BMP, or TIFF                    |
| `top_k` | int (query param) | ❌       | Number of top predictions to return (default: 3) |

**Response**

```json
{
  "label": "NIF_certificate",
  "confidence": 0.8731,
  "top3": [
    { "label": "NIF_certificate", "confidence": 0.8731 },
    { "label": "certificat_existence", "confidence": 0.0612 },
    { "label": "tax_declaration_form", "confidence": 0.0341 }
  ]
}
```

| Field        | Type        | Description                                    |
| ------------ | ----------- | ---------------------------------------------- |
| `label`      | string      | Predicted document category                    |
| `confidence` | float (0–1) | Probability of the top prediction              |
| `top3`       | list        | Top-k predictions with their confidence scores |

**Error responses**

| Code  | Meaning                                          |
| ----- | ------------------------------------------------ |
| `415` | Unsupported file type (not an image)             |
| `422` | File is corrupt or cannot be decoded as an image |
| `500` | Internal inference error                         |
| `503` | Model not loaded yet                             |

---

### `GET /health`

Check that the API is running and the model is loaded.

**Response**

```json
{
  "status": "ok",
  "model": "google/siglip-base-patch16-256-multilingual",
  "device": "cuda",
  "num_classes": 13
}
```

---

## Testing

### Option 1 — Swagger UI (browser)

Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.

- Click `POST /classify` → **Try it out**
- Upload any document image
- Click **Execute**
- Read the JSON response directly in the browser

### Option 2 — Python test script

```bash
python test_api.py path/to/your/document.jpg
```

Example output:

```
──────────────────────────────────────────────────
1️⃣  GET /health
   Status code : 200
   Response    : {'status': 'ok', 'model': '...', 'device': 'cuda', 'num_classes': 13}
   ✅ Health check passed.

──────────────────────────────────────────────────
2️⃣  POST /classify  ←  document.jpg
   Status code : 200
   🏆 Label      : NIF_certificate
   📊 Confidence : 0.8731  (87.3%)
   📋 Top-3 predictions:
      1. NIF_certificate              0.8731  ██████████████████████████
      2. certificat_existence         0.0612  █
      3. tax_declaration_form         0.0341
```

### Option 3 — curl

```bash
curl -X POST "http://localhost:8000/classify" \
     -H "accept: application/json" \
     -F "file=@/path/to/document.jpg"
```

### Option 4 — Python requests (inline)

```python
import requests

with open("document.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/classify",
        files={"file": ("document.jpg", f, "image/jpeg")},
    )

print(response.json())
# {'label': 'NIF_certificate', 'confidence': 0.8731, 'top3': [...]}
```

---

## Performance

| Setting    | Typical inference time |
| ---------- | ---------------------- |
| CPU        | 500–1500 ms / image    |
| GPU (CUDA) | 50–150 ms / image      |

Text embeddings are pre-computed once at startup. Only the image encoding runs
per request, which is why inference is fast even on CPU.

---

## Troubleshooting

**Model download fails**

> Make sure you have internet access on first run. The model (~400 MB) is cached
> in `~/.cache/huggingface/hub` after the first download.

**`python-multipart` not installed error**

> FastAPI requires this for file uploads. Run: `pip install python-multipart`

**Out of memory on CPU**

> SigLIP base uses ~400 MB of RAM for the model. Close other applications if needed.

**422 error on a valid image**

> Make sure the file is not a PDF. SigLIP works on raster images only (JPEG, PNG, etc.).
> Convert PDF pages to images first using tools like `pdf2image` or Adobe Acrobat.

---

## Dependencies

| Package            | Version | Purpose                  |
| ------------------ | ------- | ------------------------ |
| `fastapi`          | 0.111.0 | Web framework            |
| `uvicorn`          | 0.30.1  | ASGI server              |
| `python-multipart` | 0.0.9   | File upload support      |
| `Pillow`           | 10.3.0  | Image decoding           |
| `torch`            | 2.3.0   | Deep learning backend    |
| `transformers`     | 4.44.0  | SigLIP model & processor |
| `accelerate`       | 0.33.0  | Optimized model loading  |
| `numpy`            | 1.26.4  | Numerical operations     |

---

## Model information

| Property            | Value                                                                                  |
| ------------------- | -------------------------------------------------------------------------------------- |
| Model               | `google/siglip-base-patch16-256-multilingual`                                          |
| Type                | Vision-Language (image + text → similarity)                                            |
| Image resolution    | 256 × 256 px (patch size 16)                                                           |
| Embedding dimension | 768                                                                                    |
| Languages           | Multilingual (supports Arabic, French, English)                                        |
| Source              | [Hugging Face Hub](https://huggingface.co/google/siglip-base-patch16-256-multilingual) |

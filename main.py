import io
import uuid
import asyncio
import urllib.request
import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, Response

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

COLAB_BASE_URL = "https://blockish-fran-contently.ngrok-free.dev"

# In-memory store: { session_id: { "image_url": str, "age": int, ... } }
image_store: dict = {}

# ── CLIP model (loaded once at startup) ───────────────────────────────────────
print("Loading CLIP model...")
_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
_clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
_clip_model.eval()
print("CLIP model ready.")

age_prompt_groups = [
    {
        "label": "baby",
        "center": 2,
        "prompts": [
            "a photo of a baby or toddler",
            "a photo of an infant with chubby cheeks",
            "a portrait of a very young child under 3 years old",
        ]
    },
    {
        "label": "child",
        "center": 8,
        "prompts": [
            "a photo of a young child",
            "a portrait of a kid in elementary school",
            "a photo of a child aged 4 to 12",
        ]
    },
    {
        "label": "teenager",
        "center": 16,
        "prompts": [
            "a photo of a teenager",
            "a portrait of a high school student",
            "a photo of an adolescent with youthful features",
        ]
    },
    {
        "label": "young adult",
        "center": 27,
        "prompts": [
            "a photo of a young adult in their twenties",
            "a portrait of a college-aged person",
            "a photo of someone in their mid-twenties with smooth skin",
        ]
    },
    {
        "label": "middle-aged",
        "center": 45,
        "prompts": [
            "a photo of a middle-aged person",
            "a portrait of someone in their forties with slight wrinkles",
            "a photo of a mature adult with some gray hair",
        ]
    },
    {
        "label": "senior",
        "center": 68,
        "prompts": [
            "a photo of an elderly person",
            "a portrait of an old person with white hair and deep wrinkles",
            "a photo of a senior citizen over sixty",
        ]
    },
]

def predict_demographics(image_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # --- Gender ---
    gender_texts = [
        "a photo of a male face",
        "a photo of a female face",
    ]
    inputs = _clip_processor(text=gender_texts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = _clip_model(**inputs)
    gender_sims = torch.nn.functional.cosine_similarity(
        outputs.image_embeds.expand(len(gender_texts), -1),
        outputs.text_embeds
    )
    gender = "male" if gender_sims.argmax().item() == 0 else "female"

    # --- Age (ensemble per group) ---
    group_scores = []
    for group in age_prompt_groups:
        inputs = _clip_processor(text=group["prompts"], images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = _clip_model(**inputs)
        # 그룹 내 프롬프트 평균 similarity
        sims = torch.nn.functional.cosine_similarity(
            outputs.image_embeds.expand(len(group["prompts"]), -1),
            outputs.text_embeds
        )
        group_scores.append(sims.mean().item())

    best_idx = group_scores.index(max(group_scores))
    current_age = age_prompt_groups[best_idx]["center"]

    print(gender, current_age)

    return {"gender": gender, "current_age": current_age}




# ── Colab helpers ──────────────────────────────────────────────────────────────

def _post_age_to_colab(
    url: str, filename: str, content_type: str, data: bytes,
    age: int, gender: str, current_age: int
) -> dict:
    """Aging: sends file + age + gender + current_age to Colab"""
    boundary = "----FormBoundary" + uuid.uuid4().hex

    def field(name, value):
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        ).encode()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + data + b"\r\n"

    body += field("age", age)
    body += field("gender", gender)
    body += field("current_age", current_age)
    body += f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _post_file_to_colab(url: str, filename: str, content_type: str, data: bytes) -> dict:
    """3D: sends file only to Colab"""
    boundary = "----FormBoundary" + uuid.uuid4().hex

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _get_from_colab(url: str) -> bytes:
    """Fetches raw bytes from Colab"""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


# ── DEV ONLY ───────────────────────────────────────────────────────────────────
# Test the 3D viewer locally without a running Colab server.
# PLY file must be placed in static/uploads/gaussians.ply
# Visit: http://localhost:8000/test/viewer

@app.get("/test/viewer")
def test_viewer(request: Request):
    return templates.TemplateResponse(
        "pages/viewer.html",
        {
            "request": request,
            "session_id": "test",
            "image_url": "",
            "model_url": "/static/uploads/gaussians.ply",
            "age": 0,
        }
    )


# ── PAGE ROUTES ────────────────────────────────────────────────────────────────

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("pages/home.html", {"request": request})


@app.get("/create")
def create(request: Request, session_id: str = ""):
    """Step 1 — photo upload + age input. Pre-fills previous data if session_id is provided."""
    entry = image_store.get(session_id)
    return templates.TemplateResponse(
        "pages/create.html",
        {
            "request": request,
            "session_id": session_id,
            "prev_age": entry["age"] if entry else None,
            "prev_image_url": f"/api/images/{session_id}" if entry else None,
        }
    )


@app.get("/find")
def find(request: Request):
    """Missing persons gallery — shows all registered missing persons with aged photos."""
    return templates.TemplateResponse("pages/find.html", {"request": request})


@app.get("/find/add")
def find_add(request: Request):
    """Add a missing person — upload a past photo to generate an aged version."""
    return templates.TemplateResponse("pages/find_add.html", {"request": request})


@app.get("/find/{person_id}")
def person_detail(request: Request, person_id: str):
    """Missing person detail page — before/after slider, physical info, outfit, report form."""
    return templates.TemplateResponse("pages/person_detail.html", {"request": request, "person_id": person_id})


@app.get("/preview")
def preview(request: Request, session_id: str = ""):
    """Step 2 — review aging result and trigger 3D build"""
    entry = image_store.get(session_id)
    return templates.TemplateResponse(
        "pages/preview.html",
        {
            "request": request,
            "session_id": session_id,
            "image_url": f"/api/images/{session_id}" if entry else "",
            "age": entry["age"] if entry else 0,
        }
    )


@app.get("/viewer")
def viewer(request: Request, session_id: str = ""):
    """Step 3 — 3D model viewer"""
    entry = image_store.get(session_id)
    return templates.TemplateResponse(
        "pages/viewer.html",
        {
            "request": request,
            "session_id": session_id,
            "image_url": f"/api/images/{session_id}" if entry else "",
            "model_url": f"/api/models/{session_id}" if entry and entry.get("model_url") else "",
            "age": entry["age"] if entry else 0,
        }
    )


# ── API ROUTES ─────────────────────────────────────────────────────────────────

@app.post("/api/age")
async def api_age(
    photo: UploadFile = File(...),
    age: int = Form(...),
):
    """
    Preprocessing with CLIP → forward to Colab aging model → redirect to /preview.
    CLIP predicts gender and current age from the uploaded photo.
    """
    contents = await photo.read()
    loop = asyncio.get_event_loop()

    # Step 1: predict demographics with CLIP (runs in thread pool)
    demographics = await loop.run_in_executor(None, predict_demographics, contents)

    # Step 2: forward image + age + CLIP predictions to Colab
    result = await loop.run_in_executor(
        None,
        _post_age_to_colab,
        f"{COLAB_BASE_URL}/api/age",
        photo.filename,
        photo.content_type or "image/jpeg",
        contents,
        age,
        demographics["gender"],
        demographics["current_age"],
    )

    image_id = result["image_id"]
    session_id = str(uuid.uuid4())
    image_store[session_id] = {
        "image_url": f"{COLAB_BASE_URL}/api/images/{image_id}",
        "age": age,
        "gender": demographics["gender"],
        "current_age": demographics["current_age"],
    }

    return RedirectResponse(url=f"/preview?session_id={session_id}", status_code=303)


@app.get("/api/images/{session_id}")
async def api_get_image(session_id: str):
    """Proxies the aged image from Colab to the browser"""
    entry = image_store.get(session_id)
    if not entry:
        return Response(status_code=404)

    loop = asyncio.get_event_loop()
    image_bytes = await loop.run_in_executor(None, _get_from_colab, entry["image_url"])

    return Response(content=image_bytes, media_type="image/jpeg")


@app.get("/api/models/{session_id}")
async def api_get_model(session_id: str):
    """Proxies the 3D model file from Colab to the browser"""
    entry = image_store.get(session_id)
    if not entry or not entry.get("model_url"):
        return Response(status_code=404)

    loop = asyncio.get_event_loop()
    model_bytes = await loop.run_in_executor(None, _get_from_colab, entry["model_url"])

    return Response(content=model_bytes, media_type="application/octet-stream")


@app.post("/api/build-3d")
async def api_build_3d(session_id: str = Form(...)):
    """Sends the aged image to the Colab 3D generation endpoint"""
    entry = image_store.get(session_id)
    if not entry:
        return Response(status_code=404)

    loop = asyncio.get_event_loop()

    image_bytes = await loop.run_in_executor(None, _get_from_colab, entry["image_url"])

    result = await loop.run_in_executor(
        None,
        _post_file_to_colab,
        f"{COLAB_BASE_URL}/api/build-3d",
        f"{session_id}.jpg",
        "image/jpeg",
        image_bytes,
    )

    # image_store[session_id]["model_url"] = f"{COLAB_BASE_URL}/api/models/{result['model_id']}"
    image_store[session_id]["model_url"] = f"{COLAB_BASE_URL}/api/models/dummy"
    return RedirectResponse(url=f"/viewer?session_id={session_id}", status_code=303)

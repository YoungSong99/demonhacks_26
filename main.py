import uuid
import asyncio
import urllib.request
import json

from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, Response

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

COLAB_BASE_URL = "https://blockish-fran-contently.ngrok-free.dev"
image_store: dict = {}


def _post_age_to_colab(url: str, filename: str, content_type: str, data: bytes, age: int) -> dict:
    """Aging: sends file + age to Colab"""
    boundary = "----FormBoundary" + uuid.uuid4().hex

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + data + b"\r\n"

    body += (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="age"\r\n\r\n'
        f"{age}\r\n"
    ).encode()

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


# ── PAGE  ──────────────────────────────────────────────────────────────

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


# ── API  ─────────────────────────────────────────────────────────────────

@app.post("/api/age")
async def api_age(
    photo: UploadFile = File(...),
    age: int = Form(...),
):
    """Receives photo + age, forwards to Colab aging model, redirects to /preview"""
    contents = await photo.read()
    loop = asyncio.get_event_loop()

    result = await loop.run_in_executor(
        None,
        _post_age_to_colab,
        f"{COLAB_BASE_URL}/api/age",
        photo.filename,
        photo.content_type or "image/jpeg",
        contents,
        age,
    )

    image_id = result["image_id"]
    session_id = str(uuid.uuid4())
    image_store[session_id] = {
        "image_url": f"{COLAB_BASE_URL}/api/images/{image_id}",
        "age": age,
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

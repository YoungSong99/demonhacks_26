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

COLAB_BASE_URL = "https://a9a4-146-148-108-227.ngrok-free.app"

# 메모리 저장소: { session_id: { "image_url": str, "age": int } }
image_store: dict = {}


def _post_age_to_colab(url: str, filename: str, content_type: str, data: bytes, age: int) -> dict:
    """aging: file + age를 Colab으로 전송"""
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
    """3D: file만 Colab으로 전송"""
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
    """Colab에서 바이트 가져오기"""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


# ── 페이지 라우트 ──────────────────────────────────────────────────────────────

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("pages/home.html", {"request": request})


@app.get("/create")
def create(request: Request, session_id: str = ""):
    """Step 1 — 사진 업로드 + 나이 입력. session_id가 있으면 이전 데이터 pre-fill"""
    entry = image_store.get(session_id)
    return templates.TemplateResponse(
        "pages/aging.html",
        {
            "request": request,
            "session_id": session_id,
            "prev_age": entry["age"] if entry else None,
            "prev_image_url": f"/api/images/{session_id}" if entry else None,
        }
    )


@app.get("/preview")
def preview(request: Request, session_id: str = ""):
    """Step 2 — aging 결과 확인 + 3D 빌드"""
    entry = image_store.get(session_id)
    return templates.TemplateResponse(
        "pages/three_d.html",
        {
            "request": request,
            "session_id": session_id,
            "image_url": f"/api/images/{session_id}" if entry else "",
            "age": entry["age"] if entry else 0,
        }
    )


# ── API 라우트 ─────────────────────────────────────────────────────────────────

@app.post("/api/age")
async def api_age(
    photo: UploadFile = File(...),
    age: int = Form(...),
):
    """사진 + 나이 → Colab aging 처리 → /preview 리다이렉트"""
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
    """aging 결과 이미지를 Colab에서 프록시"""
    entry = image_store.get(session_id)
    if not entry:
        return Response(status_code=404)

    loop = asyncio.get_event_loop()
    image_bytes = await loop.run_in_executor(None, _get_from_colab, entry["image_url"])

    return Response(content=image_bytes, media_type="image/jpeg")


@app.post("/api/build-3d")
async def api_build_3d(session_id: str = Form(...)):
    """aging 결과 이미지를 Colab 3D 엔드포인트로 전송"""
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

    image_store[session_id]["model_url"] = f"{COLAB_BASE_URL}/api/models/{result['model_id']}"
    return RedirectResponse(url=f"/preview?session_id={session_id}", status_code=303)

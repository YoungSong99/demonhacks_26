import uuid
import asyncio
from pathlib import Path
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import RedirectResponse, Response
from store import image_store
from config import COLAB_BASE_URL_AGE, COLAB_BASE_URL_3D
from services.clip_service import predict_demographics
from services.colab_service import _post_age_to_colab, _post_3d_to_colab, _get_from_colab

# Project root (parent of routers/)
PROJECT_ROOT = Path(__file__).parent.parent


def _read_local(image_url: str) -> bytes:
    """Converts a local: prefix URL to an absolute path and reads the file."""
    relative = image_url[len("local:"):]
    full_path = PROJECT_ROOT / relative
    with open(full_path, "rb") as f:
        return f.read()

router = APIRouter(prefix="/api")


@router.post("/age")
async def api_age(
    photo: UploadFile = File(...),
    age: int = Form(...),
):
    """CLIP preprocessing → Colab aging → redirect to /preview."""
    contents = await photo.read()
    loop = asyncio.get_event_loop()

    # Step 1: predict demographics with CLIP
    demographics = await loop.run_in_executor(None, predict_demographics, contents)

    # Step 2: send to Colab aging model
    result = await loop.run_in_executor(
        None,
        _post_age_to_colab,
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
        "image_url": f"{COLAB_BASE_URL_AGE}/api/images/{image_id}",
        "age": age,
        "gender": demographics["gender"],
        "current_age": demographics["current_age"],
    }

    return RedirectResponse(url=f"/preview?session_id={session_id}", status_code=303)


@router.get("/images/{session_id}")
async def api_get_image(session_id: str):
    """Proxies the aged image from Colab (or local file in dev mode)."""
    entry = image_store.get(session_id)
    if not entry:
        return Response(status_code=404)

    image_url = entry["image_url"]

    if image_url.startswith("local:"):
        return Response(content=_read_local(image_url), media_type="image/jpeg")

    loop = asyncio.get_event_loop()
    image_bytes = await loop.run_in_executor(None, _get_from_colab, image_url)
    return Response(content=image_bytes, media_type="image/jpeg")


@router.get("/models/{session_id}")
async def api_get_model(session_id: str):
    """Proxies the 3D model (PLY) from Colab."""
    entry = image_store.get(session_id)
    if not entry or not entry.get("model_url"):
        return Response(status_code=404)

    loop = asyncio.get_event_loop()
    model_bytes = await loop.run_in_executor(None, _get_from_colab, entry["model_url"])
    return Response(content=model_bytes, media_type="application/octet-stream")


@router.post("/build-3d")
async def api_build_3d(session_id: str = Form(...)):
    """Sends the aged image to Colab 3D generation endpoint."""
    entry = image_store.get(session_id)
    if not entry:
        return Response(status_code=404)

    loop = asyncio.get_event_loop()
    image_url = entry["image_url"]

    # DEV: local file prefix → read directly without Colab request
    if image_url.startswith("local:"):
        image_bytes = _read_local(image_url)
    else:
        image_bytes = await loop.run_in_executor(None, _get_from_colab, image_url)

    # Send to 3D Colab
    result = await loop.run_in_executor(
        None,
        _post_3d_to_colab,
        f"{session_id}.jpg",
        "image/jpeg",
        image_bytes,
    )

    image_store[session_id]["model_url"] = f"{COLAB_BASE_URL_3D}/api/models/{result['model_id']}"
    return RedirectResponse(url=f"/viewer?session_id={session_id}", status_code=303)

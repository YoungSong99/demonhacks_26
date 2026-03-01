import uuid
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from store import image_store

router = APIRouter()
templates = Jinja2Templates(directory="templates")


# ── Dev routes ─────────────────────────────────────────────────────────────────

@router.get("/test/preview")
def test_preview(request: Request):
    """DEV: Skip aging, go straight to preview with a dummy image → test Build in 3D."""
    session_id = "test-" + uuid.uuid4().hex[:8]
    image_store[session_id] = {
        "image_url": "local:static/uploads/test_face.jpg",
        "age": 60,
        "gender": "male",
        "current_age": 30,
    }
    return RedirectResponse(url=f"/preview?session_id={session_id}", status_code=303)


@router.get("/test/viewer")
def test_viewer(request: Request):
    """DEV: 3D viewer with local PLY file."""
    return templates.TemplateResponse(
        "pages/viewer.html",
        {
            "request": request,
            "session_id": "test",
            "image_url": "",
            "model_url": "/static/uploads/gaussians.ply",
            "age": 0,
        },
    )


# ── Page routes ────────────────────────────────────────────────────────────────

@router.get("/")
def home(request: Request):
    return templates.TemplateResponse("pages/home.html", {"request": request})


@router.get("/create")
def create(request: Request, session_id: str = ""):
    entry = image_store.get(session_id)
    return templates.TemplateResponse(
        "pages/create.html",
        {
            "request": request,
            "session_id": session_id,
            "prev_age": entry["age"] if entry else None,
            "prev_image_url": f"/api/images/{session_id}" if entry else None,
        },
    )


@router.get("/find")
def find(request: Request):
    return templates.TemplateResponse("pages/find.html", {"request": request})


@router.get("/find/add")
def find_add(request: Request):
    return templates.TemplateResponse("pages/find_add.html", {"request": request})


@router.get("/find/{person_id}")
def person_detail(request: Request, person_id: str):
    return templates.TemplateResponse(
        "pages/person_detail.html",
        {"request": request, "person_id": person_id},
    )


@router.get("/preview")
def preview(request: Request, session_id: str = ""):
    entry = image_store.get(session_id)
    return templates.TemplateResponse(
        "pages/preview.html",
        {
            "request": request,
            "session_id": session_id,
            "image_url": f"/api/images/{session_id}" if entry else "",
            "age": entry["age"] if entry else 0,
        },
    )


@router.get("/viewer")
def viewer(request: Request, session_id: str = ""):
    entry = image_store.get(session_id)
    return templates.TemplateResponse(
        "pages/viewer.html",
        {
            "request": request,
            "session_id": session_id,
            "image_url": f"/api/images/{session_id}" if entry else "",
            "model_url": f"/api/models/{session_id}" if entry and entry.get("model_url") else "",
            "age": entry["age"] if entry else 0,
        },
    )

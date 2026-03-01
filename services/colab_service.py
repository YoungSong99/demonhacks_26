import uuid
import json
import urllib.request
from config import COLAB_BASE_URL_AGE, COLAB_BASE_URL_3D


def _post_age_to_colab(
    filename: str,
    content_type: str,
    data: bytes,
    age: int,
    gender: str,
    current_age: int,
) -> dict:
    """Aging: sends file + demographics to Colab aging endpoint."""
    url = f"{COLAB_BASE_URL_AGE}/api/age"
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
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def _post_3d_to_colab(filename: str, content_type: str, data: bytes) -> dict:
    """3D: sends aged image to Colab 3D generation endpoint."""
    url = f"{COLAB_BASE_URL_3D}/api/build-3d"
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
    with urllib.request.urlopen(req, timeout=500) as resp:
        return json.loads(resp.read())


def _get_from_colab(url: str) -> bytes:
    """Fetches raw bytes from any Colab URL."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=500) as resp:
        return resp.read()

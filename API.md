# Your Tomorrow — API Reference

## Overview

```
Browser (HTML Pages)             FastAPI Frontend Server          Colab ML Server
────────────────────             ───────────────────────          ───────────────
GET  /                      →    home.html
GET  /create                →    aging.html (Step 1)
POST /api/age               →    ──────────────────────────→      POST /aging
GET  /api/images/{id}       →    ──────────────────────────→      GET  /aging/{id}
GET  /preview               →    three_d.html (Step 2)
POST /api/build-3d          →    ──────────────────────────→      POST /3d
```

---

## Page Routes (user-facing)

### `GET /`
- **Role** : Home screen
- **Template** : `pages/home.html`
- **Entry points** : "Start With a Photo" and "See Their Face Again" buttons

---

### `GET /create`
- **Role** : Step 1 — photo upload and target age input form
- **Template** : `pages/aging.html`
- **Query parameters**

  | Parameter | Required | Description |
  |-----------|----------|-------------|
  | `session_id` | Optional | Previous session ID. Pre-fills age and photo preview if provided |

- **On submit** : `POST /api/age`

---

### `GET /preview`
- **Role** : Step 2 — review aging result and trigger 3D build
- **Template** : `pages/three_d.html`
- **Query parameters**

  | Parameter | Required | Description |
  |-----------|----------|-------------|
  | `session_id` | Required | Session ID from the previous step |

- **Buttons**
  - Prev → `GET /create?session_id={id}`
  - Build in 3D → `POST /api/build-3d`

---

## API Routes (server-side processing)

### `POST /api/age`
- **Role** : Receives photo and target age, runs CLIP-based demographic prediction, forwards all fields to Colab aging model, saves result to memory, then redirects
- **Input** : `multipart/form-data`

  | Field | Type | Description |
  |-------|------|-------------|
  | `photo` | File | Face image (JPEG / PNG / WEBP) |
  | `age` | int (1–100) | Target age to age the face to |

- **CLIP Preprocessing** (runs server-side before forwarding to Colab)

  The server uses `openai/clip-vit-base-patch32` to automatically predict two demographic fields from the uploaded photo. These are added to the Colab request and do **not** need to be submitted by the user.

  | Predicted Field | Type | Values | Method |
  |----------------|------|--------|--------|
  | `gender` | string | `"male"` / `"female"` | Cosine similarity between image embedding and gender text prompts |
  | `current_age` | int | `2`, `8`, `16`, `27`, `45`, `68` | Ensemble of 3 prompts per age group; group with highest mean similarity wins |

  **Age groups and center values:**

  | Label | Center | Example prompts (3 per group, averaged) |
  |-------|--------|----------------------------------------|
  | baby | 2 | "a photo of a baby or toddler", "a photo of an infant with chubby cheeks", … |
  | child | 8 | "a photo of a young child", "a portrait of a kid in elementary school", … |
  | teenager | 16 | "a photo of a teenager", "a portrait of a high school student", … |
  | young adult | 27 | "a photo of a young adult in their twenties", "a portrait of a college-aged person", … |
  | middle-aged | 45 | "a photo of a middle-aged person", "a portrait of someone in their forties with slight wrinkles", … |
  | senior | 68 | "a photo of an elderly person", "a portrait of an old person with white hair and deep wrinkles", … |

- **Fields forwarded to Colab** : `photo`, `age`, `gender` (predicted), `current_age` (predicted)

- **Flow**
  ```
  Receive photo + age
    → Run CLIP → predict gender + current_age
    → Forward to Colab POST /aging  (photo, age, gender, current_age)
    → Receive { image_id }
    → Save image_store[session_id] = { image_url, age, gender, current_age }
    → 303 Redirect → GET /preview?session_id={id}
  ```
- **Colab endpoint** : `POST {COLAB_BASE_URL}/aging`

---

### `GET /api/images/{session_id}`
- **Role** : Proxies the aged image from Colab and returns it to the browser
- **Path parameters**

  | Parameter | Description |
  |-----------|-------------|
  | `session_id` | Session ID issued by `POST /api/age` |

- **Response** : `image/jpeg` byte stream
- **Used by** : `<img src="/api/images/{session_id}">` in `three_d.html`
- **Colab endpoint** : `GET {COLAB_BASE_URL}/aging/{image_id}`

---

### `POST /api/build-3d`
- **Role** : Sends the aged image to the Colab 3D generation endpoint
- **Input** : `application/x-www-form-urlencoded`

  | Field | Type | Description |
  |-------|------|-------------|
  | `session_id` | string | Session ID from the previous step |

- **Flow**
  ```
  Receive session_id
    → Look up aging result URL in image_store
    → Download image bytes from Colab
    → Forward to Colab POST /3d
    → Receive { model_id }
    → Update image_store[session_id]["model_url"]
    → 303 Redirect → GET /preview?session_id={id}
  ```
- **Colab endpoint** : `POST {COLAB_BASE_URL}/3d`

---

## User Flow

```
① GET /
   Click "Start With a Photo" on the home screen

② GET /create
   Upload a photo and enter a target age, then click "Next"

③ POST /api/age
   Colab processes the image → session_id is issued
   → Automatically redirected to /preview

④ GET /preview
   Review the aging result, then:
   ┌─ Click Prev → GET /create?session_id={id}  (edit and resubmit)
   └─ Click Build in 3D → POST /api/build-3d

⑤ POST /api/build-3d
   Colab generates the 3D model
   → Automatically redirected to /preview (with 3D result)
```

---

## In-Memory Store Structure

```python
image_store = {
    "{session_id}": {
        "image_url":    "https://{ngrok}/aging/{image_id}",  # Colab aging result URL
        "age":          45,                                    # Target age (user input)
        "gender":       "female",                             # CLIP-predicted gender
        "current_age":  27,                                   # CLIP-predicted current age (center value)
        "model_url":    "https://{ngrok}/3d/{model_id}",     # Colab 3D result URL (optional)
    }
}
```

> **Note** : `image_store` lives in server memory only. It is cleared on server restart.

---

## Configuration

Update the Colab ngrok URL at the top of `main.py`:

```python
COLAB_BASE_URL = "https://xxxx-xx-xx-xxx-xxx.ngrok-free.app"
```

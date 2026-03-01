import io
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ── Model (loaded once at startup) ────────────────────────────────────────────
_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
_clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
_clip_model.eval()

# ── Age prompt groups ──────────────────────────────────────────────────────────
age_prompt_groups = [
    {
        "label": "baby",
        "center": 2,
        "prompts": [
            "a photo of a baby or toddler",
            "a photo of an infant with chubby cheeks",
            "a portrait of a very young child under 3 years old",
        ],
    },
    {
        "label": "child",
        "center": 8,
        "prompts": [
            "a photo of a young child",
            "a portrait of a kid in elementary school",
            "a photo of a child aged 4 to 12",
        ],
    },
    {
        "label": "teenager",
        "center": 16,
        "prompts": [
            "a photo of a teenager",
            "a portrait of a high school student",
            "a photo of an adolescent with youthful features",
        ],
    },
    {
        "label": "young adult",
        "center": 27,
        "prompts": [
            "a photo of a young adult in their twenties",
            "a portrait of a college-aged person",
            "a photo of someone in their mid-twenties with smooth skin",
        ],
    },
    {
        "label": "middle-aged",
        "center": 45,
        "prompts": [
            "a photo of a middle-aged person",
            "a portrait of someone in their forties with slight wrinkles",
            "a photo of a mature adult with some gray hair",
        ],
    },
    {
        "label": "senior",
        "center": 68,
        "prompts": [
            "a photo of an elderly person",
            "a portrait of an old person with white hair and deep wrinkles",
            "a photo of a senior citizen over sixty",
        ],
    },
]

def predict_demographics(image_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # --- Gender ---
    gender_texts = [
        "a photo of a male face",
        "a photo of a female face",
    ]
    inputs = _clip_processor(
        text=gender_texts, images=image, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        outputs = _clip_model(**inputs)
    gender_sims = torch.nn.functional.cosine_similarity(
        outputs.image_embeds.expand(len(gender_texts), -1),
        outputs.text_embeds,
    )
    gender = "male" if gender_sims.argmax().item() == 0 else "female"

    # --- Age (ensemble per group) ---
    group_scores = []
    for group in age_prompt_groups:
        inputs = _clip_processor(
            text=group["prompts"], images=image, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            outputs = _clip_model(**inputs)
        sims = torch.nn.functional.cosine_similarity(
            outputs.image_embeds.expand(len(group["prompts"]), -1),
            outputs.text_embeds,
        )
        group_scores.append(sims.mean().item())

    best_idx = group_scores.index(max(group_scores))
    current_age = age_prompt_groups[best_idx]["center"]

    print(f"[CLIP] gender={gender}, age={current_age}")
    return {"gender": gender, "current_age": current_age}

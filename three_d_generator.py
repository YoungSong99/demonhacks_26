import sys
sys.path.insert(0, "./FaceLift")

import os
import yaml
import json
import importlib
import warnings
from typing import List, Tuple, Optional

from dataclasses import dataclass
from typing import Optional
import torch

import torch
import numpy as np
from PIL import Image
from einops import rearrange
from easydict import EasyDict as edict
from rich import print
from rembg import remove
from facenet_pytorch import MTCNN
from huggingface_hub import snapshot_download

from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline
from gslrm.model.gaussians_renderer import render_turntable, imageseq2video
from utils_folder.face_utils import preprocess_image, preprocess_image_without_cropping

import util

# suppress FutureWarning from facenet_pytorch
warnings.filterwarnings("ignore", category=FutureWarning, module="facenet_pytorch")

IMAGE_PATH = "./000.jpg"
OUTPUT_PATH = "./outputs"

# Configuration constants
DEFAULT_IMG_SIZE = 512
DEFAULT_TURNTABLE_VIEWS = 150
DEFAULT_TURNTABLE_FPS = 30
HF_REPO_ID = "wlyu/OpenFaceLift"

@dataclass
class ModelBundle:
    device: torch.device
    diffusion_pipeline: object
    random_generator: torch.Generator
    color_prompt_embeddings: torch.Tensor
    gslrm_model: object
    camera_intrinsics_tensor: torch.Tensor
    camera_extrinsics_tensor: torch.Tensor

def build_bundle() -> ModelBundle:
    mvdiffusion_checkpoint_path, gslrm_checkpoint_path, gslrm_config_path = util.get_model_paths()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    diffusion_pipeline, random_generator, color_prompt_embeddings = util.initialize_mvdiffusion_pipeline(
        mvdiffusion_checkpoint_path, device
    )
    gslrm_model = util.initialize_gslrm_model(gslrm_checkpoint_path, gslrm_config_path, device)
    camera_intrinsics_tensor, camera_extrinsics_tensor = util.setup_camera_parameters(device)

    return ModelBundle(
        device=device,
        diffusion_pipeline=diffusion_pipeline,
        random_generator=random_generator,
        color_prompt_embeddings=color_prompt_embeddings,
        gslrm_model=gslrm_model,
        camera_intrinsics_tensor=camera_intrinsics_tensor,
        camera_extrinsics_tensor=camera_extrinsics_tensor,
    )

bundle = build_bundle()

class ThreeDGenerator:
    def __init__(self,  age_transformed_img, bundle: ModelBundle, auto_crop: bool = True):
        self.image_2d = age_transformed_img
        self.bundle = bundle
        self.auto_crop = auto_crop

        self.face_detector = None
        if auto_crop:
            self.face_detector = util.initialize_face_detector(bundle.device)

    def process_single_image(
        self,
        output_dir: str,
        guidance_scale_2D: float,
        step_2D: int,
    ) -> None:
        print(f"Processing image...")

        # access to the model with bundle
        unclip_pipeline = self.bundle.diffusion_pipeline
        generator = self.bundle.random_generator
        color_prompt_embedding = self.bundle.color_prompt_embeddings
        gs_lrm_model = self.bundle.gslrm_model
        demo_fxfycxcy = self.bundle.camera_intrinsics_tensor
        demo_c2w = self.bundle.camera_extrinsics_tensor

        input_image_np = np.array(self.image_2d)

        # preprocess image
        try:
            if self.auto_crop:
                input_image = preprocess_image(input_image_np)
            else:
                input_image = preprocess_image_without_cropping(input_image_np)
        except Exception as e:
            print(f"Failed to process image: {e}, applying fallback processing")
            try:
                input_image = remove(input_image)
                input_image = input_image.resize((DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE), Image.LANCZOS)
            except Exception as e2:
                print(f"Background removal also failed: {e2}, using original image")
                input_image = input_image.resize((DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE), Image.LANCZOS)

        #input_image.save(os.path.join("input.png"))

        # generate multi-view images
        mv_imgs = unclip_pipeline(
            input_image,
            None,
            prompt_embeds=color_prompt_embedding,
            guidance_scale=guidance_scale_2D,
            num_images_per_prompt=1,
            num_inference_steps=step_2D,
            generator=generator,
            eta=1.0,
        ).images

        # Always use 6 views
        if len(mv_imgs) == 7:
            views = [mv_imgs[i] for i in [1, 2, 3, 4, 5, 6]]
        elif len(mv_imgs) == 6:
            views = [mv_imgs[i] for i in [0, 1, 2, 3, 4, 5]]
        else:
            raise ValueError(f"Unexpected number of views: {len(mv_imgs)}")

        # Save multi-view image
        lrm_input_save = Image.new("RGB", (DEFAULT_IMG_SIZE * len(mv_imgs), DEFAULT_IMG_SIZE))
        for i, view in enumerate(mv_imgs):
            lrm_input_save.paste(view, (DEFAULT_IMG_SIZE * i, 0))
        lrm_input_save.save(os.path.join(output_dir, "multiview.png"))

        # Prepare input for 3D reconstruction
        lrm_input = np.stack([np.array(view) for view in views], axis=0)
        lrm_input = torch.from_numpy(lrm_input).float()[None].to(demo_fxfycxcy.device) / 255
        lrm_input = rearrange(lrm_input, "b v h w c -> b v c h w")

        index = torch.stack([
            torch.zeros(lrm_input.size(1)).long(),
            torch.arange(lrm_input.size(1)).long(),
        ], dim=-1)
        demo_index = index[None].to(demo_fxfycxcy.device)

        # Create batch
        batch = edict({
            "image": lrm_input,
            "c2w": demo_c2w,
            "fxfycxcy": demo_fxfycxcy,
            "index": demo_index,
        })

        # 3D reconstruction inference
        with torch.autocast(enabled=True, device_type="cuda", dtype=torch.float16):
            result = gs_lrm_model.forward(batch, create_visual=False, split_data=True)

        # Save Gaussian splatting result
        result.gaussians[0].apply_all_filters(
            opacity_thres=0.04,
            scaling_thres=0.1,
            floater_thres=0.6,
            crop_bbx=[-0.91, 0.91, -0.91, 0.91, -1.0, 1.0],
            cam_origins=None,
            nearfar_percent=(0.0001, 1.0),
        ).save_ply(os.path.join(output_dir, "gaussians.ply"))

        # Save rendered output
        comp_image = result.render[0].unsqueeze(0).detach()
        v = comp_image.size(1)
        if v > 10:
            comp_image = comp_image[:, :: v // 10, :, :, :]
        comp_image = rearrange(comp_image, "x v c h w -> (x h) (v w) c")
        comp_image = (comp_image.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
        Image.fromarray(comp_image).save(os.path.join(output_dir, "output.png"))

        # Generate turntable video
        vis_image = render_turntable(
            result.gaussians[0],
            rendering_resolution=DEFAULT_IMG_SIZE,
            num_views=DEFAULT_TURNTABLE_VIEWS,
        )
        vis_image = rearrange(vis_image, "h (v w) c -> v h w c", v=DEFAULT_TURNTABLE_VIEWS)
        vis_image = np.ascontiguousarray(vis_image)
        imageseq2video(
            vis_image,
            os.path.join(output_dir, "turntable.mp4"),
            fps=DEFAULT_TURNTABLE_FPS
        )

    def generate_3d_img(
        self,
        output_dir: str,
        seed: int = 4,
        guidance_scale_2D: float = 3.0,
        step_2D: int = 50,
    ) -> None:
        # seed
        self.bundle.random_generator.manual_seed(seed)

        self.process_single_image(
            output_dir=output_dir,
            guidance_scale_2D=guidance_scale_2D,
            step_2D=step_2D,
        )


def main():
    test_img_path = "./000.jpg"

    test_img = Image.open(test_img_path).convert("RGB")
    three_d_generator = ThreeDGenerator(test_img, bundle, auto_crop=True)
    three_d_generator.generate_3d_img(OUTPUT_PATH)

if __name__ == "__main__":
    main()
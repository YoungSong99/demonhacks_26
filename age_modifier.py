# input/output (512, 512)

import sys
sys.path.insert(0, "./FADING_stable")

import os
import argparse
from diffusers import StableDiffusionPipeline, DDIMScheduler

from FADING_util import util
from p2p import *
from null_inversion import *
import io
from PIL import Image
import tempfile

##################################
specialized_path = "./FADING_stable/finetune_double_prompt_150_random"

class AgeModifier:
    def __init__(self, image_bytes, init_age, target_age, gender):
        self.image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)

        self.image.save(tmp.name)
        self.image_path = tmp.name
        tmp.close()
        self.init_age = init_age
        self.target_age = target_age # we have only one target age
        self.gt_gender = int(gender == 'female')
        self.person_placeholder = util.get_person_placeholder(init_age, self.gt_gender) # classify person depends on age
        self.inversion_prompt = f"photo of {init_age} year old {self.person_placeholder}"

        # create scheduler for diffusion model
        self.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                  clip_sample=False, set_alpha_to_one=False,
                                  steps_offset=1)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.g_cuda = torch.Generator(device=self.device)  # for same seed
        self.ldm_stable = StableDiffusionPipeline.from_pretrained(specialized_path,
                                                             scheduler=self.scheduler,
                                                             safety_checker=None).to(self.device)
        self.tokenizer = self.ldm_stable.tokenizer

    def generate_age_img(self):
        # %% null text inversion
        # image preprocessing + create latent representation
        null_inversion = NullInversion(self.ldm_stable)
        (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(self.image_path, self.inversion_prompt,
                                                                              offsets=(0, 0, 0, 0), verbose=True)

        # generate new prompt
        print(f'Age editing with target age {self.target_age}...')
        new_person_placeholder = util.get_person_placeholder(self.target_age, self.gt_gender)
        new_prompt = self.inversion_prompt.replace(self.person_placeholder, new_person_placeholder)
        new_prompt = new_prompt.replace(str(self.init_age), str(self.target_age))

        blend_word = (((str(self.init_age), self.person_placeholder,), (str(self.target_age), new_person_placeholder,)))
        is_replace_controller = True #

        prompts = [self.inversion_prompt, new_prompt]

        cross_replace_steps = {'default_': .8, }
        self_replace_steps = .5

        eq_params = {"words": (str(self.target_age)), "values": (1,)}

        controller = make_controller(prompts, is_replace_controller, cross_replace_steps, self_replace_steps,
                                     self.tokenizer, blend_word, eq_params)

        images, _ = p2p_text2image(self.ldm_stable, prompts, controller, generator=self.g_cuda.manual_seed(0),
                                   latent=x_t, uncond_embeddings=uncond_embeddings)

        new_img = images[-1]
        new_img_pil = Image.fromarray(new_img)
        new_img_pil.save(f'test_{self.target_age}.png')

        return new_img_pil


def main():
    """
    
    test_img = "000.jpg"
    test_init_age = 22
    test_target_age = 30
    test_gender = 'female'

    modifier = AgeModifier(test_img, test_init_age, test_target_age, test_gender)
    modifier.generate_age_img()
    """



if __name__ == "__main__":
    main()

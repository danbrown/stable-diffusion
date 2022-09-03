import argparse, os, sys, glob
import gradio as gr
import numpy as np
import torch
from torchvision.utils import make_grid
from einops import rearrange
import os, re
from PIL import Image
import torch
import pandas as pd
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

# Fix ldm not found for apple silicon devices
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ldm.util import instantiate_from_config
from optimUtils import split_weighted_subprompts, logger

from transformers import logging
logging.set_verbosity_error()
import mimetypes
mimetypes.init()
mimetypes.add_type("application/javascript", ".js")


# Fix running in non cuda devices
def get_device():
    if(torch.cuda.is_available()):
        return 'cuda'
    elif(torch.backends.mps.is_available()):
        return 'mps'
    else:
        return 'cpu'

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


# Patch the nsfw images (remove the filter)
def patch_nsfw():
  with open('/usr/local/lib/python3.7/dist-packages/diffusers/pipelines/stable_diffusion/safety_checker.py','w') as file:
        file.write('''
import numpy as np
import torch
import torch.nn as nn

from transformers import CLIPConfig, CLIPVisionModel, PreTrainedModel

from ...utils import logging


logger = logging.get_logger(__name__)


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.T)


class StableDiffusionSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

        self.concept_embeds = nn.Parameter(torch.ones(17, config.projection_dim), requires_grad=False)
        self.special_care_embeds = nn.Parameter(torch.ones(3, config.projection_dim), requires_grad=False)

        self.register_buffer("concept_embeds_weights", torch.ones(17))
        self.register_buffer("special_care_embeds_weights", torch.ones(3))

    @torch.no_grad()
    def forward(self, clip_input, images):
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds).cpu().numpy()
        cos_dist = cosine_distance(image_embeds, self.concept_embeds).cpu().numpy()

        result = []
        batch_size = image_embeds.shape[0]
        for i in range(batch_size):
            result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": []}

            # increase this value to create a stronger `nfsw` filter
            # at the cost of increasing the possibility of filtering benign images
            adjustment = 0.0

            for concet_idx in range(len(special_cos_dist[0])):
                concept_cos = special_cos_dist[i][concet_idx]
                concept_threshold = self.special_care_embeds_weights[concet_idx].item()
                result_img["special_scores"][concet_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["special_scores"][concet_idx] > 0:
                    result_img["special_care"].append({concet_idx, result_img["special_scores"][concet_idx]})
                    adjustment = 0.01

            for concet_idx in range(len(cos_dist[0])):
                concept_cos = cos_dist[i][concet_idx]
                concept_threshold = self.concept_embeds_weights[concet_idx].item()
                result_img["concept_scores"][concet_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["concept_scores"][concet_idx] > 0:
                    result_img["bad_concepts"].append(concet_idx)

            result.append(result_img)

        has_nsfw_concepts = [len(res["bad_concepts"]) > 0 for res in result]

        #for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
        #    if has_nsfw_concept:
        #        images[idx] = np.zeros(images[idx].shape)  # black image

        if any(has_nsfw_concepts):
            logger.warning(
                "Potential NSFW content was detected in one or more images, but the NSFW filter is off."
            )

        return images, has_nsfw_concepts''')


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd

config = "optimizedSD/v1-inference.yaml"
ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"
sd = load_model_from_config(f"{ckpt}")
li, lo = [], []
for key, v_ in sd.items():
    sp = key.split(".")
    if (sp[0]) == "model":
        if "input_blocks" in sp:
            li.append(key)
        elif "middle_block" in sp:
            li.append(key)
        elif "time_embed" in sp:
            li.append(key)
        else:
            lo.append(key)
for key in li:
    sd["model1." + key[6:]] = sd.pop(key)
for key in lo:
    sd["model2." + key[6:]] = sd.pop(key)

config = OmegaConf.load(f"{config}")

model = instantiate_from_config(config.modelUNet)
_, _ = model.load_state_dict(sd, strict=False)
model.eval()

modelCS = instantiate_from_config(config.modelCondStage)
_, _ = modelCS.load_state_dict(sd, strict=False)
modelCS.eval()

modelFS = instantiate_from_config(config.modelFirstStage)
_, _ = modelFS.load_state_dict(sd, strict=False)
modelFS.eval()
del sd


def generate(
    prompt,
    ddim_steps,
    n_iter,
    batch_size,
    Height,
    Width,
    scale,
    ddim_eta,
    unet_bs,
    device,
    seed,
    outdir,
    img_format,
    turbo,
    full_precision,
    sampler,
):

    device = get_device()

    C = 4
    f = 8
    start_code = None
    model.unet_bs = unet_bs
    model.turbo = turbo
    model.cdevice = device
    modelCS.cond_stage_model.device = device

    if seed == "":
        seed = randint(0, 1000000)
    seed = int(seed)
    seed_everything(seed)
    # Logging
    logger(locals(), "logs/txt2img_gradio_logs.csv")

    if device != "cpu" and full_precision == False:
        model.half()
        modelFS.half()
        modelCS.half()

    tic = time.time()
    os.makedirs(outdir, exist_ok=True)
    outpath = outdir
    sample_path = os.path.join(outpath, "_".join(re.split(":| ", prompt)))[:150]
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    
    # n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    assert prompt is not None
    data = [batch_size * [prompt]]

    if full_precision == False and device != "cpu":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    all_samples = []
    seeds = ""
    with torch.no_grad():

        all_samples = list()
        for _ in trange(n_iter, desc="Sampling"):
            for prompts in tqdm(data, desc="data"):
                with precision_scope("cuda"):
                    modelCS.to(device)
                    uc = None
                    if scale != 1.0:
                        uc = modelCS.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    subprompts, weights = split_weighted_subprompts(prompts[0])
                    if len(subprompts) > 1:
                        c = torch.zeros_like(uc)
                        totalWeight = sum(weights)
                        # normalize each "sub prompt" and add it
                        for i in range(len(subprompts)):
                            weight = weights[i]
                            # if not skip_normalize:
                            weight = weight / totalWeight
                            c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                    else:
                        c = modelCS.get_learned_conditioning(prompts)

                    shape = [batch_size, C, Height // f, Width // f]

                    if device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelCS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    samples_ddim = model.sample(
                        S=ddim_steps,
                        conditioning=c,
                        seed=seed,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                        eta=ddim_eta,
                        x_T=start_code,
                        sampler = sampler,
                    )

                    modelFS.to(device)
                    print("saving images")
                    for i in range(batch_size):

                        x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                        x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        all_samples.append(x_sample.to("cpu"))
                        x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(sample_path, "seed_" + str(seed) + "_" + f"{base_count:05}.{img_format}")
                        )
                        seeds += str(seed) + ","
                        seed += 1
                        base_count += 1

                    if device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelFS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    del samples_ddim
                    del x_sample
                    del x_samples_ddim
                    print("memory_final = ", torch.cuda.memory_allocated() / 1e6)

    toc = time.time()

    time_taken = (toc - tic) / 60.0
    grid = torch.cat(all_samples, 0)
    grid = make_grid(grid, nrow=n_iter)
    grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()

    txt = (
        "Samples finished in "
        + str(round(time_taken, 3))
        + " minutes and exported to "
        + sample_path
        + "\nSeeds used = "
        + seeds[:-1]
    )
    return Image.fromarray(grid.astype(np.uint8)), txt


demo = gr.Interface(
    fn=generate,
    inputs=[
        "text",
        gr.Slider(1, 1000, value=50),
        gr.Slider(1, 100, step=1),
        gr.Slider(1, 100, step=1),
        gr.Slider(64, 4096, value=512, step=64),
        gr.Slider(64, 4096, value=512, step=64),
        gr.Slider(0, 50, value=7.5, step=0.1),
        gr.Slider(0, 1, step=0.01),
        gr.Slider(1, 2, value=1, step=1),
        gr.Text(value="cuda"),
        "text",
        gr.Text(value="outputs/txt2img-samples"),
        gr.Radio(["png", "jpg"], value='png'),
        "checkbox",
        "checkbox",
        gr.Radio(["ddim", "plms"], value="plms"),
    ],
    outputs=["image", "text"],
)
demo.launch()
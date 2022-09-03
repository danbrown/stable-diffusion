
from contextlib import contextmanager, nullcontext
from torch import autocast
from pytorch_lightning import seed_everything
import time
from torchvision.utils import make_grid
from einops import rearrange
from itertools import islice
from imwatermark import WatermarkEncoder
from tqdm import tqdm, trange
from PIL import Image
from omegaconf import OmegaConf
import numpy as np
import torch
import cv2
import sys
import os
# import gradio as gr
from safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

def get_device():
    if (torch.cuda.is_available()):
        return 'cuda'
    elif (torch.backends.mps.is_available()):
        return 'mps'
    else:
        return 'cpu'




# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(
    safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(get_device())
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open(
            "assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    # assert x_checked_image.shape[0] == len(has_nsfw_concept)
    # for i in range(len(has_nsfw_concept)):
    #     if has_nsfw_concept[i]:
    #         x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

 # parser.add_argument(
    #     "--prompt",
    #     type=str,
    #     nargs="?",
    #     default="a painting of a virus monster playing guitar",
    #     help="the prompt to render"
    # )
    # parser.add_argument(
    #     "--outdir",
    #     type=str,
    #     nargs="?",
    #     help="dir to write results to",
    #     default="outputs/txt2img-samples"
    # )
    # parser.add_argument(
    #     "--skip_grid",
    #     action='store_true',
    #     help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    # )
    # parser.add_argument(
    #     "--skip_save",
    #     action='store_true',
    #     help="do not save individual samples. For speed measurements.",
    # )
    # parser.add_argument(
    #     "--ddim_steps",
    #     type=int,
    #     default=50,
    #     help="number of ddim sampling steps",
    # )
    # parser.add_argument(
    #     "--plms",
    #     action='store_true',
    #     help="use plms sampling",
    # )
    # parser.add_argument(
    #     "--laion400m",
    #     action='store_true',
    #     help="uses the LAION400M model",
    # )
    # parser.add_argument(
    #     "--fixed_code",
    #     action='store_true',
    #     help="if enabled, uses the same starting code across samples ",
    # )
    # parser.add_argument(
    #     "--ddim_eta",
    #     type=float,
    #     default=0.0,
    #     help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    # )
    # parser.add_argument(
    #     "--n_iter",
    #     type=int,
    #     default=1,
    #     help="sample this often",
    # )
    # parser.add_argument(
    #     "--H",
    #     type=int,
    #     default=512,
    #     help="image height, in pixel space",
    # )
    # parser.add_argument(
    #     "--W",
    #     type=int,
    #     default=512,
    #     help="image width, in pixel space",
    # )
    # parser.add_argument(
    #     "--C",
    #     type=int,
    #     default=4,
    #     help="latent channels",
    # )
    # parser.add_argument(
    #     "--f",
    #     type=int,
    #     default=8,
    #     help="downsampling factor",
    # )
    # parser.add_argument(
    #     "--n_samples",
    #     type=int,
    #     default=1,
    #     help="how many samples to produce for each given prompt. A.k.a. batch size",
    # )
    # parser.add_argument(
    #     "--n_rows",
    #     type=int,
    #     default=0,
    #     help="rows in the grid (default: n_samples)",
    # )
    # parser.add_argument(
    #     "--scale",
    #     type=float,
    #     default=7.5,
    #     help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    # )
    # parser.add_argument(
    #     "--from-file",
    #     type=str,
    #     help="if specified, load prompts from this file",
    # )
    # parser.add_argument(
    #     "--config",
    #     type=str,
    #     default="configs/stable-diffusion/v1-inference.yaml",
    #     help="path to config which constructs model",
    # )
    # parser.add_argument(
    #     "--ckpt",
    #     type=str,
    #     default="models/ldm/stable-diffusion-v1/model.ckpt",
    #     help="path to checkpoint of model",
    # )
    # parser.add_argument(
    #     "--seed",
    #     type=int,
    #     default=42,
    #     help="the seed (for reproducible sampling)",
    # )
    # parser.add_argument(
    #     "--precision",
    #     type=str,
    #     help="evaluate at this precision",
    #     choices=["full", "autocast"],
    #     default="autocast"
    # )


def generate(
    prompt,
    outdir,
    skip_grid,
    skip_save,
    ddim_steps,
    plms,
    laion400m,
    fixed_code,
    ddim_eta,
    n_iter,
    Height,
    Width,
    Channels,
    downsamplingFactor,
    n_samples,
    n_rows,
    scale,
    from_file,
    config,
    ckpt,
    seed,
    precision,
):

    if laion400m:
        print("Falling back to LAION 400M model...")
        config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        ckpt = "models/ldm/text2img-large/model.ckpt"
        outdir = "outputs/txt2img-samples-laion400m"

    

    config = OmegaConf.load(f"{config}")
    model = load_model_from_config(config, f"{ckpt}")

    device = torch.device(get_device())
    gen_seed = torch.Generator("cpu").manual_seed(int(seed))
    model = model.to(device)

    seed_everything(gen_seed)

    if plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(outdir, exist_ok=True)
    outpath = outdir

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size
    if not from_file:
        prompt = prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {from_file}")
        with open(from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if fixed_code:
        start_code = torch.randn(
            [n_samples, Channels, Height // downsamplingFactor, Width // downsamplingFactor], device="cpu"
        ).to(torch.device(device))

    precision_scope = autocast if precision == "autocast" else nullcontext
    if device.type == 'mps':
        precision_scope = nullcontext  # have to use f32 on mps
    with torch.no_grad():
        with precision_scope(device.type):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(
                                batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [Channels, Height // downsamplingFactor,
                                 Width // downsamplingFactor]
                        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                         conditioning=c,
                                                         batch_size=n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=uc,
                                                         eta=ddim_eta,
                                                         x_T=start_code,
                                                         )

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp(
                            (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image, has_nsfw_concept = check_safety(
                            x_samples_ddim)

                        x_checked_image_torch = torch.from_numpy(
                            x_checked_image).permute(0, 3, 1, 2)

                        if not skip_save:
                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * \
                                    rearrange(x_sample.cpu().numpy(),
                                              'c h w -> h w c')
                                img = Image.fromarray(
                                    x_sample.astype(np.uint8))
                                img = put_watermark(img, wm_encoder)
                                img.save(os.path.join(
                                    sample_path,  "seed_" + str(gen_seed) + "_" + f"{base_count:05}.png"))
                                base_count += 1

                        if not skip_grid:
                            all_samples.append(x_checked_image_torch)

                if not skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * \
                        rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(outpath, "seed_" +
                             str(gen_seed) + "_" + f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()

    time_taken = (toc - tic) / 60.0

    txt = (
        "Samples finished in "
        + str(round(time_taken, 3))
        + " minutes and exported to "
        + sample_path
        + "\nSeeds used = "
        + str(seed)
    )

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")

    return img, txt


# demo = gr.Interface(
#     fn=generate,
#     inputs=[
#         gr.inputs.Textbox(lines=2, label="Prompt",
#                           placeholder="a painting of a virus monster playing guitar"),
#         gr.inputs.Textbox(lines=1, label="Output directory",
#                           default="outputs/txt2img-samples"),
#         gr.inputs.Checkbox(label="Skip grid", default=False),
#         gr.inputs.Checkbox(label="Skip save", default=False),
#         gr.Slider(1, 1000, value=50),
#         gr.inputs.Checkbox(label="plms", default=False),
#         gr.inputs.Checkbox(label="laion400m", default=False),
#         gr.inputs.Checkbox(label="fixed code", default=False),
#         gr.Slider(0.0, 1.0, value=0.0),
#         gr.Slider(1, 10, value=1, step=1),
#         gr.Slider(256, 4096, value=512, step=64),
#         gr.Slider(256, 4096, value=512, step=64),
#         gr.Slider(1, 8, value=4, step=1),
#         gr.Slider(1, 16, value=8, step=1),
#         gr.Slider(1, 10, value=1, step=1),
#         gr.Slider(0, 10, value=0, step=1),
#         gr.Slider(0.0, 10.0, value=7.5),
#         gr.inputs.Textbox(lines=1, label="from file",
#                           placeholder="path/to/file"),
#         gr.inputs.Textbox(lines=1, label="config",
#                           default="configs/stable-diffusion/v1-inference.yaml"),
#         gr.inputs.Textbox(lines=1, label="ckpt",
#                           default="models/ldm/stable-diffusion-v1/model.ckpt"),
#         gr.inputs.Textbox(lines=1, label="seed", default="0"),
#         gr.inputs.Dropdown(label="precision", default="autocast", choices=[
#                            "full", "autocast"]),
#     ],
#     outputs=["image", "text"],
# )
# demo.launch()


# import numpy as np
# import torch
# import torch.nn as nn

# from transformers import CLIPConfig, CLIPVisionModel, PreTrainedModel

# from ...utils import logging


# logger = logging.get_logger(__name__)


# def cosine_distance(image_embeds, text_embeds):
#     normalized_image_embeds = nn.functional.normalize(image_embeds)
#     normalized_text_embeds = nn.functional.normalize(text_embeds)
#     return torch.mm(normalized_image_embeds, normalized_text_embeds.T)


# class StableDiffusionSafetyChecker(PreTrainedModel):
#     config_class = CLIPConfig

#     def __init__(self, config: CLIPConfig):
#         super().__init__(config)

#         self.vision_model = CLIPVisionModel(config.vision_config)
#         self.visual_projection = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

#         self.concept_embeds = nn.Parameter(torch.ones(17, config.projection_dim), requires_grad=False)
#         self.special_care_embeds = nn.Parameter(torch.ones(3, config.projection_dim), requires_grad=False)

#         self.register_buffer("concept_embeds_weights", torch.ones(17))
#         self.register_buffer("special_care_embeds_weights", torch.ones(3))

#     @torch.no_grad()
#     def forward(self, clip_input, images):
#         pooled_output = self.vision_model(clip_input)[1]  # pooled_output
#         image_embeds = self.visual_projection(pooled_output)

#         special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds).cpu().numpy()
#         cos_dist = cosine_distance(image_embeds, self.concept_embeds).cpu().numpy()

#         result = []
#         batch_size = image_embeds.shape[0]
#         for i in range(batch_size):
#             result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": []}

#             # increase this value to create a stronger `nfsw` filter
#             # at the cost of increasing the possibility of filtering benign images
#             adjustment = 0.0

#             for concet_idx in range(len(special_cos_dist[0])):
#                 concept_cos = special_cos_dist[i][concet_idx]
#                 concept_threshold = self.special_care_embeds_weights[concet_idx].item()
#                 result_img["special_scores"][concet_idx] = round(concept_cos - concept_threshold + adjustment, 3)
#                 if result_img["special_scores"][concet_idx] > 0:
#                     result_img["special_care"].append({concet_idx, result_img["special_scores"][concet_idx]})
#                     adjustment = 0.01

#             for concet_idx in range(len(cos_dist[0])):
#                 concept_cos = cos_dist[i][concet_idx]
#                 concept_threshold = self.concept_embeds_weights[concet_idx].item()
#                 result_img["concept_scores"][concet_idx] = round(concept_cos - concept_threshold + adjustment, 3)
#                 if result_img["concept_scores"][concet_idx] > 0:
#                     result_img["bad_concepts"].append(concet_idx)

#             result.append(result_img)

#         has_nsfw_concepts = [len(res["bad_concepts"]) > 0 for res in result]

#         #for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
#         #    if has_nsfw_concept:
#         #        images[idx] = np.zeros(images[idx].shape)  # black image

#         if any(has_nsfw_concepts):
#             logger.warning(
#                 "Potential NSFW content was detected in one or more images, but the NSFW filter is off."
#             )

#         return images, has_nsfw_concepts''')
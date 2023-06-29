from __future__ import annotations
import gradio as gr
from typing import Iterable
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from PIL import Image
import os
import sys
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from einops import repeat
from imwatermark import WatermarkEncoder
from pathlib import Path
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

class Monochrome(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.neutral,
        secondary_hue: colors.Color | str = colors.neutral,
        neutral_hue: colors.Color | str = colors.red,
        spacing_size: sizes.Size | str = sizes.spacing_lg,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_md,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Open Sans"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "Consolas",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        self.name = "monochrome"
        super().set(
            # Colors
            slider_color="*neutral_900",
            slider_color_dark="*neutral_500",
            body_text_color="*neutral_900",
            block_label_text_color="*body_text_color",
            block_title_text_color="*body_text_color",
            body_text_color_subdued="*neutral_700",
            background_fill_primary_dark="*neutral_900",
            background_fill_secondary_dark="*neutral_800",
            block_background_fill_dark="*neutral_800",
            input_background_fill_dark="*neutral_700",
            # Button Colors
            button_primary_background_fill="*neutral_900",
            button_primary_background_fill_hover="*neutral_700",
            button_primary_text_color="white",
            button_primary_background_fill_dark="*neutral_600",
            button_primary_background_fill_hover_dark="*neutral_600",
            button_primary_text_color_dark="white",
            button_secondary_background_fill="*button_primary_background_fill",
            button_secondary_background_fill_hover="*button_primary_background_fill_hover",
            button_secondary_text_color="*button_primary_text_color",
            button_cancel_background_fill="*button_primary_background_fill",
            button_cancel_background_fill_hover="*button_primary_background_fill_hover",
            button_cancel_text_color="*button_primary_text_color",
            checkbox_label_background_fill="*button_primary_background_fill",
            checkbox_label_background_fill_hover="*button_primary_background_fill_hover",
            checkbox_label_text_color="*button_primary_text_color",
            checkbox_background_color_selected="*neutral_600",
            checkbox_background_color_dark="*neutral_700",
            checkbox_background_color_selected_dark="*neutral_700",
            checkbox_border_color_selected_dark="*neutral_800",
            # Padding
            checkbox_label_padding="*spacing_md",
            button_large_padding="*spacing_lg",
            button_small_padding="*spacing_sm",
            # Borders
            block_border_width="0px",
            block_border_width_dark="1px",
            shadow_drop_lg="0 1px 4px 0 rgb(0 0 0 / 0.1)",
            block_shadow="*shadow_drop_lg",
            block_shadow_dark="none",
            # Block Labels
            block_title_text_weight="600",
            block_label_text_weight="600",
            block_label_text_size="*text_md",
        )

def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    return sampler

mono = Monochrome()
torch.set_grad_enabled(False)
sampler = initialize_model(sys.argv[1], sys.argv[2])

def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch


def inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h // 8, w // 8)
    start_code = torch.from_numpy(start_code).to(
        device=device, dtype=torch.float32)

    with torch.no_grad(), \
            torch.autocast("cuda"):
        batch = make_batch_sd(image, mask, txt=prompt,
                              device=device, num_samples=num_samples)

        c = model.cond_stage_model.encode(batch["txt"])

        c_cat = list()
        for ck in model.concat_keys:
            cc = batch[ck].float()
            if ck != model.masked_image_key:
                bchw = [num_samples, 4, h // 8, w // 8]
                cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
            else:
                cc = model.get_first_stage_encoding(
                    model.encode_first_stage(cc))
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)

        # cond
        cond = {"c_concat": [c_cat], "c_crossattn": [c]}

        # uncond cond
        uc_cross = model.get_unconditional_conditioning(num_samples, "")
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

        shape = [model.channels, h // 8, w // 8]
        samples_cfg, intermediates = sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=1.0,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_full,
            x_T=start_code,
        )
        x_samples_ddim = model.decode_first_stage(samples_cfg)

        result = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                             min=0.0, max=1.0)

        result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
    return [Image.fromarray(img.astype(np.uint8)) for img in result]

def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded

def pred(img, prompt, ddim_steps, num_samples, scale, seed, limit, radio):
    _img = img["image"].convert("RGB")
    _mask = img["mask"].convert("RGB")
    print(type(_img))

    if limit == "Constrained (512x512)":
        _img = _img.resize((1024, 1024), Image.NEAREST)
        _mask = _mask.resize((1024, 1024), Image.NEAREST)
        

    image = pad_image(_img) # resize to integer multiple of 32
    mask = pad_image(_mask) # resize to integer multiple of 32
    width, height = image.size
    print("Inpainting...", width, height)

    if radio == "Free-flow":
        prompt += "nothing, fill, background, "

    prompt += "photorealistic, 8k, best quality, high-resolution"
    print("Prompt is:",prompt)

    result = inpaint(
        sampler=sampler,
        image=image,
        mask=mask,
        prompt=prompt,
        seed=seed,
        scale=scale,
        ddim_steps=ddim_steps,
        num_samples=num_samples,
        h=height, w=width
    )

    return result

css = ".gradio-container {background-color: #FFFFFF,primary-hue: #FFFFFF} #accord{color:black} #title{color:black}}"

with gr.Blocks(theme = mono,css=css) as demo:
    demo.queue()
    with gr.Row():
        gr.Markdown('# Customize Rocket Mortgage Listings',elem_id="title")
    with gr.Row():
        with gr.Column():
            img = gr.Image(label='Image',source='upload', tool='sketch', type='pil', interactive='True').style(height=340)
            prompt = gr.Textbox(label='Prompt', placeholder="Your prompt (what you want in place of what is erased)!")
            run = gr.Button(label='Run')
            with gr.Accordion("More Customization",open=False,elem_id="accord"):
                num_samples = gr.Slider(label="Output Images", minimum=1, maximum=4, step=1)
                limit = gr.Dropdown(["Constrained (512x512)","No Limits"], label="Image Size", info="Constrain Output Size?", value="No Limits")
                radio = gr.Radio(["Free-flow", "Erase"], label="Mode", info="Select customization mode!", value="Free-flow")
                ddim_steps = gr.Slider(label="DDIM Steps", minimum=1, maximum=50, value=45, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=10, step=0.1)
                seed = gr.Slider(label="Seed (Randomize)", minimum=0,maximum=2147483647, step=1, randomize = True)

        with gr.Column():
            gallery = gr.Gallery(label="Output",show_label=True).style(columns=2,height='auto')
    with gr.Row():
        gr.Examples(inputs=img, examples=[os.path.join(os.path.dirname(__file__),"1_rml.jpg"),os.path.join(os.path.dirname(__file__),"2_rml.jpg"),os.path.join(os.path.dirname(__file__),"3_rml.jpg"),os.path.join(os.path.dirname(__file__),"4_rml.jpg")])
            
    run.click(fn=pred, inputs=[img, prompt, ddim_steps, num_samples, scale, seed, limit, radio],outputs=[gallery])

demo.launch(share=True)
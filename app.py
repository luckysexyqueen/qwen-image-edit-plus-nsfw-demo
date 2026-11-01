import random
import os
from typing import Optional

import gradio as gr
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

BASE_MODEL = "Qwen/Qwen-Image-Edit-2511"
LORA_MODEL = "ScottzillaSystems/qwen-image-edit-plus-nsfw-lora"

pipe = None
HAS_CUDA = torch.cuda.is_available()
if HAS_CUDA:
    print(f"Loading {BASE_MODEL} with LoRA {LORA_MODEL} …")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
    )
    pipe.load_lora_weights(LORA_MODEL)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print("Pipeline ready.")
else:
    print("No CUDA GPU detected. Starting UI in hosting-safe mode.")


def _resize_for_model(image: Image.Image, max_side: int = 1536) -> Image.Image:
    image = image.convert("RGB")
    scale = min(1.0, max_side / max(image.size))
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    return image


def generate(
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    steps: int,
    true_cfg_scale: float,
    seed: int,
) -> tuple[Optional[Image.Image], str]:
    """Edit an uploaded image with Qwen Image Edit Plus NSFW LoRA using the provided prompt."""
    if image is None:
        raise gr.Error("Upload an input image first.")
    if not prompt or not prompt.strip():
        raise gr.Error("Enter an editing prompt.")
    if pipe is None:
        raise gr.Error(
            "This Render instance has no CUDA GPU. The interface is online, "
            "but image generation requires a GPU deployment."
        )

    image = _resize_for_model(image)
    actual_seed = random.randint(0, 2**31 - 1) if seed < 0 else int(seed)
    generator = torch.manual_seed(actual_seed)
    result = pipe(
        image=[image],
        prompt=prompt.strip(),
        negative_prompt=negative_prompt.strip() or None,
        true_cfg_scale=float(true_cfg_scale),
        guidance_scale=1.0,
        num_inference_steps=int(steps),
        num_images_per_prompt=1,
        generator=generator,
    ).images[0]
    return result, f"Seed: {actual_seed}"


custom_css = """
.gradio-container {max-width: 1200px !important;}
.warning {border-left: 4px solid #eab308; padding: 10px 14px; background: #fff7ed;}
"""

with gr.Blocks(title="Qwen Image Edit Plus — NSFW LoRA", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Qwen Image Edit Plus — NSFW LoRA")
    gr.Markdown(
        '<div class="warning"><b>18+ only.</b> This model may generate sensitive adult content. '
        "Use only lawful, consensual content involving adults. Never use minors, non-consensual sexual content, "
        "sexual violence, or real people without permission.</div>"
    )
    gr.Markdown(
        "Upload an image and describe the edit in detail. The demo uses "
        f"[{LORA_MODEL}](https://huggingface.co/{LORA_MODEL}) on top of "
        f"[{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})."
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input image")
            prompt = gr.Textbox(
                label="Editing prompt",
                placeholder="Example: Change the outfit to a black evening dress, cinematic studio lighting",
                lines=4,
            )
            negative_prompt = gr.Textbox(
                label="Negative prompt (optional)",
                value="low quality, blurry, distorted anatomy, extra fingers, watermark",
                lines=2,
            )
            with gr.Row():
                steps = gr.Slider(10, 50, value=30, step=1, label="Inference steps")
                cfg = gr.Slider(1.0, 6.0, value=4.0, step=0.1, label="True CFG scale")
            seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
            generate_button = gr.Button("Generate", variant="primary")
        with gr.Column():
            output_image = gr.Image(label="Result", type="pil")
            status = gr.Markdown()

    gr.Examples(
        examples=[
            ["Turn the subject's outfit into a futuristic silver fashion design, preserve the pose and face", "", 30, 4.0, -1],
            ["Place the subject in a warm cinematic sunset beach scene, realistic photography", "", 30, 4.0, -1],
        ],
        inputs=[prompt, negative_prompt, steps, cfg, seed],
        label="Prompt examples",
    )

    generate_button.click(
        fn=generate,
        inputs=[input_image, prompt, negative_prompt, steps, cfg, seed],
        outputs=[output_image, status],
        api_name="generate",
    )

    gr.Markdown("Model weights and licenses: [Qwen Image Edit](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) · [NSFW LoRA](https://huggingface.co/ScottzillaSystems/qwen-image-edit-plus-nsfw-lora)")


demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", "7860")),
    mcp_server=True,
)

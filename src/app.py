import random
import os
import uuid
from datetime import datetime
import gradio as gr
import numpy as np
import spaces
import torch
from diffusers import DiffusionPipeline
from PIL import Image

# Create permanent storage directory
SAVE_DIR = "saved_images"  # Gradio will handle the persistence
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)

# Load the default image
DEFAULT_IMAGE_PATH = "cover1.webp"

device = "cuda" if torch.cuda.is_available() else "cpu"
repo_id = "black-forest-labs/FLUX.1-dev"
adapter_id = "prithivMLmods/EBook-Creative-Cover-Flux-LoRA"

pipeline = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.bfloat16)
pipeline.load_lora_weights(adapter_id)
pipeline = pipeline.to(device)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

def save_generated_image(image, prompt):
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{unique_id}.png"
    filepath = os.path.join(SAVE_DIR, filename)
    
    # Save the image
    image.save(filepath)
    
    # Save metadata
    metadata_file = os.path.join(SAVE_DIR, "metadata.txt")
    with open(metadata_file, "a", encoding="utf-8") as f:
        f.write(f"{filename}|{prompt}|{timestamp}\n")
    
    return filepath

def load_generated_images():
    if not os.path.exists(SAVE_DIR):
        return []
    
    # Load all images from the directory
    image_files = [os.path.join(SAVE_DIR, f) for f in os.listdir(SAVE_DIR) 
                  if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    # Sort by creation time (newest first)
    image_files.sort(key=lambda x: os.path.getctime(x), reverse=True)
    return image_files

def load_predefined_images():
    # Return empty list since we're not using predefined images
    return []

@spaces.GPU(duration=120)
def inference(
    prompt: str,
    seed: int,
    randomize_seed: bool,
    width: int,
    height: int,
    guidance_scale: float,
    num_inference_steps: int,
    lora_scale: float,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)
    
    image = pipeline(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=generator,
        joint_attention_kwargs={"scale": lora_scale},
    ).images[0]
    
    # Save the generated image
    filepath = save_generated_image(image, prompt)
    
    # Return the image, seed, and updated gallery
    return image, seed, load_generated_images()

examples = [
   "A haunting cathedral ruins bathed in ethereal moonlight, with ancient stone archways stretching toward a starlit sky. The title 'WHISPERS OF ETERNITY' appears in weathered silver lettering that seems to float between the pillars. Ghostly wisps of fog curl around crumbling gothic sculptures, while 'By Alexander Blackwood' is inscribed in elegant script that glows with a subtle blue luminescence. Delicate patterns of celestial symbols and arcane runes border the edges. [trigger]",
   
   "A massive ancient tree with crystalline leaves dominates the composition, its translucent branches reaching across a sunset sky streaked with impossible colors. 'THE LUMINOUS Crown' is written in intricate golden calligraphy that intertwines with the branches. Mysterious glowing orbs float among the leaves, casting prismatic light. 'By Isabella Moonshadow' appears to be carved into the tree's bark. Sacred geometry patterns shimmer in the background. [trigger]",
   
   "A dramatic spiral staircase made of weathered copper and stained glass descends into swirling cosmic depths. The title 'CHRONICLES OF THE INFINITE' spans the spiral in bold art deco typography that seems to be crafted from constellations. Nebulae and galaxies swirl in the background, while 'By Marcus Starweaver' appears to be formed from falling stardust. Complex mechanical clockwork elements frame the corners. [trigger]",
   
   "An intricate doorway carved from ancient jade stands solitary in a field of shimmering black sand. 'GATES OF THE IMMORTAL' is emblazoned across the top in powerful metallic letters that seem to be forged from liquid mercury. Ethereal phoenix feathers drift across the scene, leaving trails of golden light. 'By Victoria Jade' flows along the bottom in brushstrokes that resemble living smoke. Sacred Chinese characters appear to float in the background. [trigger]",
   
   "A magnificent underwater city of pearl and coral rises from abyssal depths, illuminated by bioluminescent sea life. 'DEPTHS OF WONDER' ripples across the scene in iridescent letters that appear to be formed from living water. Schools of ethereal fish create flowing patterns of light, while 'By Neptune Rivers' shimmers like mother-of-pearl below. Ancient Atlantean symbols pulse with a subtle aqua glow around the borders. [trigger]",
   
   "A colossal steampunk clocktower pierces through storm clouds, its gears and mechanisms visible through crystalline walls. 'TIMEKEEPER'S LEGACY' is constructed from intricate brass and copper mechanisms that appear to be in constant motion. Lightning arcs between copper spires, while 'By Theodore Cogsworth' is etched in burnished bronze below. Mathematical equations and alchemical symbols float in the turbulent sky. [trigger]"
]

css = """
footer {
    visibility: hidden;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css, analytics_enabled=False) as demo:
    gr.HTML('<div class="title"> eBOOK Cover generation </div>')
    
    gr.HTML("""<a href="https://visitorbadge.io/status?path=https%3A%2F%2Fginigen-Book-Cover.hf.space">
               <img src="https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fginigen-Book-Cover.hf.space&countColor=%23263759" />
               </a>""")
    
    with gr.Tabs() as tabs:
        with gr.Tab("Generation"):
            with gr.Column(elem_id="col-container"):
                with gr.Row():
                    prompt = gr.Text(
                        label="Prompt",
                        show_label=False,
                        max_lines=1,
                        placeholder="Enter your prompt",
                        container=False,
                    )
                    run_button = gr.Button("Run", scale=0)

                # Modified to include the default image
                result = gr.Image(
                    label="Result",
                    show_label=False,
                    value=DEFAULT_IMAGE_PATH  # Set the default image
                )

                with gr.Accordion("Advanced Settings", open=False):
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=42,
                    )
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

                    with gr.Row():
                        width = gr.Slider(
                            label="Width",
                            minimum=256,
                            maximum=MAX_IMAGE_SIZE,
                            step=32,
                            value=768,
                        )
                        height = gr.Slider(
                            label="Height",
                            minimum=256,
                            maximum=MAX_IMAGE_SIZE,
                            step=32,
                            value=1024,
                        )

                    with gr.Row():
                        guidance_scale = gr.Slider(
                            label="Guidance scale",
                            minimum=0.0,
                            maximum=10.0,
                            step=0.1,
                            value=3.5,
                        )
                        num_inference_steps = gr.Slider(
                            label="Number of inference steps",
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=30,
                        )
                        lora_scale = gr.Slider(
                            label="LoRA scale",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=1.0,
                        )

                gr.Examples(
                    examples=examples,
                    inputs=[prompt],
                    outputs=[result, seed],
                )

        with gr.Tab("Gallery"):
            gallery_header = gr.Markdown("### Generated Images Gallery")
            generated_gallery = gr.Gallery(
                label="Generated Images",
                columns=6,
                show_label=False,
                value=load_generated_images(),
                elem_id="generated_gallery",
                height="auto"
            )
            refresh_btn = gr.Button("🔄 Refresh Gallery")

    # Event handlers
    def refresh_gallery():
        return load_generated_images()

    refresh_btn.click(
        fn=refresh_gallery,
        inputs=None,
        outputs=generated_gallery,
    )

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=inference,
        inputs=[
            prompt,
            seed,
            randomize_seed,
            width,
            height,
            guidance_scale,
            num_inference_steps,
            lora_scale,
        ],
        outputs=[result, seed, generated_gallery],
    )

demo.queue()
demo.launch()
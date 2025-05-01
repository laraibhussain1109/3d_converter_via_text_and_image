import torch
from diffusers import ShapEPipeline
from diffusers.utils import export_to_obj  # or export_to_ply

def generate_3d_from_text(
    prompt: str,
    output_path: str = "output/model_from_text.obj"
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the one-line Shap-E pipeline
    pipe = ShapEPipeline.from_pretrained(
        "openai/shap-e",
        torch_dtype=torch.float32
    ).to(device)

    print(f"🔄 Generating 3D mesh for prompt: {prompt!r}")
    result = pipe(
        prompt,
        guidance_scale=15.0,
        num_inference_steps=64,
        frame_size=256,
        output_type="mesh",           # ← request mesh output
    )

    # The generated mesh is now in result.images[0]:
    mesh = result.images[0]

    # Export to your chosen format:
    export_to_obj(mesh, output_path)  # writes an .obj file
    # — or, to get a .ply:
    # ply_path = export_to_ply(mesh, output_path.replace(".obj", ".ply"))

    print(f"✅ Saved mesh → {output_path}")

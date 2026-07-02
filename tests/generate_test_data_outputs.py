
import json
import gc
from pathlib import Path

import torch
from transformers.models.sam3.modeling_sam3 import Sam3Model
from transformers import Sam3Processor
from PIL import Image

from utils_for_tests import annotate_image, model_outputs_to_annotations, model_inputs_for_prompts, x1y1x2y2_to_xywh


def perform_inference_for_prompt_file(
        prompt_path: str | Path,
):
    with open(prompt_path, "r") as f:
            prompt_data = json.load(f)
        

    image_name = prompt_data["image_name"]
    print("Processing prompts for image:", image_name)
    image_path = images_dir / image_name
    if not image_path.exists():
        print(f"Image file {image_name} not found. Skipping.")
        return

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_outputs = {
        "image_name": image_name,
        "tests": [],
    }

    for prompts_set in prompt_data["tests"]:
        text, input_boxes, input_boxes_labels = model_inputs_for_prompts(prompts_set["prompts"])

        with torch.inference_mode():
            inputs = processor(
                images=image,
                text=text,
                input_boxes=input_boxes,
                input_boxes_labels=input_boxes_labels,
                return_tensors="pt",
            ).to(device)
            raw_outputs = model(**inputs)
            outputs = processor.post_process_instance_segmentation(
                raw_outputs,
                target_sizes=[image.size[::-1]],
            )
        expected_outputs = model_outputs_to_annotations(outputs)
        image_outputs["tests"].append(
            {
                "id": prompts_set["id"],
                "expected_outputs": expected_outputs,
            }
        )
        annotated_images_dir = Path(__file__).parent / "data/torch_annotated_images"
        annotated_images_dir.mkdir(parents=True, exist_ok=True)
        annotate_image(
            image_path=image_path,
            prompts=prompts_set["prompts"],
            annotations=expected_outputs,
            save_path=annotated_images_dir / f"{prompt_path.stem.replace('_prompts', '')}_{prompts_set['id']}_annotated.jpg",
            prompt_id=prompts_set["id"],
        )

        del inputs, raw_outputs, outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        

    output_name = prompt_path.stem.replace("_prompts", "") + "_outputs.json"
    output_path = outputs_dir / output_name
    with open(output_path, "w") as f:
        json.dump(image_outputs, f, indent=4)
    print(f"Saved outputs to {output_path}")

    
if __name__ == "__main__":
    print("Gathering test data...")
    images_dir = Path(__file__).parent / "data/images"
    prompts_dir = Path(__file__).parent / "data/prompts"
    outputs_dir = Path(__file__).parent / "data/torch_outputs"
    

    prompt_paths = [p.resolve() for p in prompts_dir.iterdir() if p.suffix == ".json" and p.name != "prompt_template.json"]
    print(f"Found {len(prompt_paths)} prompt files.")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("Loading SAM3 model...")
    model = Sam3Model.from_pretrained("facebook/sam3", device_map=device).eval()
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    for prompt_path in prompt_paths:
        perform_inference_for_prompt_file(prompt_path)


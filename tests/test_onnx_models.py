"""
These are non-traditional tests (not unit tests) that run inference on test data to compare the quality of outputs
between the original PyTorch model and the ONNX exported models. 
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "exportsam3"))

from exportsam3.onnx_inference import Sam3ONNXInference


from exportsam3.utils import annotate_image, model_inputs_for_prompts, model_outputs_to_annotations


def test_onnx_inference_results() -> None:
	models_dir = PROJECT_ROOT / "outputs" / "onnx"
	data_dir = PROJECT_ROOT / "tests" / "data"
	prompts_dir = data_dir / "prompts"
	images_dir = data_dir / "images"
	onnx_outputs_dir = data_dir / "onnx_outputs"
	onnx_annotated_images_dir = data_dir / "onnx_annotated_images"

	image_encoder_path = models_dir / "sam3_image_encoder.onnx"
	text_encoder_path = models_dir / "sam3_text_encoder.onnx"
	decoder_path = models_dir / "sam3_decoder.onnx"
	bpe_path = PROJECT_ROOT / "exportsam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"

	assert image_encoder_path.exists()
	assert text_encoder_path.exists()
	assert decoder_path.exists()
	assert bpe_path.exists()

	onnx_outputs_dir.mkdir(parents=True, exist_ok=True)
	onnx_annotated_images_dir.mkdir(parents=True, exist_ok=True)

	providers = ort.get_available_providers()
	device = "cuda" if "CUDAExecutionProvider" in providers else "cpu"
	engine = Sam3ONNXInference(
		vision_encoder_path=str(image_encoder_path),
		text_encoder_path=str(text_encoder_path),
		decoder_path=str(decoder_path),
		bpe_path=str(bpe_path),
		device=device,
	)

	prompt_paths = sorted(
		[
			p
			for p in prompts_dir.iterdir()
			if p.suffix == ".json" and p.name != "prompt_template.json"
		]
	)
	assert len(prompt_paths) > 0

	for prompt_path in prompt_paths:
		with open(prompt_path, "r") as f:
			prompt_data = json.load(f)

		image_name = prompt_data["image_name"]
		image_path = images_dir / image_name
		assert image_path.exists()

		image_np = np.array(Image.open(image_path).convert("RGB"))
		file_outputs = {"image_name": image_name, "tests": []}

		for prompts_set in prompt_data["tests"]:
			
			prompts = prompts_set["prompts"]
			text, boxes, box_labels = model_inputs_for_prompts(prompts)
           
			if boxes is not None and len(boxes) > 0 and isinstance(boxes[0], list):
				# test_utils returns nested structure for processor-style inputs.
				if len(boxes) == 1 and len(boxes[0]) > 0 and isinstance(boxes[0][0], list):
					boxes = boxes[0]
					#convert boxes from x1y1x2y2 to xywh format
					boxes = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]
			if box_labels is not None and len(box_labels) > 0 and isinstance(box_labels[0], list):
				if len(box_labels) == 1 and len(box_labels[0]) > 0:
					box_labels = box_labels[0]

			results = engine.predict(
				image=image_np,
				text=text,
				boxes=boxes,
				box_labels=box_labels,
				conf_threshold=0.3,
			)

			expected_outputs = model_outputs_to_annotations([results])
			file_outputs["tests"].append(
				{
					"id": prompts_set["id"],
					"expected_outputs": expected_outputs,
				}
			)

			annotated_name = (
				f"{Path(image_name).stem}_{prompts_set['id']}_onnx_annotated.jpg"
			)
			annotate_image(
				image_path=image_path,
				prompts=prompts,
				annotations=expected_outputs,
				save_path=onnx_annotated_images_dir / annotated_name,
				prompt_id=prompts_set["id"],
			)

		output_name = prompt_path.stem.replace("_prompts", "") + "_outputs.json"
		output_path = onnx_outputs_dir / output_name
		with open(output_path, "w") as f:
			json.dump(file_outputs, f, indent=4)

		assert output_path.exists()

def test_onnx_mixed_inference_results() -> None:
	models_dir = PROJECT_ROOT / "outputs" / "onnx_mixed"
	data_dir = PROJECT_ROOT / "tests" / "data"
	prompts_dir = data_dir / "prompts"
	images_dir = data_dir / "images"
	onnx_outputs_dir = data_dir / "onnx_mixed_outputs"
	onnx_annotated_images_dir = data_dir / "onnx_mixed_annotated_images"

	image_encoder_path = models_dir / "sam3_image_encoder_mixed.onnx"
	text_encoder_path = models_dir / "sam3_text_encoder_mixed.onnx"
	decoder_path = models_dir / "sam3_decoder_mixed.onnx"
	bpe_path = PROJECT_ROOT / "exportsam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"

	assert image_encoder_path.exists()
	assert text_encoder_path.exists()
	assert decoder_path.exists()
	assert bpe_path.exists()

	onnx_outputs_dir.mkdir(parents=True, exist_ok=True)
	onnx_annotated_images_dir.mkdir(parents=True, exist_ok=True)

	providers = ort.get_available_providers()
	device = "cuda" if "CUDAExecutionProvider" in providers else "cpu"
	engine = Sam3ONNXInference(
		vision_encoder_path=str(image_encoder_path),
		text_encoder_path=str(text_encoder_path),
		decoder_path=str(decoder_path),
		bpe_path=str(bpe_path),
		device=device,
	)

	prompt_paths = sorted(
		[
			p
			for p in prompts_dir.iterdir()
			if p.suffix == ".json" and p.name != "prompt_template.json"
		]
	)
	assert len(prompt_paths) > 0

	for prompt_path in prompt_paths:
		with open(prompt_path, "r") as f:
			prompt_data = json.load(f)

		image_name = prompt_data["image_name"]
		image_path = images_dir / image_name
		assert image_path.exists()

		image_np = np.array(Image.open(image_path).convert("RGB"))
		file_outputs = {"image_name": image_name, "tests": []}

		for prompts_set in prompt_data["tests"]:
			
			prompts = prompts_set["prompts"]
			text, boxes, box_labels = model_inputs_for_prompts(prompts)
           
			if boxes is not None and len(boxes) > 0 and isinstance(boxes[0], list):
				# test_utils returns nested structure for processor-style inputs.
				if len(boxes) == 1 and len(boxes[0]) > 0 and isinstance(boxes[0][0], list):
					boxes = boxes[0]
					#convert boxes from x1y1x2y2 to xywh format
					boxes = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]
			if box_labels is not None and len(box_labels) > 0 and isinstance(box_labels[0], list):
				if len(box_labels) == 1 and len(box_labels[0]) > 0:
					box_labels = box_labels[0]

			results = engine.predict(
				image=image_np,
				text=text,
				boxes=boxes,
				box_labels=box_labels,
				conf_threshold=0.3,
			)

			expected_outputs = model_outputs_to_annotations([results])
			file_outputs["tests"].append(
				{
					"id": prompts_set["id"],
					"expected_outputs": expected_outputs,
				}
			)

			annotated_name = (
				f"{Path(image_name).stem}_{prompts_set['id']}_onnx_annotated.jpg"
			)
			annotate_image(
				image_path=image_path,
				prompts=prompts,
				annotations=expected_outputs,
				save_path=onnx_annotated_images_dir / annotated_name,
				prompt_id=prompts_set["id"],
			)

		output_name = prompt_path.stem.replace("_prompts", "") + "_outputs.json"
		output_path = onnx_outputs_dir / output_name
		with open(output_path, "w") as f:
			json.dump(file_outputs, f, indent=4)

		assert output_path.exists()


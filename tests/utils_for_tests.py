from pathlib import Path

from PIL import Image
import numpy as np
from pycocotools import mask as mask_utils
import torch

def model_inputs_for_prompts(prompts: list[dict]) -> tuple:
    text = None
    input_boxes = None
    input_boxes_labels = None

    for prompt in prompts:
        if prompt["type"] == "text":
            text = prompt["data"]
        elif prompt["type"] == "box":
            if input_boxes is None:
                input_boxes = [[]]
            input_boxes[0].append(list(map(float, prompt["data"])))
            if input_boxes_labels is None:
                input_boxes_labels = [[]]
            input_boxes_labels[0].append(int(prompt.get("label", 1)))

    return text, input_boxes, input_boxes_labels

def mask_to_compressed_rle(mask: np.ndarray) -> dict:
    binary_mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(binary_mask)
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return {
        "size": rle["size"],
        "counts": rle["counts"],
    }


def x1y1x2y2_to_xywh(box: list[float]) -> list[float]:
    if len(box) != 4:
        raise ValueError("Box must have 4 elements: [x1, y1, x2, y2]")
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    return [x1, y1, width, height]


def model_outputs_to_annotations(outputs: list[dict]) -> list[dict]:
    if not isinstance(outputs, list) or len(outputs) == 0:
        return []

    first = outputs[0]
    if not isinstance(first, dict):
        return []

    masks = first.get("masks")
    scores = first.get("scores")
    boxes = first.get("boxes")
    if masks is None:
        return []

    if isinstance(masks, torch.Tensor):
        masks_np = masks.detach().cpu().numpy()
    else:
        masks_np = np.asarray(masks)

    if isinstance(scores, torch.Tensor):
        scores_np = scores.detach().cpu().numpy()
    elif scores is None:
        scores_np = np.array([], dtype=np.float32)
    else:
        scores_np = np.asarray(scores)

    if isinstance(boxes, torch.Tensor):
        boxes_np = boxes.detach().cpu().numpy()
    elif boxes is None:
        boxes_np = np.array([], dtype=np.float32)
    else:
        boxes_np = np.asarray(boxes)

    annotations: list[dict] = []
    for idx in range(len(masks_np)):
        mask_np = np.asarray(masks_np[idx])
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        binary_mask = mask_np > 0.5

        rle_for_geometry = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
        area = float(mask_utils.area(rle_for_geometry))
        if area <= 0:
            continue

        if idx < len(boxes_np):
            bbox = x1y1x2y2_to_xywh([float(v) for v in np.asarray(boxes_np[idx]).tolist()])
        else:
            bbox = [float(v) for v in mask_utils.toBbox(rle_for_geometry).tolist()]

        score = float(scores_np[idx]) if idx < len(scores_np) else 1.0
        compressed_rle = mask_to_compressed_rle(binary_mask)

        annotations.append(
            {
                "id": idx,
                "type": "mask",
                "bbox": bbox,
                "area": area,
                "segmentation": compressed_rle,
                "score": score,
            }
        )

    return annotations



def annotate_image(
    image_path: str | Path,
    prompts: list[dict],
    annotations: list[dict],
    save_path: str | Path,
    prompt_id: str | None = None,
) -> None:
    from PIL import ImageDraw, ImageFont

    image_rgba = Image.open(image_path).convert("RGBA")
    image_np = np.array(image_rgba, dtype=np.uint8)
    h, w = image_np.shape[:2]

    palette = [
        (255, 99, 71),
        (65, 105, 225),
        (50, 205, 50),
        (255, 165, 0),
        (186, 85, 211),
        (64, 224, 208),
    ]

    for idx, ann in enumerate(annotations):
        color = palette[idx % len(palette)]
        seg = ann.get("segmentation")
        if isinstance(seg, dict) and "counts" in seg and "size" in seg:
            decoded = mask_utils.decode(seg)
            if decoded.ndim == 3:
                decoded = decoded[:, :, 0]
            mask = decoded.astype(bool)
            alpha = 0.45
            image_np[mask, 0] = (1.0 - alpha) * image_np[mask, 0] + alpha * color[0]
            image_np[mask, 1] = (1.0 - alpha) * image_np[mask, 1] + alpha * color[1]
            image_np[mask, 2] = (1.0 - alpha) * image_np[mask, 2] + alpha * color[2]

    annotated = Image.fromarray(image_np, mode="RGBA")
    draw = ImageDraw.Draw(annotated)

    def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size=size)
        except OSError:
            return ImageFont.load_default()

    title_font = _load_font(36)
    annotation_font = _load_font(12)

    prompt_summaries: list[str] = []
    for prompt in prompts:
        ptype = prompt.get("type")
        data = prompt.get("data", [])
        if ptype == "text":
            prompt_summaries.append(f"Text: {str(data)}")
        elif ptype in {"box", "rectangle"} and len(data) == 4:
            x1, y1, x2_or_w, y2_or_h = [float(v) for v in data]
            if x2_or_w > x1 and y2_or_h > y1:
                x2, y2 = x2_or_w, y2_or_h
            else:
                x2, y2 = x1 + x2_or_w, y1 + y2_or_h
            prompt_summaries.append(f"Box: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")

    if prompt_id:
        prompt_summaries.insert(0, f"Prompt ID: {prompt_id}")

    title = " | ".join(prompt_summaries) if prompt_summaries else "Prompt: (none)"

    title_pad_x = 8
    title_pad_y = 6
    title_box_h = draw.textbbox((0, 0), title, font=title_font)[3]
    draw.rectangle(
        [(0, 0), (w, title_box_h + title_pad_y * 2)],
        fill=(0, 0, 0, 170),
    )
    draw.text((title_pad_x, title_pad_y), title, fill=(255, 255, 255, 255), font=title_font)

    for prompt in prompts:
        ptype = prompt.get("type")
        data = prompt.get("data", [])
        if ptype in {"box", "rectangle"} and len(data) == 4:
            x1, y1, x2_or_w, y2_or_h = [float(v) for v in data]
            if x2_or_w > x1 and y2_or_h > y1:
                x2, y2 = x2_or_w, y2_or_h
            else:
                x2, y2 = x1 + x2_or_w, y1 + y2_or_h
            draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 255, 0, 255), width=2)

    for idx, ann in enumerate(annotations):
        bbox = ann.get("bbox", [])
        if len(bbox) == 4:
            x, y, _, _ = [float(v) for v in bbox]
            score = float(ann.get("score", 0.0))
            label = f"score={score:.3f}"
            draw.text((x + 2, max(0.0, y - 14.0)), label, fill=(255, 255, 255, 255), font=annotation_font)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    annotated.convert("RGB").save(save_path)

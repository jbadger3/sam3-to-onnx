# Model input and output shapes

This project exports three ONNX components that are run in sequence:

1. Image encoder
2. Text encoder
3. Decoder (includes geometry encoder)


Symbols used below:

- B: batch size (dynamic)
- H, W: model input image height and width (export-time fixed, default 1008 x 1008)
- N: number of box prompts (dynamic)
- Q: number of decoder queries (model-fixed)

Image encoder: outputs/onnx/sam3_image_encoder.onnx

- Input
	- images: float32, shape [B, 3, H, W]

- Output
	- fpn_feat_0: float32, shape [B, 256, 4 x (H/14), 4 x (W/14)]
	- fpn_feat_1: float32, shape [B, 256, 2 x (H/14), 2 x (W/14)]
	- fpn_feat_2: float32, shape [B, 256, H/14, W/14]
	- fpn_pos_2: float32, shape [B, 256, H/14, W/14]

With default export size H=W=1008:

- fpn_feat_0: [B, 256, 288, 288]
- fpn_feat_1: [B, 256, 144, 144]
- fpn_feat_2: [B, 256, 72, 72]
- fpn_pos_2: [B, 256, 72, 72]

Text encoder: outputs/onnx/sam3_text_encoder.onnx

- Input
	- input_ids: int64, shape [B, 32]
	- attention_mask: int64, shape [B, 32]

- Output
	- text_features: float32, shape [B, 32, 256]
	- text_mask: bool, shape [B, 32]

Decoder: outputs/onnx/sam3_decoder.onnx

- Input
	- fpn_feat_0: float32, shape [B, 256, 4 x (H/14), 4 x (W/14)]
	- fpn_feat_1: float32, shape [B, 256, 2 x (H/14), 2 x (W/14)]
	- fpn_feat_2: float32, shape [B, 256, H/14, W/14]
	- fpn_pos_2: float32, shape [B, 256, H/14, W/14]
	- text_features: float32, shape [B, 32, 256]
	- text_mask: bool, shape [B, 32]
	- input_boxes: float32, shape [B, N, 4]
	- input_boxes_labels: int64, shape [B, N]

Notes for decoder box inputs:

- input_boxes format is normalized cx, cy, w, h in [0, 1] relative to model input size.
- input_boxes_labels uses 1 for positive boxes, 0 for negative boxes, and -10 for no-box padding.

- Output
	- pred_masks: float32, shape [B, Q, mask_h, mask_w]
	- pred_boxes: float32, shape [B, Q, 4] in normalized xyxy format
	- pred_logits: float32, shape [B, Q]
	- presence_logits: float32, shape [B, 1]
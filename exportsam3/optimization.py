import argparse
from pathlib import Path

import onnx
from onnxruntime.transformers import optimizer


DEFAULT_PRECISION = "mixed"


def get_model_paths(models_dir: Path) -> dict[str, Path]:
    model_paths = {
        "image_encoder": models_dir / "sam3_image_encoder.onnx",
        "text_encoder": models_dir / "sam3_text_encoder.onnx",
        "decoder": models_dir / "sam3_decoder.onnx",
    }

    for model_path in model_paths.values():
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")

    return model_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize exported SAM3 ONNX models")
    parser.add_argument(
        "--precision",
        type=str,
        default=DEFAULT_PRECISION,
        help="Target quantization precision (default: mixed)",
    )
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=["image_encoder", "text_encoder", "decoder", "all"],
        default=["all"],
        help=(
            "Model parts to quantize. Use one or more of: image_encoder, "
            "text_encoder, decoder, or all. Decoder conversion is deferred."
        ),
    )
    parser.add_argument(
        "--use-external-data",
        action="store_true",
        default=False,
        help="Store model weights in external .data files.",
    )
    return parser.parse_args()


def convert_image_encoder_to_mixed_precision(
    model_path: Path,
    outputs_dir: Path,
    use_external_data: bool = False,
) -> None:
    print(f"Converting image_encoder to mixed precision with ORT optimizer: {model_path}")
    optimized_model = optimizer.optimize_model(
        str(model_path),
        model_type="vit",
        num_heads=16,
        hidden_size=1024,
        use_gpu=True,
    )
    optimized_model.convert_float_to_float16(
        keep_io_types=True,
        op_block_list=["Conv", "Cast"],
        #min_positive_val=5.96e-08,
    )

    save_path = outputs_dir / "sam3_image_encoder_mixed.onnx"
    optimized_model.save_model_to_file(
        str(save_path),
        use_external_data_format=use_external_data,
    )
    print(f"Saved mixed precision image encoder to: {save_path}")


def convert_text_encoder_to_mixed_precision(
    model_path: Path,
    outputs_dir: Path,
    use_external_data: bool = False,
) -> None:
    print(f"Converting text_encoder to mixed precision with ORT optimizer: {model_path}")
    optimized_model = optimizer.optimize_model(
        str(model_path),
        model_type="clip",
        use_gpu=True,
    )
    optimized_model.convert_float_to_float16(
        keep_io_types=True,
        #op_block_list=["Cast"],
        #min_positive_val=5.96e-08,
    )

    save_path = outputs_dir / "sam3_text_encoder_mixed.onnx"
    optimized_model.save_model_to_file(
        str(save_path),
        use_external_data_format=use_external_data,
    )
    print(f"Saved mixed precision text encoder to: {save_path}")


def convert_decoder_noop_to_mixed(
    model_path: Path,
    outputs_dir: Path,
    use_external_data: bool = False,
) -> None:
    print(f"Performing no-op conversion for decoder: {model_path}")
    model = onnx.load(str(model_path), load_external_data=True)

    save_path = outputs_dir / "sam3_decoder_mixed.onnx"
    onnx.save_model(
        model,
        str(save_path),
        save_as_external_data=use_external_data,
        all_tensors_to_one_file=use_external_data,
        location=f"{save_path.name}.data" if use_external_data else None,
        size_threshold=0,
        convert_attribute=False,
    )
    print(f"Saved no-op mixed decoder model to: {save_path}")


def convert_models_and_save(
    model_paths: dict[str, Path],
    outputs_dir: Path,
    precision: str,
    parts: list[str],
    use_external_data: bool = False,
) -> None:
    if precision != "mixed":
        raise ValueError(f"Unsupported precision: {precision}")

    selected_parts = set(parts)
    if "all" in selected_parts:
        selected_parts = {"image_encoder", "text_encoder", "decoder"}

    if "image_encoder" in selected_parts:
        convert_image_encoder_to_mixed_precision(
            model_paths["image_encoder"],
            outputs_dir,
            use_external_data=use_external_data,
        )

    if "text_encoder" in selected_parts:
        convert_text_encoder_to_mixed_precision(
            model_paths["text_encoder"],
            outputs_dir,
            use_external_data=use_external_data,
        )

    if "decoder" in selected_parts:
        convert_decoder_noop_to_mixed(
            model_paths["decoder"],
            outputs_dir,
            use_external_data=use_external_data,
        )

def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    base_outputs_dir = project_root / "outputs"
    models_dir = base_outputs_dir / "onnx"
    outputs_dir = base_outputs_dir / "onnx_mixed"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    model_paths = get_model_paths(models_dir)
    print(f"Validated {len(model_paths)} ONNX models from {models_dir}")
    print(f"Requested precision: {args.precision}")
    print(f"Requested parts: {args.parts}")
    print(f"Use external data: {args.use_external_data}")

    convert_models_and_save(
        model_paths,
        outputs_dir,
        args.precision,
        args.parts,
        use_external_data=args.use_external_data,
    )


if __name__ == "__main__":
    main()

"""Download pretrained RF-DETR Nano and export ONNX with inference metadata."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("models/rfdetr-nano"))
    args = parser.parse_args()
    from rfdetr import RFDETRNano

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = RFDETRNano(device="cpu")
    artifact = model.export(
        output_dir=str(args.output_dir), format="onnx", verbose=False
    )
    print(f"Exported {artifact}")


if __name__ == "__main__":
    main()

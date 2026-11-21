"""Download/cache the configured pretrained plate detector and OCR models."""

import argparse

from onnx_camera import load_config
from plates import create_alpr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/onnx.yml")
    args = parser.parse_args()
    config = load_config(args.config)
    create_alpr(config.lpr)
    print(f"Cached {config.lpr.detector_model} and {config.lpr.ocr_model}")


if __name__ == "__main__":
    main()

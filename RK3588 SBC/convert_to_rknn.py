#!/usr/bin/env python3
"""
Run on an x86 DEV MACHINE, not the RK3588 board -- rknn-toolkit2's conversion tooling only ships
x86_64 wheels. Converts an ONNX-exported model (e.g. `yolo export model=best.pt format=onnx
opset=12` from an ultralytics YOLOv8 mammography checkpoint) to Rockchip's `.rknn` format for
on-device NPU inference via rknn-toolkit-lite2 (see rknn_infer.py, which runs ON the board).

Install rknn-toolkit2 first (not a normal PyPI package -- pull the wheel matching your Python
version from https://github.com/airockchip/rknn-toolkit2/releases):
    pip install rknn-toolkit2

Usage:
    python convert_to_rknn.py --onnx best.onnx --output mammography.rknn --platform rk3588
"""
import argparse
import sys


def convert(onnx_path, output_path, platform, mean_values, std_values, quantize, dataset_txt):
    try:
        from rknn.api import RKNN
    except ImportError:
        print("rknn-toolkit2 not installed. On an x86_64 machine, run:\n"
              "  pip install rknn-toolkit2\n"
              "(wheel URL depends on your Python version -- see "
              "https://github.com/airockchip/rknn-toolkit2/releases)", file=sys.stderr)
        sys.exit(1)

    rknn = RKNN(verbose=True)

    print(f"[1/4] Configuring for platform={platform} ...")
    rknn.config(mean_values=[mean_values], std_values=[std_values], target_platform=platform)

    print(f"[2/4] Loading ONNX model: {onnx_path}")
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print("load_onnx failed", file=sys.stderr)
        sys.exit(1)

    print(f"[3/4] Building{' (INT8-quantized)' if quantize else ' (FP16)'} ...")
    if quantize and not dataset_txt:
        print("--quantize requires --dataset (a text file listing calibration image paths, one "
              "per line -- ~50-100 representative images from ../App/Models/Mammography/Data Set/ "
              "is a reasonable start)", file=sys.stderr)
        sys.exit(1)
    ret = rknn.build(do_quantization=quantize, dataset=dataset_txt if quantize else None)
    if ret != 0:
        print("build failed", file=sys.stderr)
        sys.exit(1)

    print(f"[4/4] Exporting: {output_path}")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print("export_rknn failed", file=sys.stderr)
        sys.exit(1)

    rknn.release()
    print(f"Done. Copy {output_path} to the RK3588 board and run rknn_infer.py there.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--onnx", required=True, help="Path to the ONNX-exported model")
    ap.add_argument("--output", required=True, help="Output .rknn path")
    ap.add_argument("--platform", default="rk3588", help="Target SoC (rk3588, rk3588s, ...)")
    ap.add_argument("--mean", type=float, default=0.0, help="Per-channel mean used at export/training time")
    ap.add_argument("--std", type=float, default=255.0, help="Per-channel std used at export/training time")
    ap.add_argument("--quantize", action="store_true",
                     help="INT8-quantize for extra NPU speed (needs --dataset); omit for FP16")
    ap.add_argument("--dataset", default=None, help="Text file of calibration image paths (INT8 only)")
    args = ap.parse_args()
    convert(args.onnx, args.output, args.platform, [args.mean] * 3, [args.std] * 3,
             args.quantize, args.dataset)


if __name__ == "__main__":
    main()

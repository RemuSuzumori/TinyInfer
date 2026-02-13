#!/usr/bin/env python3
import argparse
import struct


def main() -> None:
    parser = argparse.ArgumentParser(description="Build raw_tensor_mnist payload")
    parser.add_argument("--input", required=True, help="input .bin file with 784 float32 values")
    parser.add_argument("--output", required=True, help="output payload path")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        buf = f.read()
    if len(buf) != 784 * 4:
        raise ValueError(f"expected 3136 bytes, got {len(buf)}")

    struct.unpack("<784f", buf)
    with open(args.output, "wb") as f:
        f.write(buf)
    print(f"wrote {len(buf)} bytes -> {args.output}")


if __name__ == "__main__":
    main()

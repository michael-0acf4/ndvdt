import cv2
import numpy as np
import argparse
from tqdm import tqdm


def dvdt(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    # For animes, video might be at 24fps but actual
    # motion happens at 4, 6, 10fps
    dt = max((1 / fps), 0.5)

    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Cannot read first frame")
    prev_frame = prev_frame.astype(np.float32)

    h, w, c = prev_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(frame_count - 1), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame.astype(np.float32)
        dv = (frame - prev_frame) * dt
        out.write(np.clip(np.abs(dv), 0, 255).astype(np.uint8))

        prev_frame = frame

    cap.release()
    out.release()
    print(f"Result at {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute the full derivative of a video"
    )
    parser.add_argument("input", help="Input path")
    parser.add_argument("output", help="Output path")
    args = parser.parse_args()

    dvdt(args.input, args.output)


if __name__ == "__main__":
    main()

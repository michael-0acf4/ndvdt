import cv2
import numpy as np
import argparse
from tqdm import tqdm


def half_derivative(prev_frames, current_frame, weights=[1.0, -0.25, 0.125, -0.0625]):
    dc = np.zeros_like(current_frame, dtype=np.float32)
    for k, w in enumerate(weights):
        if k == 0:
            dc += w * current_frame
        elif k <= len(prev_frames):
            dc += w * prev_frames[-k]
    return dc


def half_dvdt(input_path, output_path):
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

    prev_frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(frame_count - 1), desc="Processing"):
        ret, next_frame = cap.read()
        if not ret:
            break
        next_frame = next_frame.astype(np.float32)
        dc = half_derivative(prev_frames, next_frame)
        out.write(np.clip(np.abs(dc), 0, 255).astype(np.uint8))

        prev_frames.append(next_frame.copy())
        if len(prev_frames) > 4:
            prev_frames.pop(0)

    cap.release()
    out.release()
    print(f"Saved half-derivative video (no audio) to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute the derivative of a video"
    )
    parser.add_argument("input", help="Input path")
    parser.add_argument("output", help="Output path")
    args = parser.parse_args()

    half_dvdt(args.input, args.output)


if __name__ == "__main__":
    main()

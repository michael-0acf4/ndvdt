import cv2
import numpy as np
import argparse
from tqdm import tqdm


def nth_grunwald_letkinov(frame, prev_frames, dt, alpha, max_history):
    dv = np.zeros_like(frame, dtype=np.float32)
    n_history = min(len(prev_frames), max_history)

    for k in range(n_history + 1):
        if k == 0:
            binom = 1.0
        else:
            binom *= (alpha - (k - 1)) / k
        sign = (-1) ** k
        if k == 0:
            dv += sign * binom * frame
        else:
            dv += sign * binom * prev_frames[-k]

    dv /= dt ** alpha

    return dv


def ndvdt(input_path, output_path, alpha=0.5, max_history=24):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    # For animes, video might be at 24fps but actual
    # motion happens at 4, 6, 10fps
    dt = max((1 / fps), 0.5)

    ret, frame = cap.read()
    if not ret:
        raise ValueError("Cannot read first frame")
    frame = frame.astype(np.float32)

    h, w, c = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    prev_frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(frame_count - 1), desc="Processing"):
        ret, next_frame = cap.read()
        if not ret:
            break
        next_frame = next_frame.astype(np.float32)

        # dt is irrelevant but keeping it to make things clearer
        dv = nth_grunwald_letkinov(
            next_frame, prev_frames, dt, alpha=alpha, max_history=max_history
        ) * (dt**alpha)

        out.write(np.clip(np.abs(dv), 0, 255).astype(np.uint8))

        prev_frames.append(next_frame.copy())
        if len(prev_frames) > max_history:
            prev_frames.pop(0)

    cap.release()
    out.release()
    print(f"Result (no audio) at {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute the nth-derivative of a video"
    )
    parser.add_argument(
        "--nth",
        "--alpha",
        dest="alpha",
        type=float,
        default=0.5,
        help="Derivative order (default: 0.5)",
    )
    parser.add_argument(
        "--history", type=int, default=24, help="Max history length (default: 24)"
    )
    parser.add_argument("input", help="Input path")
    parser.add_argument("output", help="Output path")
    args = parser.parse_args()

    ndvdt(args.input, args.output, alpha=args.alpha, max_history=args.history)


if __name__ == "__main__":
    main()

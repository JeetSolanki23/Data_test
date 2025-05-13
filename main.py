import cv2
import hashlib
import numpy as np
import json
import os
from tqdm import tqdm

def sha256_frame_hash(frame):
    return hashlib.sha256(frame.tobytes()).hexdigest()

def compute_motion_vector(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                         None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))

def detect_scene_change(prev_frame, curr_frame, threshold=0.4):
    diff = cv2.absdiff(prev_frame, curr_frame)
    diff_score = np.sum(diff) / diff.size / 255
    return diff_score > threshold

def extract_metadata(video_path, output_json="video_metadata.json"):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    metadata = []

    ret, prev_frame = cap.read()
    if not ret:
        print("Cannot read video.")
        return

    frame_number = 0
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_hash = sha256_frame_hash(prev_frame)

    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing Frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hash_val = sha256_frame_hash(frame)
        motion_score = compute_motion_vector(prev_gray, gray)
        scene_change = detect_scene_change(prev_gray, gray)

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        metadata.append({
            "frame_number": frame_number,
            "timestamp": round(timestamp, 3),
            "hash": hash_val,
            "motion_score": round(motion_score, 4),
            "scene_change": bool(scene_change)
        })

        prev_gray = gray
        prev_frame = frame
        frame_number += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    with open(output_json, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {output_json}")

# Example usage
if __name__ == "__main__":
    extract_metadata("n4.mp4")

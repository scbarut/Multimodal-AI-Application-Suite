import os
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

def extract_frames(video_path, interval_sec=3, output_dir="frames"):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_sec)
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            filename = os.path.join(output_dir, f"frame_{saved:03}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1
        count += 1
    cap.release()
    return [os.path.join(output_dir, f) for f in sorted(os.listdir(output_dir))]

def clip_similarity(img_paths, text="A high-quality product image"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    scores = []
    for path in img_paths:
        image = Image.open(path).convert("RGB")
        inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
        outputs = clip_model(**inputs)
        score = outputs.logits_per_image.item()
        scores.append((path, score))
    return sorted(scores, key=lambda x: x[1], reverse=True)

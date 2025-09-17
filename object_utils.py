import torch
import requests
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

from segment_anything import SamPredictor, sam_model_registry


def detect_and_segment_object(image_input, query_text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # GroundingDINO model
    grounding_model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(grounding_model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(device)
    
    try:
        if isinstance(image_input, str):
            if image_input.startswith(("http://", "https://")):
                response = requests.get(image_input, timeout=10)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("Geçersiz görüntü formatı. Dosya yolu, URL veya PIL.Image bekleniyor.")
    except Exception as e:
        print(f"Görüntü yükleme hatası: {e}")
        return None

    inputs = processor(images=image, text=query_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]
    )[0]

    if results["boxes"].size(0) == 0:
        print("Belirtilen nesne bulunamadı.")
        return None

    best_idx = torch.argmax(results["scores"])
    box = results["boxes"][best_idx].tolist()
    x0, y0, x1, y1 = map(int, box)

    return image, x0, y0, x1, y1


def sam_integration_method(image, x0, y0, x1, y1):
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    
    try:
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        predictor = SamPredictor(sam)

        image_np = np.array(image)
        predictor.set_image(image_np)

        input_box = np.array([x0, y0, x1, y1])
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        mask = masks[0]
        pil_mask = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")

        result = Image.new("RGBA", image.size, (0, 0, 0, 0))
        result.paste(image.convert("RGBA"), mask=pil_mask)

        return result.crop((x0, y0, x1, y1))
    
    except Exception as e:
        print(f"SAM hatası: {e}")
        return None


def place_object_optimally(object_image, background_image, query_text):
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64")
    clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64")
    clipseg_model.eval()

    inputs = processor(
        text=[f"a photo of {query_text}"],
        images=background_image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = clipseg_model(**inputs)
    heatmap = outputs.logits.squeeze().cpu().numpy()

    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    x = int(x * background_image.width / heatmap.shape[1])
    y = int(y * background_image.height / heatmap.shape[0])

    bg_cv = cv2.cvtColor(np.array(background_image), cv2.COLOR_RGB2BGR)
    fg_cv = cv2.cvtColor(np.array(object_image), cv2.COLOR_RGBA2BGRA)
    bg_area = bg_cv.shape[0] * bg_cv.shape[1]
    fg_area = fg_cv.shape[0] * fg_cv.shape[1]
    target_area = bg_area * 0.25
    scale = np.sqrt(target_area / fg_area)
    scale = min(max(scale, 0.1), 1.0)
    fg_resized = cv2.resize(fg_cv, (int(fg_cv.shape[1] * scale), int(fg_cv.shape[0] * scale)))

    x = max(0, min(x, bg_cv.shape[1] - fg_resized.shape[1]))
    y = max(0, min(y, bg_cv.shape[0] - fg_resized.shape[0]))

    alpha = cv2.merge([fg_resized[:, :, 3] / 255.0] * 3)
    foreground_part = fg_resized[:, :, :3]
    background_part = bg_cv[y:y+fg_resized.shape[0], x:x+fg_resized.shape[1]]
    blended = cv2.convertScaleAbs(foreground_part * alpha + background_part * (1 - alpha))
    bg_cv[y:y+fg_resized.shape[0], x:x+fg_resized.shape[1]] = blended

    result = cv2.cvtColor(bg_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result)

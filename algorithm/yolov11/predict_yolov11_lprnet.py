import os
import sys
import cv2
import argparse
import torch
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from yolo_utils import YOLOPlateDetector
from model.LPRNet import build_lprnet
from data.load_data import CHARS
from demo_integrated_lpr import greedy_decode, load_lprnet_model, recognize_plate


def draw_chinese_text(img, text, position, font_size=28, color=(255, 0, 0)):
    """Draw Chinese text on image using PIL."""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font_path = os.path.join(os.path.dirname(__file__), 'data', 'NotoSansCJK-Regular.ttc')
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        try:
            font = ImageFont.truetype("simhei.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def process_image(image_path, output_path, yolo_detector, lprnet, device):
    start_time = time.time()

    image = cv2.imread(image_path)
    if image is None:
        return

    plate_coords = yolo_detector.detect_plates(image)

    plate_texts = []
    detect_count = sum(1 for p in plate_coords if p[4] >= 0.5)

    for i, plate_info in enumerate(plate_coords):
        x1, y1, x2, y2, conf = plate_info

        h, w = image.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))

        if x2 <= x1 or y2 <= y1 or conf < 0.5:
            continue

        plate_image = image[y1:y2, x1:x2]
        plate_text = recognize_plate(lprnet, plate_image, device)
        plate_texts.append(f"{plate_text} -")

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{plate_text} ({conf:.2f})"
        image = draw_chinese_text(image, label, (x1, max(0, y1 - 32)))

    cv2.imwrite(output_path, image)

    processing_time = (time.time() - start_time) * 1000
    plates_str = " | ".join(plate_texts) if plate_texts else "-"
    filename = os.path.basename(image_path)
    # Format expected by parseYolo26Output in AnalyzeServiceImpl.java
    print(f"[1/1] {filename} | det={detect_count} | plates={plates_str} | time={processing_time:.1f}ms | save={output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True, help='Path to input image or directory')
    parser.add_argument('--output', required=True, help='Path to output directory')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu or cuda)')

    args = parser.parse_args()

    yolo_model_path = os.path.join(os.path.dirname(__file__), 'weights', 'best.pt')
    lpr_model_path = os.path.join(os.path.dirname(__file__), 'weights', 'Final_LPRNet_model.pth')

    device = 'cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu'

    yolo_detector = YOLOPlateDetector(yolo_model_path, conf_threshold=0.5, iou_threshold=0.45)
    lprnet = load_lprnet_model(lpr_model_path, device)

    image_path = args.image_path
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # Support both single file and directory input (backend passes a directory)
    if os.path.isdir(image_path):
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        images = [f for f in os.listdir(image_path)
                  if os.path.splitext(f)[1].lower() in exts]
        for img_name in images:
            src = os.path.join(image_path, img_name)
            dst = os.path.join(output_dir, img_name)
            process_image(src, dst, yolo_detector, lprnet, device)
    else:
        img_name = os.path.basename(image_path)
        dst = os.path.join(output_dir, img_name)
        process_image(image_path, dst, yolo_detector, lprnet, device)


if __name__ == '__main__':
    main()

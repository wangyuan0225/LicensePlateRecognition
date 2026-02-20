import argparse
import os
import time

import cv2
import numpy as np
import torch
from PIL import ImageFont
from ultralytics import YOLO

from fonts.cv_puttext import cv2ImgAddText
from plate_recognition.double_plate_split_merge import get_split_merge
from plate_recognition.plate_rec import get_plate_result, init_model


def collect_files(root_path):
    file_list = []
    for root, _, files in os.walk(root_path):
        for name in files:
            file_list.append(os.path.join(root, name))
    return sorted(file_list)


def four_point_transform(image, pts):
    rect = pts.astype(np.float32)
    (tl, tr, br, bl) = rect

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def load_model(weights, device):
    model = YOLO(weights)
    model.to(device)
    return model


def det_rec_plate(img_ori, detect_model, plate_rec_model, device, conf=0.3, iou=0.5):
    result_list = []
    results = detect_model(img_ori, conf=conf, iou=iou, verbose=False)

    for result in results:
        boxes = result.boxes
        keypoints = result.keypoints
        if len(boxes) == 0 or keypoints is None:
            continue

        kpts_xy = keypoints.xy
        num_det = min(len(boxes), len(kpts_xy))
        for idx in range(num_det):
            box = boxes.xyxy[idx].cpu().numpy()
            det_conf = float(boxes.conf[idx])
            plate_type = int(boxes.cls[idx])

            landmarks = kpts_xy[idx].cpu().numpy().astype(np.int64)
            roi_img = four_point_transform(img_ori, landmarks)
            if plate_type == 1:
                roi_img = get_split_merge(roi_img)

            plate_number, _, plate_color, color_conf = get_plate_result(
                roi_img, device, plate_rec_model, is_color=True
            )

            result_list.append(
                {
                    "plate_no": plate_number,
                    "plate_color": plate_color,
                    "rect": [int(v) for v in box],
                    "detect_conf": det_conf,
                    "landmarks": landmarks.tolist(),
                    "roi_height": roi_img.shape[0],
                    "color_conf": color_conf,
                    "plate_type": plate_type,  # 0: 单层, 1: 双层
                }
            )
    return result_list


def _clamp(value, low, high):
    return max(low, min(high, value))


def _normalize_rect(rect):
    if not rect or len(rect) < 4:
        return None
    x1 = int(round(float(rect[0])))
    y1 = int(round(float(rect[1])))
    x2 = int(round(float(rect[2])))
    y2 = int(round(float(rect[3])))
    left = min(x1, x2)
    top = min(y1, y2)
    right = max(x1, x2)
    bottom = max(y1, y2)
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def _get_plate_theme(plate_color):
    theme_map = {
        "蓝色": {"bg": (72, 33, 6), "border": (250, 165, 96), "text": (254, 242, 224), "glow": (246, 130, 59)},
        "黄色": {"bg": (0, 49, 74), "border": (21, 204, 250), "text": (195, 249, 254), "glow": (8, 179, 234)},
        "绿色": {"bg": (27, 53, 4), "border": (128, 222, 74), "text": (231, 252, 220), "glow": (94, 197, 34)},
        "白色": {"bg": (68, 50, 38), "border": (240, 232, 226), "text": (250, 250, 248), "glow": (184, 163, 148)},
        "黑色": {"bg": (20, 12, 8), "border": (184, 163, 148), "text": (240, 232, 226), "glow": (139, 116, 100)},
    }
    return theme_map.get(
        plate_color,
        {"bg": (43, 24, 8), "border": (248, 189, 56), "text": (255, 248, 232), "glow": (200, 140, 32)},
    )


def _draw_alpha_rect(img, x1, y1, x2, y2, color, alpha=0.75):
    h, w = img.shape[:2]
    x1 = _clamp(x1, 0, w - 1)
    y1 = _clamp(y1, 0, h - 1)
    x2 = _clamp(x2, 0, w)
    y2 = _clamp(y2, 0, h)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


def _measure_text(text, text_size=16):
    try:
        font = ImageFont.truetype("fonts/platech.ttf", text_size, encoding="utf-8")
        left, top, right, bottom = font.getbbox(text)
        return right - left, bottom - top
    except Exception:
        size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        return size[0], size[1] + baseline


def _draw_glow_border(img, x1, y1, x2, y2, border_color, glow_color):
    h, w = img.shape[:2]
    x1 = _clamp(x1, 0, w - 1)
    y1 = _clamp(y1, 0, h - 1)
    x2 = _clamp(x2, 0, w - 1)
    y2 = _clamp(y2, 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return
    glow_layer = np.zeros_like(img)
    cv2.rectangle(glow_layer, (x1, y1), (x2, y2), glow_color, 2)
    glow_layer = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=1.6, sigmaY=1.6)
    cv2.addWeighted(glow_layer, 0.45, img, 1.0, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 1)


def _draw_tech_box(img, x1, y1, x2, y2, border_color, glow_color, track_id=None):
    h, w = img.shape[:2]
    x1 = _clamp(x1, 0, w - 1)
    y1 = _clamp(y1, 0, h - 1)
    x2 = _clamp(x2, 0, w - 1)
    y2 = _clamp(y2, 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return

    bw = x2 - x1
    bh = y2 - y1
    diag = float(np.hypot(bw, bh))
    base_thick = _clamp(int(round(diag / 70.0)), 2, 5)
    glow_sigma = _clamp(diag / 55.0, 1.2, 3.6)

    glow_layer = np.zeros_like(img)
    cv2.rectangle(glow_layer, (x1, y1), (x2, y2), glow_color, max(1, base_thick - 1))
    glow_layer = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=glow_sigma, sigmaY=glow_sigma)
    cv2.addWeighted(glow_layer, 0.48, img, 1.0, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), border_color, max(1, base_thick - 1))

    corner_len = _clamp(int(round(min(bw, bh) * 0.28)), 8, 20)
    t = base_thick
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), border_color, t)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), border_color, t)
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), border_color, t)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), border_color, t)
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), border_color, t)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), border_color, t)
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), border_color, t)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), border_color, t)

    if track_id is not None:
        badge = "T%02d" % track_id
        badge_w_txt, badge_h_txt = _measure_text(badge, text_size=12)
        pad_x = 5
        pad_y = 3
        badge_w = badge_w_txt + pad_x * 2
        badge_h = badge_h_txt + pad_y * 2
        bx2 = _clamp(x2, badge_w + 2, w - 2)
        by1 = _clamp(y1 - badge_h - 2, 2, h - badge_h - 2)
        bx1 = bx2 - badge_w
        by2 = by1 + badge_h
        _draw_alpha_rect(img, bx1, by1, bx2, by2, (18, 18, 18), alpha=0.65)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), border_color, 1)
        text_x = bx1 + pad_x
        text_y = by1 + max(1, int((badge_h - badge_h_txt) / 2))
        img[:] = cv2ImgAddText(img, badge, text_x, text_y, border_color, 12)


def _draw_tech_landmark(img, x, y, border_color, glow_color):
    h, w = img.shape[:2]
    x = _clamp(int(round(x)), 0, w - 1)
    y = _clamp(int(round(y)), 0, h - 1)
    glow_layer = np.zeros_like(img)
    cv2.circle(glow_layer, (x, y), 5, glow_color, -1)
    glow_layer = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=1.2, sigmaY=1.2)
    cv2.addWeighted(glow_layer, 0.5, img, 1.0, 0, img)
    cv2.circle(img, (x, y), 2, border_color, -1)
    cv2.circle(img, (x, y), 4, border_color, 1)


def _plate_width_from_landmarks(landmarks, fallback_width):
    if not landmarks or len(landmarks) < 4:
        return float(fallback_width)
    try:
        p0 = np.array(landmarks[0], dtype=np.float32)
        p1 = np.array(landmarks[1], dtype=np.float32)
        p2 = np.array(landmarks[2], dtype=np.float32)
        p3 = np.array(landmarks[3], dtype=np.float32)
        top_w = float(np.linalg.norm(p1 - p0))
        bottom_w = float(np.linalg.norm(p2 - p3))
        width = (top_w + bottom_w) / 2.0
        if np.isfinite(width) and width > 1:
            return width
    except Exception:
        pass
    return float(fallback_width)


def _plate_height_from_landmarks(landmarks, fallback_height):
    if not landmarks or len(landmarks) < 4:
        return float(fallback_height)
    try:
        p0 = np.array(landmarks[0], dtype=np.float32)
        p1 = np.array(landmarks[1], dtype=np.float32)
        p2 = np.array(landmarks[2], dtype=np.float32)
        p3 = np.array(landmarks[3], dtype=np.float32)
        left_h = float(np.linalg.norm(p3 - p0))
        right_h = float(np.linalg.norm(p2 - p1))
        height = (left_h + right_h) / 2.0
        if np.isfinite(height) and height > 1:
            return height
    except Exception:
        pass
    return float(fallback_height)


def _fit_font_size(text, max_w, max_h, min_size=10, max_size=24):
    if max_w <= 0 or max_h <= 0:
        return min_size
    for size in range(max_size, min_size - 1, -1):
        tw, th = _measure_text(text, text_size=size)
        if tw <= max_w and th <= max_h:
            return size
    return min_size


def draw_result(orgimg, result_list):
    landmark_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    result_str = []
    img_h, img_w = orgimg.shape[:2]
    for idx, result in enumerate(result_list, start=1):
        raw_rect = _normalize_rect(result.get("rect"))
        if raw_rect is None:
            continue

        x1, y1, x2, y2 = raw_rect
        w = x2 - x1
        h = y2 - y1
        padding_w = int(round(0.05 * w))
        padding_h = int(round(0.11 * h))
        rx1 = _clamp(x1 - padding_w, 0, img_w - 1)
        ry1 = _clamp(y1 - padding_h, 0, img_h - 1)
        rx2 = _clamp(x2 + padding_w, 0, img_w - 1)
        ry2 = _clamp(y2 + padding_h, 0, img_h - 1)

        landmarks = result.get("landmarks", [])
        plate_no = result.get("plate_no", "")
        plate_color = result.get("plate_color", "")
        if result.get("plate_type", 0) == 1:
            result_p = "%s %s双层" % (plate_no, plate_color)
        else:
            result_p = "%s %s" % (plate_no, plate_color)
        result_str.append(result_p)

        theme = _get_plate_theme(plate_color)
        for i in range(min(4, len(landmarks))):
            point = landmarks[i]
            if len(point) < 2:
                continue
            point_color = landmark_colors[i]
            _draw_tech_landmark(orgimg, point[0], point[1], point_color, point_color)

        _draw_tech_box(orgimg, rx1, ry1, rx2, ry2, theme["border"], theme["glow"], track_id=idx)

        label = "%s | %s" % (plate_no, plate_color)
        plate_w = _plate_width_from_landmarks(landmarks, rx2 - rx1)
        plate_h = _plate_height_from_landmarks(landmarks, ry2 - ry1)
        pre_card_h = _clamp(int(round(plate_h)), 24, min(110, img_h - 4))
        pre_pad_y = _clamp(int(round(pre_card_h * 0.16)), 3, 10)
        pre_inner_h = max(8, pre_card_h - pre_pad_y * 2)
        pre_max_font = _clamp(int(round(pre_card_h * 0.72)), 14, 44)
        pre_min_font = _clamp(int(round(pre_card_h * 0.42)), 10, pre_max_font)
        pre_font_size = _fit_font_size(label, 4096, pre_inner_h, min_size=pre_min_font, max_size=pre_max_font)
        pre_text_w, _ = _measure_text(label, text_size=pre_font_size)

        min_w_by_text = pre_text_w + 20
        base_w_by_plate = int(round(plate_w * 1.05))
        card_w = max(90, base_w_by_plate, min_w_by_text)
        card_w = min(card_w, img_w - 8)

        card_h = pre_card_h
        card_pad_x = _clamp(int(round(card_w * 0.08)), 8, 18)
        card_pad_y = _clamp(int(round(card_h * 0.16)), 3, 10)

        card_x = int(rx1 + (rx2 - rx1 - card_w) / 2)
        card_x = _clamp(card_x, 4, max(4, img_w - card_w - 4))
        card_y = ry1 - card_h - 2
        if card_y < 2:
            card_y = _clamp(ry1 + 2, 2, max(2, img_h - card_h - 2))

        _draw_alpha_rect(orgimg, card_x, card_y, card_x + card_w, card_y + card_h, theme["bg"], alpha=0.78)
        _draw_glow_border(orgimg, card_x, card_y, card_x + card_w, card_y + card_h, theme["border"], theme["glow"])

        inner_w = max(8, card_w - card_pad_x * 2)
        inner_h = max(8, card_h - card_pad_y * 2)
        dynamic_max_font = _clamp(int(round(card_h * 0.72)), 14, 44)
        dynamic_min_font = _clamp(int(round(card_h * 0.42)), 10, dynamic_max_font)
        font_size = _fit_font_size(label, inner_w, inner_h, min_size=dynamic_min_font, max_size=dynamic_max_font)
        text_w, text_h = _measure_text(label, text_size=font_size)
        text_x = card_x + max(card_pad_x, int((card_w - text_w) / 2))
        text_y = card_y + max(card_pad_y - 1, int((card_h - text_h) / 2))
        orgimg = cv2ImgAddText(orgimg, label, text_x, text_y, theme["text"], font_size)

    return orgimg, result_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detect_model", type=str, default="weights/yolo26s-plate-detect.pt", help="detect model path")
    parser.add_argument("--rec_model", type=str, default="weights/plate_rec_color.pth", help="rec model path")
    parser.add_argument("--image_path", type=str, default="imgs", help="input image folder")
    parser.add_argument("--output", type=str, default="result", help="output folder")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda:0")
    opt = parser.parse_args()

    device = torch.device(opt.device)
    os.makedirs(opt.output, exist_ok=True)

    detect_model = load_model(opt.detect_model, device)
    plate_rec_model = init_model(device, opt.rec_model, is_color=True)

    detect_params = sum(p.numel() for p in detect_model.parameters())
    rec_params = sum(p.numel() for p in plate_rec_model.parameters())
    print("model_params | detect=%.2fM | rec=%.2fM" % (detect_params / 1e6, rec_params / 1e6))
    detect_model.eval()

    file_list = collect_files(opt.image_path)
    if not file_list:
        raise FileNotFoundError("no files found in %s" % opt.image_path)

    total_time = 0.0
    timed_total_time = 0.0
    begin_time = time.time()
    valid_count = 0
    timed_count = 0
    warmup_skipped = 0
    skip_first_timing = device.type == "cuda"

    for idx, pic_path in enumerate(file_list, start=1):
        t0 = time.time()

        img = cv2.imread(pic_path)
        if img is None:
            print("[%d/%d] %s | skip=read_failed" % (idx, len(file_list), os.path.basename(pic_path)))
            continue

        result_list = det_rec_plate(img, detect_model, plate_rec_model, device)
        t1 = time.time()
        vis_img, plate_texts = draw_result(img, result_list)

        save_img_path = os.path.join(opt.output, os.path.basename(pic_path))
        cv2.imwrite(save_img_path, vis_img)

        infer_time = t1 - t0
        total_time += infer_time
        valid_count += 1
        if skip_first_timing and warmup_skipped == 0:
            warmup_skipped = 1
        else:
            timed_total_time += infer_time
            timed_count += 1
        plate_info = " | ".join(plate_texts) if plate_texts else "-"
        print(
            "[%d/%d] %s | det=%d | plates=%s | time=%.1fms | save=%s"
            % (idx, len(file_list), os.path.basename(pic_path), len(result_list), plate_info, infer_time * 1000, save_img_path)
        )

    elapsed = time.time() - begin_time
    avg = (timed_total_time / timed_count) if timed_count else 0.0
    print(
        "summary | images=%d/%d | total=%.4fs | avg=%.4fs | timed_images=%d | warmup_skipped=%d"
        % (valid_count, len(file_list), elapsed, avg, timed_count, warmup_skipped)
    )

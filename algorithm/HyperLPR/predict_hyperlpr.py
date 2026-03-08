import argparse
import os
import time
import cv2
import hyperlpr3 as lpr3
import traceback

# 类别映射字典
TYPE_MAP = {
    1: "单行蓝牌",
    2: "单行黄牌",
    3: "新能源车牌",
    4: "教练车牌",
    5: "白色警用车牌",
    6: "使馆/港澳车牌",
    7: "双层黄牌",
    8: "武警车牌"
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='source')
    parser.add_argument('--output', type=str, required=True, help='output folder')
    parser.add_argument('--device', type=str, default='cpu')
    opt = parser.parse_args()

    out_dir = opt.output
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    img_path = opt.image_path
    if os.path.isdir(img_path):
        files = [os.path.join(img_path, f) for f in os.listdir(img_path) if f.endswith(('.jpg','.png','.jpeg', '.bmp'))]
    else:
        files = [img_path]

    try:
        catcher = lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_HIGH)
    except Exception as e:
        print("Model load error:", e)
        traceback.print_exc()
        # Fallback or exit
        return

    time_all = 0
    count = 0
    time_begin = time.time()

    for i, file in enumerate(files):
        time_b = time.time()
        img = cv2.imread(file)
        if img is None:
            continue

        results = catcher(img)
        time_e = time.time()
        
        plates_str = ""
        det_count = len(results) if results else 0
        
        if det_count > 0:
            for res in results:
                plate_text, conf, type_idx, rect = res
                # HyperLPR3 returns lists like [plate_code, conf, type, [x1, y1, x2, y2]]
                x1, y1, x2, y2 = rect
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                
                type_name = TYPE_MAP.get(type_idx, "未知")
                plates_str += f"{plate_text} {type_name}|"
                
            plates_str = plates_str.strip("|")
        else:
            plates_str = "-"

        save_name = os.path.basename(file)
        save_path = os.path.join(out_dir, save_name)
        cv2.imwrite(save_path, img)
        
        time_gap = time_e - time_b
        time_all += time_gap
        count += 1
        
        # We must output in a format that parseYolo26Output can parse:
        # e.g.: [1/1] img.png | det=1 | plates=皖1149885 单行蓝牌 | time=287.8ms | save=output.png
        time_ms = time_gap * 1000
        print(f"[{i+1}/{len(files)}] {save_name} | det={det_count} | plates={plates_str} | time={time_ms:.1f}ms | save={save_name}")

if __name__ == '__main__':
    main()
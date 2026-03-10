"""Diagnostic script to check YOLO model and test detection."""
import os, sys, cv2
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO

model = YOLO('weights/best.pt')
print("=== MODEL INFO ===")
print(f"Task: {model.task}")
print(f"Names: {model.names}")
print(f"Size: {round(os.path.getsize('weights/best.pt')/1024/1024, 2)} MB")

# Check what classes are in test.jpg
img = cv2.imread('images/test.jpg')
print(f"\n=== TEST images/test.jpg (shape={img.shape}) ===")
results = model(img, conf=0.5, verbose=False)
print(f"Total boxes (conf>=0.5): {len(results[0].boxes)}")
# Show unique classes
classes_found = set()
for b in results[0].boxes:
    cls_id = int(b.cls.item())
    classes_found.add(cls_id)
print(f"Unique classes found: {classes_found}")
print(f"Class names for found classes: {[model.names[c] for c in classes_found]}")

# Show first 5 boxes
for i, b in enumerate(results[0].boxes[:5]):
    cls_id = int(b.cls.item())
    print(f"  Box {i}: cls={cls_id}({model.names[cls_id]}) conf={b.conf.item():.4f} xyxy={[round(x,1) for x in b.xyxy[0].tolist()]}")


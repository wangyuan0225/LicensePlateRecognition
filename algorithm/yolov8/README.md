# YOLOv8 车牌识别算法

基于 YOLOv8 的中文车牌检测与识别，支持 12 种车牌类型，适配单层/双层车牌，CPU/GPU 均可运行。

---

## 环境要求

| 项目 | 版本要求 |
|------|----------|
| Python | 3.10 ~ 3.12 |
| torch  | 2.4.x（CPU 版）或更高（GPU 版需 CUDA） |
| 操作系统 | Windows 10/11 / Linux |

---

## 安装流程

### 1. 创建虚拟环境

```bash
# 在 algorithm/yolov8 目录下执行
python -m venv .venv
```

### 2. 激活虚拟环境

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. 安装依赖

> ⚠️ 注意：**不要直接 `pip install -r requirements.txt`**，因为 requirements.txt 中 `torch>=1.7.0` 会导致 pip 安装最新版 torch，在 Python 3.12 + Windows 下会报 `DLL 初始化失败` 错误。
>
> 请按以下顺序分步安装：

**Step 1：先安装指定版本的 PyTorch（CPU 版）**

```bash
pip install "torch==2.4.1" "torchvision==0.19.1" --index-url https://download.pytorch.org/whl/cpu
```

如果你有 NVIDIA 显卡并希望使用 GPU 加速，将上面命令改为（以 CUDA 12.1 为例）：

```bash
pip install "torch==2.4.1" "torchvision==0.19.1" --index-url https://download.pytorch.org/whl/cu121
```

**Step 2：安装其余依赖**

```bash
pip install -r requirements.txt
```

pip 会自动跳过已安装的 torch/torchvision，直接安装其他依赖。

### 4. 验证安装

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"
```

输出示例（CPU 版）：
```
torch: 2.4.1+cpu
cuda: False
```

---

## 测试运行

将待识别图片放入 `imgs/` 文件夹，然后执行：

```bash
python detect_rec_plate.py \
  --detect_model weights/yolov8s.pt \
  --rec_model weights/plate_rec_color.pth \
  --image_path imgs \
  --output result
```

识别结果图片保存在 `result/` 目录，终端输出车牌号及颜色信息。

---

## 支持的车牌类型

- [x] 1. 单行蓝牌
- [x] 2. 单行黄牌
- [x] 3. 新能源车牌
- [x] 4. 白色警用车牌
- [x] 5. 教练车牌
- [x] 6. 武警车牌
- [x] 7. 双层黄牌
- [x] 8. 双层白牌
- [x] 9. 使馆车牌
- [x] 10. 港澳粤Z牌
- [x] 11. 双层绿牌
- [x] 12. 民航车牌

---

## 常见问题

**Q: 报错 `OSError: [WinError 1114] 动态链接库(DLL)初始化例程失败`**

A: 这是 torch 版本与 Python 3.12 不兼容导致的。请严格按照上面的步骤，**先手动安装 torch 2.4.1**，再安装其他依赖。

**Q: `import torch` 后卡住不动**

A: 首次 import 时 torch 会初始化，CPU 版可能需要 5~10 秒，属于正常现象。

**Q: 模型文件 `weights/` 在哪里获取？**

A: 请前往原项目仓库下载：
- [yolov8-plate](https://github.com/we0091234/yolov8-plate)

---

## References

- [https://github.com/derronqi/yolov8-face](https://github.com/derronqi/yolov8-face)
- [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- [https://github.com/we0091234/crnn_plate_recognition](https://github.com/we0091234/crnn_plate_recognition)

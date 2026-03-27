# Algorithm 模块说明

该目录存放可被后端调用的 Python 车牌识别算法。后端不会直接链接 Python 库，而是通过进程方式执行脚本并解析标准输出。

## 目录结构

```text
algorithm/
├─ yolo26/
├─ yolov8/
├─ yolov11/
└─ HyperLPR/
```

## 与后端的集成协议

后端读取 `backend/src/main/resources/application.yml` 中的 `app.algorithms.*` 配置，按以下约定调用脚本：

- 工作目录：`base-dir`
- 解释器：`python-path`
- 脚本：`script-name`
- 公共参数：`--image_path <inputDir>`、`--output <outputDir>`
- 除 `yolov8` 外默认会附加 `--device cpu`

`yolov8` 额外参数：
- `--detect_model <path>`
- `--rec_model <path>`

## 输出约定（必须满足）

1. 结果图片写入 `--output` 指定目录。
2. 结果图片文件名必须与输入文件名一致（后端按文件名拷贝）。
3. 标准输出要可被后端解析：

- yolo26 / hyperlpr / yolov11 兼容格式：

```text
[1/1] demo.jpg | det=1 | plates=粤B12345 蓝色 | time=123.4ms | save=.../demo.jpg
```

- yolov8 输出由 `AnalyzeServiceImpl.parseYolov8Output` 解析，需包含可匹配车牌和颜色的文本。

## 已接入算法

### yolo26

- 默认推荐模型。
- 脚本：`detect_rec_plate.py`
- 支持目录批量输入。

### yolov8

- 脚本：`detect_rec_plate.py`
- 依赖 `detect-model`、`rec-model` 参数。

### yolov11

- 脚本：`predict_yolov11_lprnet.py`
- 已按 yolo26 的 stdout 格式兼容输出，便于后端统一解析。

### HyperLPR

- 脚本：`predict_hyperlpr.py`
- 已按 yolo26 的 stdout 格式兼容输出。

## 快速调试（PowerShell）

以 yolo26 为例：

```powershell
cd algorithm\yolo26
.\.venv\Scripts\Activate.ps1
python detect_rec_plate.py --image_path imgs --output result --device cpu
```

如果要模拟后端调用方式，可将 `--image_path` 指向单图临时目录，检查：
- stdout 是否符合约定
- `output` 中是否生成同名结果图

## 常见问题

- 后端报算法执行失败：检查 `python-path`、脚本名和依赖是否安装。
- 后端无解析结果：检查 stdout 是否与对应解析逻辑匹配。
- 后端找不到结果图：检查输出文件名是否与输入文件名一致。
- Windows/Linux 路径切换时，注意 `python-path` 的分隔符与解释器位置。

# 车牌识别系统 LicensePlateRecognition

基于 `Spring Boot 3 + Vue 3 + Python` 的多算法车牌识别项目，包含用户认证、识别历史、错误反馈和管理员审核功能。

项目 Demo 体验地址：

https://www.wangyuan0225.org/ （快速）
http://app.wangyuan0225.org/ （慢速）

## 项目结构

```text
LicensePlateRecognition/
├─ backend/           # Spring Boot 后端（API、鉴权、任务编排、持久化）
├─ frontend/          # Vue 3 前端（Vite）
├─ algorithm/         # Python 识别算法
│  ├─ yolo26/
│  ├─ yolov8/
│  ├─ yolov11/
│  └─ HyperLPR/
├─ TestImages/        # 测试图片
└─ README.md
```

## 技术栈

- 后端：Spring Boot 3.2.5、Spring Data JPA、JJWT、Spring Mail
- 前端：Vue 3、Vite、Element Plus、Pinia、vue-i18n
- 算法：YOLO26、YOLOv8、YOLOv11+LPRNet、HyperLPR（由后端通过 `ProcessBuilder` 调用）
- 数据库：MySQL 8

## 核心识别链路

1. 前端上传图片到 `POST /api/v1/analyze/upload`（携带 JWT）。
2. 后端保存原图到 `app.upload.dir/{userId}`。
3. 后端把图片复制到临时输入目录，调用算法脚本执行识别。
4. 解析算法标准输出（stdout）中的车牌号/颜色/类型，等待输出图写入。
5. 结果图复制到 `app.result.dir/{userId}`，并写入 `recognition_records`。
6. 返回统一响应格式：`{ code, message, data }`。

关键代码：
- `backend/src/main/java/com/wy0225/service/impl/AnalyzeServiceImpl.java`
- `backend/src/main/java/com/wy0225/config/AlgorithmConfig.java`
- `backend/src/main/java/com/wy0225/config/WebConfig.java`

## 快速开始

### 1) 环境要求

- JDK 17+
- Maven 3.8+
- Node.js `^20.19.0 || >=22.12.0`（见 `frontend/package.json`）
- Python 3.10+（建议 3.10/3.11）
- MySQL 8

### 2) 初始化数据库

执行：

```sql
-- backend/init.sql
```

默认会创建库 `lpr_db` 和管理员账号：
- 邮箱：`admin@lpr.com`
- 初始密码：`123456`

### 3) 配置后端本地私密参数

复制：

```text
backend/src/main/resources/application-local.yml.example
-> backend/src/main/resources/application-local.yml
```

至少确认这些配置：
- `spring.datasource.*`
- `spring.mail.*`
- `app.mail.from`
- `app.algorithms.*`（算法目录、Python 路径、脚本名）

### 4) 准备算法环境（以 Windows 为例）

YOLO26：

```powershell
cd algorithm\yolo26
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install "torch==2.4.1" "torchvision==0.19.1" --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

YOLOv8：

```powershell
cd algorithm\yolov8
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install "torch==2.4.1" "torchvision==0.19.1" --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

YOLOv11 / HyperLPR 可按各自目录 README 安装。

### 5) 启动服务

后端：

```powershell
cd backend
mvn clean test
mvn spring-boot:run
```

前端：

```powershell
cd frontend
npm install
npm run dev
```

默认访问地址：
- 前端：`http://localhost:5173`
- 后端：`http://localhost:8088`

## API 概览

- 认证：`/api/v1/auth/*`
- 识别：`/api/v1/analyze/upload`
- 历史：`/api/v1/history/*`
- 反馈：`/api/v1/feedback/*`
- 管理：`/api/v1/admin/*`

所有控制器位于 `backend/src/main/java/com/wy0225/controller/`。

## 关键配置说明

算法配置位于 `backend/src/main/resources/application.yml` 的 `app.algorithms.*`：

- 必填：`base-dir`、`python-path`、`script-name`
- 可选：`detect-model`、`rec-model`（当前用于 `yolov8`）

调用约定：
- 后端设置工作目录为 `base-dir`
- 设置环境变量 `PYTHONIOENCODING=utf-8`
- `yolov8` 额外传 `--detect_model --rec_model`
- 其他算法默认附加 `--device cpu`

融合模式（`modelType=fusion`）会串行执行 `hyperlpr`、`yolov8`、`yolo26`、`yolov11`，并以 `yolo26` 的结果图作为最终保存图。

## 常见问题

- `CreateProcess error=2`：通常是 `python-path` 或 `base-dir` 配置错误。
- 算法执行成功但无结果图：脚本输出目录中结果文件名必须与输入文件名一致。
- Linux 部署：`python-path` 需改为 `.venv/bin/python` 风格路径。
- 若本地目录使用 `algorithm/yolov11`，请同步修正配置中的 `app.algorithms.yolov11.base-dir`。

## 模块文档

- `backend/README.md`
- `frontend/README.md`
- `algorithm/README.md`

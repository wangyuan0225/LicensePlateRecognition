# 车牌识别系统 LicensePlateRecognition

基于 **Spring Boot + Vue 3** 的全本地化多算法车牌识别平台，支持双算法引擎（YOLO26 / YOLOv8），带完整的用户认证与历史归档功能。

---

## 项目结构

```
LicensePlateRecognition/
├── algorithm/
│   ├── yolo26/          # YOLO26 车牌识别算法（默认推荐）
│   └── yolov8/          # YOLOv8 车牌识别算法
├── backend/             # Spring Boot 后端
├── frontend/            # Vue 3 前端
└── README.md
```

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Vue 3 · Element Plus · vue-i18n（中/英双语） · Vite |
| 后端 | Spring Boot 3 · Spring Data JPA · JWT 认证 · BCrypt |
| 数据库 | MySQL 8 |
| 算法 | YOLO26（推荐）· YOLOv8，均通过 Python 虚拟环境隔离 |

---

## 功能特性

- **用户认证**：注册 / 登录，JWT Token，受保护路由（`/analyze`、`/history` 需登录）
- **双算法引擎**：上传时选择 YOLO26（默认）或 YOLOv8，算法各自独立 `.venv` 环境
- **per-user 目录隔离**：上传图片与结果图片分别存储在 `upload/images/{userId}/` 和 `upload/results/{userId}/`
- **历史记录**：按用户隔离、支持关键词搜索与日期筛选、分页展示、原图/结果图双图对比
- **中英双语**：页面右上角语言切换，中文为默认

---

## 环境要求

- JDK 17+
- Maven 3.8+
- Node.js 18+
- Python 3.10 ~ 3.12
- MySQL 8

---

## 快速启动

### 1. 数据库初始化

启动 MySQL，执行建表脚本：
```sql
-- 执行 backend/init.sql
```

在 `backend/src/main/resources/application.yml` 中配置数据库连接：
```yaml
spring:
  datasource:
    url: jdbc:mysql://<host>:<port>/lpr_db?...
    username: root
    password: root
```

### 2. 算法环境安装

> ⚠️ **必须先单独安装 torch，再安装其他依赖**，避免 Python 3.12 的 DLL 兼容性问题。

**YOLO26：**
```bash
cd algorithm/yolo26
python -m venv .venv
.venv\Scripts\activate         # Windows
pip install "torch==2.4.1" "torchvision==0.19.1" --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**YOLOv8：**
```bash
cd algorithm/yolov8
python -m venv .venv
.venv\Scripts\activate
pip install "torch==2.4.1" "torchvision==0.19.1" --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. 启动后端

```bash
cd backend
mvn spring-boot:run
# 默认监听 http://localhost:8088
```

### 4. 启动前端

```bash
cd frontend
npm install
npm run dev
# 默认访问 http://localhost:5173
```

---

## 算法配置

在 `backend/src/main/resources/application.yml` 中配置各算法路径：

```yaml
app:
  upload:
    dir: upload/images
  result:
    dir: upload/results
  algorithms:
    yolo26:
      base-dir: ../algorithm/yolo26
      python-path: ../algorithm/yolo26/.venv/Scripts/python.exe
      script-name: detect_rec_plate.py
    yolov8:
      base-dir: ../algorithm/yolov8
      python-path: ../algorithm/yolov8/.venv/Scripts/python.exe
      script-name: detect_rec_plate.py
      detect-model: weights/yolov8s.pt
      rec-model: weights/plate_rec_color.pth
```

---

## 系统架构

```
[Vue 3 前端]
    │  POST /api/v1/analyze/upload  (multipart/form-data + JWT)
    ▼
[Spring Boot 后端]
    │  java.lang.ProcessBuilder
    ▼
[Python 算法脚本 (.venv)]
    │  stdout 输出车牌号、颜色、耗时
    ▼
[AnalyzeService 正则解析] ──► [MySQL 存储] ──► [JSON 响应]
```

后端通过 `ProcessBuilder` 启动 Python 子进程，捕获 stdout，用正则提取结构化识别结果，无需 JNI 或 ONNX Java 绑定。

---

## 支持的车牌类型

单行蓝牌 · 单行黄牌 · 新能源车牌 · 白色警用车牌 · 教练车牌 · 武警车牌 · 双层黄牌 · 双层白牌 · 使馆车牌 · 港澳粤Z牌 · 双层绿牌 · 民航车牌

---

## 注意事项

- 首次启动时 Hibernate 会自动根据实体类建表（`ddl-auto: update`），也可手动执行 `backend/init.sql`
- Linux 部署时将 `python-path` 改为 `.venv/bin/python`
- 请确保 `algorithm/yolov8/weights/` 目录下已放置模型文件（`yolov8s.pt`、`plate_rec_color.pth`）
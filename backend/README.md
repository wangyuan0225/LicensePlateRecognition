# Backend 模块说明

后端基于 Spring Boot 3.2.5，负责 API、JWT 鉴权、算法进程编排、识别记录持久化、静态图片映射。

## 目录结构

```text
backend/
├─ src/main/java/com/wy0225/
│  ├─ controller/     # REST 接口
│  ├─ service/impl/   # 业务实现
│  ├─ config/         # 配置绑定 / WebConfig
│  ├─ common/         # JwtUtil / Result
│  ├─ entity/         # JPA 实体
│  └─ repository/     # JPA 仓储
├─ src/main/resources/
│  ├─ application.yml
│  ├─ application-local.yml
│  └─ application-local.yml.example
├─ init.sql
└─ pom.xml
```

## 运行要求

- JDK 17+
- Maven 3.8+
- MySQL 8

## 启动步骤

1. 执行 `init.sql` 初始化数据库。
2. 复制并修改本地配置：

```text
src/main/resources/application-local.yml.example
-> src/main/resources/application-local.yml
```

3. 在 `application-local.yml` 填写：
- `spring.datasource.*`
- `spring.mail.*`
- `app.mail.from`
- `app.algorithms.*`（按本机实际路径）

4. 启动：

```powershell
mvn clean test
mvn spring-boot:run
```

默认端口：`8088`

## 核心配置

`app.algorithms.<name>`（`AlgorithmConfig` 绑定）支持：
- `base-dir`：算法工作目录
- `python-path`：Python 可执行文件
- `script-name`：执行脚本
- `detect-model`：可选，仅 yolov8 使用
- `rec-model`：可选，仅 yolov8 使用

上传目录：
- `app.upload.dir`（原图）
- `app.result.dir`（结果图）

## 识别流程

入口：`AnalyzeServiceImpl.analyzeImage`

1. 保存上传图片到 `upload/images/{userId}`。
2. 创建临时输入输出目录。
3. 通过 `ProcessBuilder` 调用 Python：
- 工作目录 = `base-dir`
- 环境变量 `PYTHONIOENCODING=utf-8`
- `yolov8` 额外参数：`--detect_model --rec_model`
- 其他算法附加：`--device cpu`
4. 解析 stdout：
- `parseYolo26Output`（兼容 yolo26 / hyperlpr / yolov11 的格式）
- `parseYolov8Output`
5. 将结果图复制到 `upload/results/{userId}`。
6. 写入 `recognition_records` 并返回数据。

融合模式：
- `modelType=fusion`
- 依次执行 `hyperlpr`、`yolov8`、`yolo26`、`yolov11`
- 对车牌号与颜色投票，最终保存 `yolo26` 的结果图

## 响应规范

所有接口统一返回 `Result<T>`：

```json
{
  "code": 200,
  "message": "Success",
  "data": {}
}
```

## 认证约定

不使用 Spring Security Filter 链，控制器内手动解析 JWT。

请求头：

```text
Authorization: Bearer <token>
```

解析工具：`common/JwtUtil.java`

## 主要接口

- 认证：`/api/v1/auth/*`
- 识别上传：`POST /api/v1/analyze/upload`
- 个人历史：`GET /api/v1/history/list`，`DELETE /api/v1/history/{id}`
- 反馈：`/api/v1/feedback/*`
- 管理员：`/api/v1/admin/*`

## 静态图片访问

`WebConfig` 映射：
- `/static/upload/**` -> `app.upload.dir`
- `/static/result/**` -> `app.result.dir`

## 调试建议

- `CreateProcess error=2`：优先检查 `python-path`、`base-dir`。
- 没有识别结果：检查脚本 stdout 是否符合解析正则。
- 没有结果图：脚本输出文件名必须和输入文件名一致。
- 如果启用 `yolov11` 报路径错误，检查 `application.yml` 的 `app.algorithms.yolov11.base-dir` 与本地目录是否一致。

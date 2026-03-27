# Frontend 模块说明

前端基于 Vue 3 + Vite + Element Plus，提供登录注册、图片识别、历史记录、错误反馈与管理员页面。

## 技术栈

- Vue 3
- Vite
- Element Plus
- Pinia
- vue-router
- vue-i18n

## 目录结构

```text
frontend/
├─ src/
│  ├─ views/          # 页面（Analyze/History/Feedback/Admin...）
│  ├─ router/         # 路由与权限守卫
│  ├─ i18n/           # 多语言
│  ├─ stores/         # 状态管理
│  ├─ api/            # 请求封装
│  └─ composables/    # 复用逻辑
├─ vite.config.js
└─ package.json
```

## 环境要求

`package.json` 中声明：
- Node.js `^20.19.0 || >=22.12.0`

## 安装与运行

```powershell
npm install
npm run dev
```

构建：

```powershell
npm run build
npm run preview
```

默认开发地址：`http://localhost:5173`

## 代理配置

`vite.config.js` 中已配置：
- `/api` -> `http://localhost:8088`
- `/static` -> `http://localhost:8088`

因此本地开发无需手动拼接后端地址，直接请求相对路径即可。

## 路由与权限

路由定义：`src/router/index.js`

主要路由：
- 公开：`/`、`/login`、`/forgot-password`
- 登录后可访问：`/analyze`、`/history`、`/feedback`、`/change-password`
- 管理员：`/admin/history`、`/admin/feedback`
- 强制改密：`/force-change-password`

守卫逻辑：
- `meta.requiresAuth` 校验 token
- `meta.role` 校验角色（ADMIN）
- `forceChangePassword` 为 true 时强制跳转改密页

## 功能页面

- `Analyze.vue`
- 上传或拍照识别
- 支持模型：`yolo26`、`yolov11`、`yolov8`、`hyperlpr`、`fusion`
- 调用 `POST /api/v1/analyze/upload`

- `History.vue`
- 分页/关键字/日期筛选
- 预览原图和结果图
- 删除记录、提交纠错反馈

- `Feedback.vue`
- 查看个人反馈记录

- `AdminHistory.vue` / `AdminFeedback.vue`
- 管理员查看全量记录与反馈
- 支持反馈状态审批

## 本地存储约定

- `localStorage.token`：JWT
- `localStorage.user`：用户信息（含 `role`、`forceChangePassword`）

## 联调说明

后端接口统一返回：

```json
{
  "code": 200,
  "message": "Success",
  "data": {}
}
```

前端根据 `code === 200` 判断成功。

## 常见问题

- 识别请求 401：检查是否登录、`Authorization` 是否带 `Bearer `。
- 图片不显示：检查后端 `/static/upload/**`、`/static/result/**` 映射和文件是否存在。
- dev 服务可访问但接口失败：确认后端已启动在 `8088`，且 Vite 代理未被改动。

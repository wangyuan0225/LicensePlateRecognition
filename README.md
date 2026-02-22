# 基于Java Spring Boot与Vue 3的多场景全本地化车牌识别系统深度研究与需求分析报告

## 一、 领域现状与文献综合评述

在智慧城市基础设施、智能交通系统（Intelligent Transportation Systems, ITS）、园区安防以及无人值守停车场等现代应用场景中，车牌识别（Automatic License Plate Recognition, ALPR）技术扮演着至关重要的角色。对当前学术界最新文献与开源社区代码仓库的深度剖析表明，车牌识别技术正经历从受限场景（Constrained Scenarios）向无约束场景（Unconstrained Scenarios）的深刻范式转变。传统的机器视觉方法通常假设车辆处于正视角度、光照充足且背景单一的环境中，而现代系统则面临着极端天气、低光照、运动模糊以及车牌严重倾斜等复杂挑战。

从开源生态的演进轨迹来看，早期的开发者大多依赖于对C++底层计算机视觉库的浅层封装。当前，基于深度学习的算法（如YOLO系列与各类OCR模型）在Python生态中极其繁荣，开源社区涌现了大量高精度的车牌识别项目。然而，将这些复杂的算法引擎与现代企业级Web应用（如Spring Boot后端）进行整合，一直是工程落地中的痛点。

学术界的最新研究极大地推动了多阶段深度学习模型在车牌识别领域的应用。文献指出，现代ALPR系统通常被解耦为两个高度专业化的子任务：车牌检测（License Plate Detection, LPD）与车牌字符识别（License Plate Recognition, LPR）。在检测阶段，研究人员广泛应用了You Only Look Once (YOLO) 系列模型。最新的YOLO模型大幅减少了参数量，同时在检测精度与推理速度上实现了显著突破。在识别阶段，以CRNN（卷积循环神经网络）和LPRNet为代表的无分割端到端序列识别模型逐渐成为主流。这类模型利用连接时序分类（Connectionist Temporal Classification, CTC）损失函数，彻底免除了传统方法中极易出错的单字符分割步骤，直接从二维特征图中解码出完整的车牌字符序列。

为了兼顾Java企业级后端的稳定性与Python AI生态的丰富性，本研究提出构建一个完全运行于本地环境的多场景车牌识别系统。该系统以Java Spring Boot为后端核心，Vue 3为前端交互框架，其最显著的架构创新在于：**采用极其松耦合的命令行（CLI）进程调用模式替代复杂的底层协议**。系统允许用户根据实际部署场景的复杂度，自主选择最优的Python算法脚本（如极速模式、标准模式、复杂模式），Java后端仅负责任务调度、结果解析与数据持久化。

## 二、 核心车牌识别算法理论与多场景自适应机制

为了赋予系统在不同环境下的自适应能力，底层的算法引擎涵盖了多种技术栈。本系统规划集成三种具有代表性的算法方案，以实现对算力与精度的精细化权衡。所有算法均封装为独立的Python可执行脚本。

### 2.1 传统形态学与机器学习联合算法（极速模式）

在算力极度匮乏且场景高度可控的环境中，基于传统机器视觉的算法依然具有优势。该算法链路主要依赖于OpenCV的图像处理能力。当图像输入系统后，首先被转换为灰度图，并应用高斯滤波器以消除高频噪声。随后采用Sobel边缘检测算子提取垂直边缘特征，并通过形态学的闭操作将密集的边缘连接成连通域，从而完成车牌的粗定位与精提取。此方案的Python脚本执行极快，完全无需GPU加速，适用于地下停车场闸机口等固定抓拍场景。

### 2.2 HyperLPR3 轻量级中文车牌识别框架（标准模式）

针对绝大多数常规的室外监控与路面抓拍场景，系统引入了HyperLPR3等专为中文车牌优化的轻量级Python框架。这类框架基于深度卷积神经网络（CNN），能够高效识别包括普通蓝牌、新能源绿牌以及大型车辆黄牌在内的多种中国大陆车牌格式。通过Python脚本调用其预测接口，可以在普通CPU上实现速度与精度的完美平衡。

### 2.3 YOLO 联合 CRNN/LPRNet 混合架构（复杂模式）

针对气候条件恶劣、夜间光照不足以及车牌存在严重倾斜或遮挡的复杂场景，系统提供了基于YOLO与序列识别网络（如CRNN）的混合深度学习模型脚本。 该架构利用YOLO（如YOLOv8或更新的定制版YOLO）进行高精度的边界框定位。在完成定位后，将归一化的车牌图像馈入CRNN或LPRNet中进行字符解码。例如最新的开源YOLO车牌检测项目，已经能够精准识别单/双层黄牌、绿牌、港澳车牌等复杂格式。这类脚本算力消耗较大，但抗干扰能力极强。

| **算法模式** | **核心技术栈构成**                   | **调度机制与接口**  | **算力与硬件依赖**      | **核心优势**                                         |
| ------------ | ------------------------------------ | ------------------- | ----------------------- | ---------------------------------------------------- |
| **极速模式** | OpenCV, Sobel边缘, 形态学            | Python CLI 脚本调用 | 极低（纯CPU即可胜任）   | 响应极快，系统资源占用微乎其微                       |
| **标准模式** | 轻量级深度学习 (如HyperLPR Python版) | Python CLI 脚本调用 | 中等（CPU多核优化并发） | 针对中文车牌格式进行了深度预训练，通用性强           |
| **复杂模式** | YOLO检测级联 + CRNN/LPRNet识别       | Python CLI 脚本调用 | 较高（建议配备GPU加速） | 召回率极高，抗极端干扰能力最强，支持多角度与双层车牌 |

## 三、 全本地化后端架构设计与进程级跨语言调用机制

本系统的核心定位是彻底的本地化脱机运行。这要求后端的架构设计不仅要处理常规的RESTful API请求，还必须严密管理本地文件系统，并通过Java标准库实现对外部Python算法脚本的安全调度与输出捕获。后端框架选用Java Spring Boot。

### 3.1 本地存储与日志记录机制

为了确保所有的图片上传、识别处理及数据持久化均在本地服务器闭环完成，Spring Boot应用被配置为直接操控操作系统的文件系统。前端上传的图像数据被封装为`MultipartFile`对象传递给后端的控制器。在服务层，系统利用`java.nio.file.Files`类将原始图像持久化到预设的本地安全目录（如`upload/images/`）中。所有的识别元数据将被持久化至轻量级的关系型数据库（如内嵌的SQLite或MySQL）中。

### 3.2 基于 ProcessBuilder 的 Python 脚本跨进程调用机制

由于舍弃了复杂的JNI与ONNX Java API，系统采用了进程间通信（IPC）中最基础且高度解耦的命令行调用方式。在Spring Boot应用中，Java层使用 `java.lang.ProcessBuilder` 来启动独立的Python解释器进程执行指定的算法脚本。

工作流如下：

1. **构建指令**：当Java接收到上传的图片并存入本地后，获取该图片的绝对路径。根据用户选择的算法模式（如复杂模式），Java端构造命令行参数阵列，例如：`["python", "detect_plate.py", "--image", "C:/path/to/upload.jpg"]`。
2. **启动进程**：利用 `ProcessBuilder` 启动该命令。为了确保Java程序不会被阻塞且能捕获所有异常，必须将 `redirectErrorStream(true)` 设置为开启，这样Python脚本的错误输出（stderr）和标准输出（stdout）会被合并到同一个流中供Java读取。
3. **环境隔离**：在构建 `ProcessBuilder` 时，还可以通过 `processBuilder.environment()` 指定特定的Conda虚拟环境路径或Python环境变量，确保算法运行环境不互相冲突。

### 3.3 标准输出流（STDOUT）解析与结果入库

在此架构下，Python算法脚本无需提供复杂的API，仅需将识别结果以及处理后的结果图片路径打印到控制台即可。

针对算法输出的固定格式（例如：`[1/9] double_lv.png | det=1 | plates=皖1149885 绿色双层 | time=287.8ms | save=result\double_lv.png`），Spring Boot后端的解析逻辑如下：

1. **流读取**：利用 `BufferedReader` 和 `InputStreamReader` 实时读取正在运行的Python `Process` 的输出流。
2. **正则提取**：针对读取到的每一行文本，应用预设的正则表达式（Regex）进行结构化提取。例如：
   - 提取车牌号及属性：匹配 `plates=(.*?)\s` 得到 "皖1149885" 和 "绿色双层"。
   - 提取耗时：匹配 `time=(.*?)ms` 得到 "287.8"。
   - 提取结果图片路径：匹配 `save=(.*)` 得到相对或绝对路径 "result\double_lv.png"。
3. **数据库存储**：Java在等待进程执行完毕（`process.waitFor()`）后，将解析出的源图片路径、识别出的车牌字符串、车牌属性（如颜色、单双层）、执行算法类型、耗时以及**输出的结果图片路径（带有检测框）**，封装为实体类（Entity），通过Spring Data JPA保存入数据库。
4. **前端响应**：最后，Java将这些结构化数据（连带结果图片的访问URL）以JSON格式响应给Vue前端展示。

这种架构极大地降低了二次开发的门槛：无论后续引入何种最新的算法（YOLOv10、甚至其他语言编写的独立程序），只需确保其能在命令行运行并按约定格式打印输出，即可实现零代码或低代码无缝接入系统。

## 四、 需求规格说明书 (PRD)

### 4.1 产品背景与目标

本系统的核心目标是打造一款部署于本地计算节点的车牌识别业务中枢。产品通过友好的B/S可视化界面，将底层的Python机器视觉算法脚本当作黑盒服务进行调度。其最大创新点在于极简的集成协议与“按场景选算法”的动态路由能力，以此在本地有限的计算资源下，实现处理效率与识别精度的动态平衡。

### 4.2 角色与权限体系

系统采用扁平化的RBAC模型：

- **系统管理员**：具备全局最高权限。能够配置系统全局参数（如Python解释器路径、算法脚本挂载目录），并拥有全量历史记录的查询与删除权限。
- **操作员**：业务层面的主要使用者。能够进入工作台上传图像文件，选择适用的算法模型执行任务，并查看本人上传的历史记录及系统生成的带检测框的结果图片。

### 4.3 功能模块需求详述

#### 4.3.1 身份认证与安全模块

- **本地登录与鉴权**：系统提供独立的登录页面。前端Vue应用与Spring Boot后端的通信采用JWT鉴权协议。

#### 4.3.2 仪表盘总览模块 (Dashboard)

- **数据统计面板**：展示当日识别总次数、累计识别次数、各类算法（极速/标准/复杂）的调用比例分布及平均响应耗时。

#### 4.3.3 核心识别工作台模块 (Workspace)

- **算法策略选择器**：提供单选组件，允许用户在发起识别前选择适用的算法：
  - **极速模式**：调用轻量级脚本，适合简单环境。
  - **标准模式**：调用中文优化脚本，平衡通用性。
  - **复杂模式 (YOLO高精模型)**：适用于夜间、雨雪、双层车牌等复杂场景。
- **本地图像上传**：支持用户选择本地图像文件。前端展示原图预览。
- **异步识别与结果反显**：用户点击“开始识别”后，前端展示Loading加载状态。Java后端通过CLI调用Python算法，解析控制台输出并返回。前端解析JSON，在界面右侧显著区域展示：**原图、带边界框的检测结果图**、识别出的车牌号码、车牌属性（如：绿色双层）、以及该次底层的处理耗时。

#### 4.3.4 历史记录归档模块 (History Archive)

- **结构化数据表格**：以分页列表的形式呈现所有历史记录。包含：流水号、时间戳、车牌号码、车牌属性、算法模式、处理耗时。
- **双图对比溯源**：在表格的操作列中，提供“查看详情”按钮。点击后以模态框（Modal）形式同时展示用户上传的“原始图像”与算法生成的“带检测框结果图像”，以便人工对系统的识别结果进行二次核对与纠偏。

### 4.4 非功能性需求

- **跨平台兼容**：后端的 `ProcessBuilder` 调用逻辑需根据宿主机操作系统（Windows的`python`与Linux的`python3`，以及路径分隔符差异）进行自适应配置。
- **异常处理**：若Python脚本崩溃、找不到环境或返回异常日志（如缺少依赖），Java后端需能够捕获stderr信息，并向前端返回友好的“算法引擎执行失败”提示，而非系统宕机。

## 五、 前端工程与交互原型实现

### 5.1 前端工程架构逻辑

原型代码采用了全局构建版本的Vue 3和Tailwind CSS集成于单个HTML文件之中。当用户发起识别任务时，前端将文件包装在 `FormData` 中发起请求，并根据后端解析Python脚本返回的JSON结果，渲染识别出的车牌文本与生成的标注结果图片。

### 5.2 核心全场景页面原型源码

HTML

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>全本地多场景车牌识别平台 (CLI引擎调度版)</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
      .fade-enter-active,.fade-leave-active { transition: opacity 0.3s ease; }
      .fade-enter-from,.fade-leave-to { opacity: 0; }
        input[type="file"] { display: none; }
    </style>
</head>
<body class="bg-slate-50 text-slate-800 font-sans antialiased h-screen overflow-hidden flex flex-col">
    <div id="app" class="h-full flex flex-col">
        
        <nav v-if="activeRoute!== 'login'" class="bg-indigo-700 text-white shadow-lg flex-none">
            <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div class="flex items-center justify-between h-16">
                    <div class="flex items-center space-x-3 cursor-default">
                        <i class="fa-solid fa-camera-retro text-2xl text-indigo-200"></i>
                        <span class="font-bold text-xl tracking-wide">本地车牌识别中枢</span>
                        <span class="ml-2 px-2 py-0.5 rounded text-xs bg-indigo-800 text-indigo-300 border border-indigo-600">CLI IPC Mode</span>
                    </div>
                    <div class="flex space-x-2">
                        <button @click="switchRoute('dashboard')" :class="{'bg-indigo-900 border-indigo-500': activeRoute === 'dashboard', 'border-transparent hover:bg-indigo-600': activeRoute!== 'dashboard'}" class="px-4 py-2 border-b-2 font-medium transition duration-150">
                            <i class="fa-solid fa-chart-pie mr-1"></i> 总览概况
                        </button>
                        <button @click="switchRoute('workspace')" :class="{'bg-indigo-900 border-indigo-500': activeRoute === 'workspace', 'border-transparent hover:bg-indigo-600': activeRoute!== 'workspace'}" class="px-4 py-2 border-b-2 font-medium transition duration-150">
                            <i class="fa-solid fa-microchip mr-1"></i> 识别工作台
                        </button>
                        <button @click="switchRoute('history')" :class="{'bg-indigo-900 border-indigo-500': activeRoute === 'history', 'border-transparent hover:bg-indigo-600': activeRoute!== 'history'}" class="px-4 py-2 border-b-2 font-medium transition duration-150">
                            <i class="fa-solid fa-database mr-1"></i> 历史记录
                        </button>
                        <div class="w-px h-6 bg-indigo-500 mx-2 self-center"></div>
                        <button @click="executeLogout" class="px-4 py-2 rounded-md font-medium text-red-100 hover:bg-red-600 hover:text-white transition duration-150">
                            <i class="fa-solid fa-arrow-right-from-bracket mr-1"></i> 退出
                        </button>
                    </div>
                </div>
            </div>
        </nav>

        <main class="flex-grow overflow-y-auto p-6 flex justify-center items-start">
            
            <transition name="fade" mode="out-in">
                <div v-if="activeRoute === 'login'" class="bg-white p-10 rounded-2xl shadow-2xl max-w-sm w-full mt-20 border border-slate-100">
                    <div class="text-center mb-10">
                        <div class="inline-flex items-center justify-center w-20 h-20 rounded-full bg-indigo-100 mb-4">
                            <i class="fa-solid fa-shield-halved text-4xl text-indigo-600"></i>
                        </div>
                        <h2 class="text-2xl font-extrabold text-slate-900">操作员验证</h2>
                        <p class="text-slate-500 text-sm mt-2">系统受控，凭证数据仅保存在本地存储</p>
                    </div>
                    <form @submit.prevent="executeLogin" class="space-y-6">
                        <div>
                            <label class="block text-sm font-semibold text-slate-700 mb-1">系统账号</label>
                            <input type="text" v-model="authForm.account" required placeholder="请输入操作员工号" class="block w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition">
                        </div>
                        <div>
                            <label class="block text-sm font-semibold text-slate-700 mb-1">授权密码</label>
                            <input type="password" v-model="authForm.secret" required placeholder="••••••••" class="block w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition">
                        </div>
                        <button type="submit" class="w-full py-3 px-4 rounded-lg text-sm font-bold text-white bg-indigo-600 hover:bg-indigo-700 transition duration-200">
                            加密登入系统
                        </button>
                    </form>
                </div>

                <div v-else-if="activeRoute === 'dashboard'" class="max-w-7xl w-full space-y-6">
                    <div class="flex items-center justify-between border-b border-slate-200 pb-4">
                        <h1 class="text-2xl font-bold text-slate-800">系统全局运行视图</h1>
                        <span class="text-sm text-slate-500">最后刷新时间: 本地时间 {{ currentTime }}</span>
                    </div>
                    <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
                        <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-100 flex items-center justify-between">
                            <div>
                                <p class="text-sm font-medium text-slate-500">本月识别存储量</p>
                                <p class="text-2xl font-bold text-slate-800 mt-1">12.5 <span class="text-sm font-normal text-slate-500">GB</span></p>
                            </div>
                            <div class="w-12 h-12 rounded-full bg-blue-100 flex items-center justify-center text-blue-600"><i class="fa-solid fa-hard-drive text-xl"></i></div>
                        </div>
                        <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-100 flex items-center justify-between">
                            <div>
                                <p class="text-sm font-medium text-slate-500">累计脚本调用任务</p>
                                <p class="text-2xl font-bold text-slate-800 mt-1">15,402 <span class="text-sm font-normal text-slate-500">次</span></p>
                            </div>
                            <div class="w-12 h-12 rounded-full bg-green-100 flex items-center justify-center text-green-600"><i class="fa-solid fa-terminal text-xl"></i></div>
                        </div>
                        <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-100 flex items-center justify-between">
                            <div>
                                <p class="text-sm font-medium text-slate-500">算法进程成功率</p>
                                <p class="text-2xl font-bold text-slate-800 mt-1">99.8 <span class="text-sm font-normal text-slate-500">%</span></p>
                            </div>
                            <div class="w-12 h-12 rounded-full bg-purple-100 flex items-center justify-center text-purple-600"><i class="fa-solid fa-check-to-slot text-xl"></i></div>
                        </div>
                        <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-100 flex items-center justify-between">
                            <div>
                                <p class="text-sm font-medium text-slate-500">YOLO脚本平均耗时</p>
                                <p class="text-2xl font-bold text-slate-800 mt-1">160 <span class="text-sm font-normal text-slate-500">ms</span></p>
                            </div>
                            <div class="w-12 h-12 rounded-full bg-orange-100 flex items-center justify-center text-orange-600"><i class="fa-solid fa-stopwatch text-xl"></i></div>
                        </div>
                    </div>
                </div>

                <div v-else-if="activeRoute === 'workspace'" class="max-w-6xl w-full bg-white rounded-2xl shadow-lg border border-slate-100 overflow-hidden flex flex-col md:flex-row">
                    <div class="md:w-5/12 p-8 bg-slate-50 border-r border-slate-200 flex flex-col">
                        <h2 class="text-xl font-extrabold text-slate-800 mb-6 flex items-center"><i class="fa-solid fa-sliders text-indigo-600 mr-2"></i>发起算法进程</h2>
                        
                        <div class="mb-8">
                            <label class="block text-sm font-bold text-slate-700 mb-3">步骤一：选择底层Python脚本环境</label>
                            <div class="space-y-3">
                                <label class="flex items-start p-4 border-2 rounded-xl cursor-pointer transition-all duration-200 relative" :class="algorithmSelection === 'cv_traditional'? 'border-indigo-500 bg-indigo-50/50 shadow-sm' : 'border-slate-200 hover:border-indigo-300 hover:bg-slate-50'">
                                    <input type="radio" v-model="algorithmSelection" value="cv_traditional" class="hidden">
                                    <div class="mt-0.5 mr-3 w-5 h-5 rounded-full border-2 flex items-center justify-center" :class="algorithmSelection === 'cv_traditional'? 'border-indigo-600' : 'border-slate-300'">
                                        <div v-if="algorithmSelection === 'cv_traditional'" class="w-2.5 h-2.5 bg-indigo-600 rounded-full"></div>
                                    </div>
                                    <div>
                                        <p class="font-bold text-slate-800">极速模式 <span class="ml-2 font-mono text-xs text-slate-500 bg-white px-1 py-0.5 rounded border">OpenCV Script</span></p>
                                        <p class="text-xs text-slate-500 mt-1 leading-relaxed">纯CPU运算脚本，仅适用于光照极佳、标准化场景。</p>
                                    </div>
                                </label>
                                <label class="flex items-start p-4 border-2 rounded-xl cursor-pointer transition-all duration-200 relative" :class="algorithmSelection === 'yolo_high'? 'border-indigo-500 bg-indigo-50/50 shadow-sm' : 'border-slate-200 hover:border-indigo-300 hover:bg-slate-50'">
                                    <input type="radio" v-model="algorithmSelection" value="yolo_high" class="hidden">
                                    <div class="mt-0.5 mr-3 w-5 h-5 rounded-full border-2 flex items-center justify-center" :class="algorithmSelection === 'yolo_high'? 'border-indigo-600' : 'border-slate-300'">
                                        <div v-if="algorithmSelection === 'yolo_high'" class="w-2.5 h-2.5 bg-indigo-600 rounded-full"></div>
                                    </div>
                                    <div>
                                        <p class="font-bold text-slate-800">复杂模式 <span class="ml-2 font-mono text-xs text-slate-500 bg-white px-1 py-0.5 rounded border">YOLO CLI Script</span></p>
                                        <p class="text-xs text-slate-500 mt-1 leading-relaxed">支持双层牌、领馆牌等各类复杂版式，通过执行Python命令行获取结构化识别结果输出。</p>
                                    </div>
                                </label>
                            </div>
                        </div>

                        <div class="mb-8 flex-grow">
                            <label class="block text-sm font-bold text-slate-700 mb-3">步骤二：挂载本地图像资源</label>
                            <label for="local-file-upload" class="flex flex-col items-center justify-center w-full h-32 border-2 border-slate-300 border-dashed rounded-xl cursor-pointer bg-white hover:bg-slate-50 hover:border-indigo-400 transition-colors">
                                <div class="flex flex-col items-center justify-center pt-5 pb-6">
                                    <i class="fa-solid fa-cloud-arrow-up text-3xl text-slate-400 mb-2"></i>
                                    <p class="text-sm text-slate-600 font-medium">点击加载本地图像</p>
                                </div>
                                <input id="local-file-upload" type="file" @change="captureLocalFile" accept="image/*">
                            </label>
                            <div v-if="targetFile" class="mt-3 p-3 bg-green-50 rounded-lg border border-green-200 flex items-center justify-between">
                                <span class="text-sm font-medium text-green-800 truncate">{{ targetFile.name }}</span>
                            </div>
                        </div>

                        <button @click="dispatchRecognitionTask" :disabled="!targetFile |

| engineRunning" class="w-full py-4 rounded-xl shadow-md text-white font-bold text-lg transition duration-200 flex justify-center items-center disabled:opacity-60 disabled:cursor-not-allowed" :class="engineRunning? 'bg-indigo-400' : 'bg-indigo-600 hover:bg-indigo-700'">
                            <i v-if="engineRunning" class="fa-solid fa-circle-notch fa-spin mr-2"></i>
                            <i v-else class="fa-solid fa-terminal mr-2"></i>
                            {{ engineRunning? 'ProcessBuilder 执行脚本中...' : '提交 CLI 识别任务' }}
                        </button>
                    </div>

                    <div class="md:w-7/12 p-8 bg-white flex flex-col">
                        <h3 class="text-lg font-bold text-slate-800 mb-4 border-b border-slate-100 pb-2">输出流解析与结果</h3>
                        
                        <div v-if="!engineResult &&!engineRunning &&!localPreviewUrl" class="flex-grow flex flex-col items-center justify-center text-slate-300">
                            <i class="fa-regular fa-image text-7xl mb-4 opacity-40"></i>
                            <p class="text-lg font-medium">等待执行 Python 算法指令</p>
                        </div>

                        <div v-if="localPreviewUrl" class="w-full bg-slate-100 rounded-xl overflow-hidden mb-6 flex items-center justify-center relative border border-slate-200" style="height: 360px;">
                            <img :src="engineResult? engineResult.resultImg : localPreviewUrl" alt="预览或结果图" class="max-w-full max-h-full object-contain shadow-sm">
                            <div v-if="engineResult" class="absolute top-2 right-2 bg-green-600 text-white text-xs px-2 py-1 rounded shadow-md opacity-90">
                                算法输出图像 (包含检测框)
                            </div>
                        </div>

                        <div v-if="engineResult" class="bg-indigo-50 border border-indigo-100 rounded-xl p-6 shadow-sm">
                            <h4 class="text-xs font-bold text-indigo-400 uppercase tracking-wider mb-4">Java解析 STDOUT 响应</h4>
                            <div class="grid grid-cols-2 gap-y-4 gap-x-6">
                                <div>
                                    <p class="text-sm text-slate-500 mb-1">识别文本 (Plate Text)</p>
                                    <p class="text-2xl font-black text-slate-800 tracking-widest font-mono">{{ engineResult.plateText }}</p>
                                </div>
                                <div>
                                    <p class="text-sm text-slate-500 mb-1">车牌属性 (Plate Attr)</p>
                                    <p class="text-lg font-bold text-green-700">{{ engineResult.plateAttr }}</p>
                                </div>
                                <div>
                                    <p class="text-sm text-slate-500 mb-1">调用脚本指令</p>
                                    <p class="text-xs font-medium text-slate-700 bg-white inline-block px-2 py-1 rounded border">{{ engineResult.engineName }}</p>
                                </div>
                                <div>
                                    <p class="text-sm text-slate-500 mb-1">CLI 执行耗时</p>
                                    <p class="text-sm font-bold text-slate-800 font-mono"><i class="fa-solid fa-bolt text-yellow-500 mr-1"></i> {{ engineResult.executionMs }} ms</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div v-else-if="activeRoute === 'history'" class="max-w-7xl w-full">
                    <div class="bg-white shadow-lg rounded-2xl overflow-hidden border border-slate-200">
                        <div class="px-8 py-6 border-b border-slate-200 bg-white flex flex-col justify-between items-start gap-2">
                            <h2 class="text-xl font-extrabold text-slate-800">数据库结构化归档视图</h2>
                            <p class="text-sm text-slate-500">展示原图与Python脚本处理后存储在 save= 路径下的结果图对比</p>
                        </div>
                        <div class="overflow-x-auto">
                            <table class="min-w-full divide-y divide-slate-200">
                                <thead class="bg-slate-50">
                                    <tr>
                                        <th scope="col" class="px-6 py-3 text-left text-xs font-bold text-slate-500 uppercase">记录ID</th>
                                        <th scope="col" class="px-6 py-3 text-left text-xs font-bold text-slate-500 uppercase">图片名称</th>
                                        <th scope="col" class="px-6 py-3 text-left text-xs font-bold text-slate-500 uppercase">车牌与属性</th>
                                        <th scope="col" class="px-6 py-3 text-left text-xs font-bold text-slate-500 uppercase">执行脚本</th>
                                        <th scope="col" class="px-6 py-3 text-right text-xs font-bold text-slate-500 uppercase">操作</th>
                                    </tr>
                                </thead>
                                <tbody class="bg-white divide-y divide-slate-200">
                                    <tr class="hover:bg-indigo-50/50 transition">
                                        <td class="px-6 py-4 whitespace-nowrap text-sm font-mono text-slate-500">CLI-001</td>
                                        <td class="px-6 py-4 whitespace-nowrap text-sm text-slate-900">double_lv.png</td>
                                        <td class="px-6 py-4 whitespace-nowrap">
                                            <span class="px-2 py-1 text-xs font-bold rounded bg-indigo-100 text-indigo-800">皖1149885</span>
                                            <span class="ml-2 text-xs text-green-700 font-bold">绿色双层</span>
                                        </td>
                                        <td class="px-6 py-4 whitespace-nowrap text-sm text-slate-500">python detect_plate.py</td>
                                        <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-right">
                                            <button class="text-indigo-600 hover:text-indigo-900"><i class="fa-solid fa-images"></i> 详情比对</button>
                                        </td>
                                    </tr>
                                    <tr class="hover:bg-indigo-50/50 transition">
                                        <td class="px-6 py-4 whitespace-nowrap text-sm font-mono text-slate-500">CLI-002</td>
                                        <td class="px-6 py-4 whitespace-nowrap text-sm text-slate-900">hongkang1.jpg</td>
                                        <td class="px-6 py-4 whitespace-nowrap">
                                            <span class="px-2 py-1 text-xs font-bold rounded bg-indigo-100 text-indigo-800">粤ZR066港</span>
                                            <span class="ml-2 text-xs text-gray-700 font-bold">黑色</span>
                                        </td>
                                        <td class="px-6 py-4 whitespace-nowrap text-sm text-slate-500">python detect_plate.py</td>
                                        <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-right">
                                            <button class="text-indigo-600 hover:text-indigo-900"><i class="fa-solid fa-images"></i> 详情比对</button>
                                        </td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </transition>
        </main>
    </div>

    <script>
        const { createApp, ref, reactive, onMounted } = Vue;

        createApp({
            setup() {
                const activeRoute = ref('workspace');
                const currentTime = ref('');
                const authForm = reactive({ account: '', secret: '' });

                const algorithmSelection = ref('yolo_high');
                const targetFile = ref(null);
                const localPreviewUrl = ref('');
                const engineRunning = ref(false);
                const engineResult = ref(null);

                const refreshClock = () => {
                    const now = new Date();
                    currentTime.value = `${now.getFullYear()}-${now.getMonth()+1}-${now.getDate()} ${now.getHours()}:${now.getMinutes()}`;
                };

                const switchRoute = (target) => {
                    activeRoute.value = target;
                };

                const executeLogin = () => switchRoute('dashboard');
                const executeLogout = () => switchRoute('login');

                const captureLocalFile = (event) => {
                    const file = event.target.files;
                    if (file) {
                        targetFile.value = file;
                        localPreviewUrl.value = URL.createObjectURL(file);
                        engineResult.value = null; 
                    }
                };

                // 模拟 Java 端利用 ProcessBuilder 执行 python 脚本并解析 STDOUT
                const dispatchRecognitionTask = () => {
                    engineRunning.value = true;
                    engineResult.value = null;

                    let simulatedLatency = algorithmSelection.value === 'yolo_high'? 287 : 80;
                    let cmdText = algorithmSelection.value === 'yolo_high'? 'python detect_plate.py --detect_model...' : 'python opencv_fast.py';
                    
                    setTimeout(() => {
                        // 此处模拟Java后端利用正则从 stdout 提取的内容： "plates=皖1149885 绿色双层 | time=287.8ms | save=result\double_lv.png"
                        engineResult.value = {
                            plateText: '皖1149885',
                            plateAttr: '绿色双层',
                            engineName: cmdText,
                            executionMs: simulatedLatency,
                            resultImg: localPreviewUrl.value // 模拟后端回传了生成好检测框的新图片URL
                        };
                        engineRunning.value = false;
                    }, simulatedLatency + 300); // 增加固定I/O网络耗时
                };

                onMounted(() => setInterval(refreshClock, 1000));

                return {
                    activeRoute, currentTime, authForm,
                    algorithmSelection, targetFile, localPreviewUrl, engineRunning, engineResult,
                    switchRoute, executeLogin, executeLogout, captureLocalFile, dispatchRecognitionTask
                };
            }
        }).mount('#app');
    </script>
</body>
</html>
```
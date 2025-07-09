# Smart Chat Frontend

这是一个智能对话前端应用，使用 Vue.js 构建。它提供一个聊天界面，用户可以输入问题，前端会调用后端API来获取并展示：
- Echarts 图表
- 文本数据
- Base64 编码的图片

## 项目设置与启动

### 前提条件
- Node.js (推荐最新 LTS 版本)
- npm (通常随 Node.js 一起安装) 或 yarn

### 依赖安装
在项目根目录下，打开终端并运行以下命令之一来安装项目依赖：
```bash
npm install
```
或者，如果您使用 yarn：
```bash
yarn install
```

### 启动开发服务器
安装完依赖后，运行以下命令来启动本地开发服务器：
```bash
npm run serve
```
或者，如果您使用 yarn：
```bash
yarn serve
```
该命令会编译项目并在本地启动一个开发服务器（通常在 `http://localhost:8080` 或类似端口）。在浏览器中打开此地址即可看到应用。

### 构建生产版本
如果您需要为生产环境构建优化后的静态文件，请运行：
```bash
npm run build
```
或者，如果您使用 yarn：
```bash
yarn build
```
构建产物将存放在 `dist/` 目录下。

## 项目结构简介
- `public/`: 存放静态资源和 `index.html` 模板。
- `src/`: 存放 Vue.js 应用的核心代码。
  - `components/`: 存放 Vue 组件。
    - `ChatInterface.vue`: 核心的聊天界面组件。
    - `EchartRenderer.vue`: 用于渲染 Echarts 图表的组件。
  - `services/`: 存放 API 调用相关的服务。
    - `apiService.js`: 封装了与后端 `/api/echarts` 和 `/api/query` 接口的交互逻辑。
  - `App.vue`: 根 Vue 组件。
  - `main.js`: Vue 应用的入口文件。
- `package.json`: 定义项目依赖和脚本。

## 后端接口说明
前端应用依赖以下后端接口，请确保后端服务在 `http://127.0.0.1:5019` 上运行：

1.  **POST /api/echarts**
    *   **请求参数 (JSON Body)**: `{"query": "用户查询内容", "database": "数据库名"}` (默认为 "archive")
    *   **成功响应**:
        ```json
        {
            "CODE": 20000,
            "DATA": {
                "echart": ["Echarts option 字符串1", "Echarts option 字符串2"]
            },
            "MSG": "Success"
        }
        ```
2.  **POST /api/query**
    *   **请求参数 (JSON Body)**: `{"query": "用户查询内容", "database": "数据库名"}` (默认为 "archive")
    *   **成功响应**:
        ```json
        {
            "CODE": 20000,
            "DATA": {
                "data": "文本结果...",
                "images": ["b'base64图片字符串1'", "b'base64图片字符串2'"],
                "urls": []
            },
            "MSG": "Success"
        }
        ```

**注意**: `apiService.js` 中已将后端基础URL硬编码为 `http://127.0.0.1:5019/api`。如果您的后端地址不同，请相应修改此文件。
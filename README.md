# 微信公众号 AI 恋爱分析助手

将聊天记录转发给公众号，AI 帮你分析对方兴趣度、性格特征、风险信号，并给出回复话术建议。

## 功能特性

- **文字分析** — 直接粘贴聊天记录，自动识别多种格式
- **截图分析** — 发送聊天截图，多模态 AI 自动识别并分析
- **语音输入** — 发语音消息，自动转文字分析
- **对话记忆** — 记住上下文，支持追问和连续分析
- **结构化报告** — 兴趣度评分、性格画像、风险扫描、话术建议

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repo-url>
cd wechat-ai-community

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

```bash
# 复制配置模板
copy .env.example .env       # Windows
# cp .env.example .env       # Linux/Mac
```

编辑 `.env` 文件，填入：

| 配置项 | 说明 |
|-------|------|
| `WECHAT_TOKEN` | 公众号后台设置的 Token |
| `WECHAT_APP_ID` | 公众号 AppID |
| `WECHAT_APP_SECRET` | 公众号 AppSecret |
| `AI_API_KEY` | AI 模型 API Key |
| `AI_BASE_URL` | API 地址（默认 DeepSeek） |
| `AI_MODEL` | 模型名称 |

**支持的 AI 服务商：**

| 服务商 | `AI_BASE_URL` | `AI_MODEL` |
|-------|---------------|------------|
| DeepSeek | `https://api.deepseek.com` | `deepseek-chat` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| 智谱 AI | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` |

### 3. 启动服务

```bash
# 开发模式
uvicorn app.main:app --reload --port 8000

# 生产模式
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

### 4. Docker 部署

```bash
docker build -t wechat-ai-coach .
docker run -d --name wechat-ai -p 8000:8000 --env-file .env wechat-ai-coach
```

### 5. 微信公众号配置

1. 登录 [微信公众平台](https://mp.weixin.qq.com)
2. 进入 **设置与开发 → 基本配置**
3. 服务器配置：
   - **URL**: `https://你的域名/wechat`
   - **Token**: 与 `.env` 中一致
   - **EncodingAESKey**: 随机生成
   - **消息加解密方式**: 明文模式（开发阶段）
4. 启用服务器配置

> ⚠️ 微信要求服务器必须有 HTTPS。推荐使用 Nginx 反向代理 + Let's Encrypt 证书。

## 架构说明

```
用户微信 ──→ 微信服务器 ──→ 你的服务器 (/wechat)
                                │
                                ├─ 文字消息 → 聊天记录解析器 → AI 分析
                                ├─ 图片消息 → 下载图片 → 多模态 AI 分析
                                └─ 语音消息 → 微信语音识别 → AI 分析
                                        │
                                        ▼
                                   分析报告 ──→ 回复用户
```

### 5 秒超时处理策略

微信要求 5 秒内回复，但 AI 分析通常需要 10-20 秒。本项目的解决方案：

1. **快速响应**：先等 4 秒看 AI 能否完成
2. **微信重试**：未完成则返回"分析中"，微信会在 5s/10s 后重试
3. **缓存结果**：AI 完成后缓存结果，下次重试时直接返回
4. **用户兜底**：超时后用户可回复「查看结果」手动获取

### 聊天记录格式支持

自动识别以下格式：

```
# 格式1：微信标准复制格式
张三 2024-01-15 10:30
你好，最近忙什么呢？

李四 2024-01-15 10:32
还好啦，上班摸鱼中😂

# 格式2：简单对话格式
我：你好
她：你好呀
我：最近在忙什么

# 格式3：带时间戳
[10:30] 张三：你好
[10:32] 李四：你好呀

# 格式4：自由文本（AI 自动理解）
今天和她聊天，她说周末有空，我约她看电影她说再看看...
```

## 项目结构

```
wechat-ai-community/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI 入口 + 路由 + 业务编排
│   ├── config.py         # 环境变量配置
│   ├── wechat.py         # 微信消息解析与回复
│   ├── analyzer.py       # AI 分析引擎 + Prompt
│   ├── parser.py         # 聊天记录格式解析
│   └── database.py       # 数据库模型 + 会话管理
├── data/                 # SQLite 数据库文件（自动创建）
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

## 常见问题

**Q：公众号没有回复？**
- 检查服务器日志 `uvicorn` 输出
- 确认微信后台服务器配置的 URL 和 Token 正确
- 确认服务器可通过 HTTPS 访问

**Q：AI 分析太慢？**
- 尝试使用更快的模型（如 `deepseek-chat` 或 `qwen-turbo`）
- 如果是认证服务号，开启 `ENABLE_CUSTOMER_SERVICE_API=true` 可突破 5 秒限制

**Q：截图分析不准确？**
- 确保 `AI_VISION_MODEL` 配置了支持图片理解的模型
- 建议使用清晰的截图，避免太小或模糊的图片
- 文字粘贴通常比截图更准确

## License

MIT

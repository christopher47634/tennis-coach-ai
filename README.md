# Tennis Coach AI

网球发球动作分析工具 — 上传视频，获得专业生物力学分析。

## 功能

- 自动识别发球5个阶段（准备→蓄力→加速→击球→随挥）
- 每阶段独立评分 + 中文生物力学评语
- 骨架叠加视频（颜色标注关节偏离程度）
- 职业球员参考骨架动画对比
- 角度对比表（你的值 vs 职业标准）

## 快速开始

### Docker（推荐）

```bash
docker compose up --build
```

打开 http://localhost:8000

### 本地运行

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 使用

1. 上传网球发球视频（mp4等格式）
2. 等待分析（约30-90秒）
3. 查看评分、阶段分析、骨架对比

## 技术栈

- Python + FastAPI
- MediaPipe 0.10.14（姿态估计）
- OpenCV（视频处理）
- 纯HTML/CSS/JS前端（无框架无CDN）

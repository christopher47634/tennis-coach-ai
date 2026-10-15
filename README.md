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

## 配置

评分与分期参数集中在 `config/default.toml`（五阶段权重、good/moderate/needs_work 阈值、分期检测窗口等），
内置缺省值与该文件一致，由 `app/settings.py` 读取。要调参：复制一份，只写想改的键，然后

```bash
TENNIS_CONFIG=/path/to/my.toml python -m uvicorn app.main:app
```

需要 Python 3.11+（使用标准库 `tomllib`）。

## 测试

```bash
pip install -r requirements-dev.txt   # 仅 numpy / pytest / flake8，无需 mediapipe 与 opencv
python -m flake8
python -m pytest
```

回归测试不使用视频：`tests/fixtures/serve_*.json` 是合成的关键点序列（约 48KB），直接喂给
`analyze_landmark_sequence()`，与 `expected_scores.json` 中锁定的得分、阶段边界、平均角度比对
（容差：分数 1、角度 0.5、边界 2 帧）。有意修改评分逻辑后，运行
`python tests/make_fixtures.py --golden` 查看新输出并手动更新期望值。姿态提取（MediaPipe）本身不在测试范围内。

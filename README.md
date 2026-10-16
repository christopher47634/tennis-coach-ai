# Tennis Coach AI

网球发球动作分析工具 — 上传视频，用姿态估计（MediaPipe）提取关节角度，按规则模板给出分阶段的参考评分与建议。

> 说明：评分对照的是经验估计的参考范围（`data/expert_baselines.json`，尚未用职业球员实测数据校准），评语是规则模板而非模型生成，结果仅供训练参考，不构成专业生物力学诊断。机位建议：固定机位、正面或背面、全身入镜、含完整发球。

## 功能

- 以「手腕最高点 = 击球点」向前反推识别发球5个阶段（准备→蓄力→加速→击球→随挥），输出置信度；找不到发球动作时明确报错
- 每阶段按各自的参考范围评分 + 中文规则评语（自动识别左/右手持拍）
- 骨架叠加视频（颜色标注关节偏离程度）
- 示意性的理想发球骨架动画对比
- 角度对比表（你的值 vs 参考范围，击球阶段）

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
2. 等待分析（约30-90秒；上限 200MB、120 秒、4096px，同时最多分析 2 段，叠加视频 1 小时后自动删除）
3. 查看评分、阶段分析、骨架对比

## 技术栈

- Python + FastAPI
- MediaPipe 0.10.14（姿态估计）
- OpenCV（由 mediapipe 依赖带入，视频处理；请勿再装 opencv-python*）
- 纯HTML/CSS/JS前端（无框架无CDN）

## 配置

评分与分期参数集中在 `config/default.toml`（五阶段权重、good/moderate/needs_work 阈值、分期检测窗口等），
内置缺省值与该文件一致，由 `app/settings.py` 读取。要调参：复制一份，只写想改的键，然后

```bash
TENNIS_CONFIG=/path/to/my.toml python -m uvicorn app.main:app
```

需要 Python 3.11+（使用标准库 `tomllib`）。

### 访问控制与限流（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `API_TOKEN` | 未设置 | 设置后 `/api/analyze` 需要 `Authorization: Bearer <token>`（常量时间比较），否则 401；未设置则保持开放，启动日志会警告 |
| `RATE_LIMIT_MAX` | `5` | 每个客户端 IP 在窗口内允许的分析次数，`0` 关闭限流 |
| `RATE_LIMIT_WINDOW_SECONDS` | `600` | 滑动窗口长度（秒） |

超限返回 429 并带 `Retry-After`；`/api/health` 不鉴权不限流。限流为进程内存实现（单 worker，重启清零）。
客户端 IP 取直连地址；放在反向代理后面时用 `uvicorn --proxy-headers --forwarded-allow-ips=<代理IP>`。

前端需要 token 时：访问 `http://host:8000/?token=<token>`，或在浏览器控制台执行
`localStorage.setItem('api_token', '<token>')`。注意 URL 里的 token 会进浏览历史，仅适合个人使用。

```bash
API_TOKEN=secret RATE_LIMIT_MAX=10 docker compose up
curl -H "Authorization: Bearer secret" -F video=@serve.mp4 http://localhost:8000/api/analyze
```

## 测试

```bash
pip install -r requirements-dev.txt   # numpy / pytest / flake8 / fastapi / httpx，无需 mediapipe 与 opencv（姿态模型在测试中被替身代替）
python -m flake8
python -m pytest
```

回归测试不使用视频：`tests/fixtures/serve_*.json` 是合成的关键点序列（约 48KB），直接喂给
`analyze_landmark_sequence()`，与 `expected_scores.json` 中锁定的得分、阶段边界、平均角度比对
（容差：分数 1、角度 0.5、边界 2 帧）。有意修改评分逻辑后，运行
`python tests/make_fixtures.py --write-golden` 重写期望值。另有角度几何、阶段检测（慢放/空镜/缺帧）、上传接口（空文件 400、超限 413、非视频 422、异常文件名不 500）等测试。姿态提取（MediaPipe）本身不在测试范围内。

依赖已锁定（`requirements.txt`，numpy<2）；Docker 镜像以非 root 运行并带 HEALTHCHECK，构建时预下载 pose 重模型权重；CI 会构建镜像并探测 `/api/health`。

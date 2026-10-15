# Changelog

## [Unreleased]

### Fixed（第 4 份审查）
- 角度改用像素坐标；trunk_lateral_tilt 改为肩线倾角，髋肩分离改为肩线/髋线夹角；按阶段给基线；平滑不再把边缘拉向 0；缺帧用 NaN+插值；分母用实际读到的帧数。
- 阶段检测改为「腕最高=击球点」向前反推，相对肩髋归一，输出置信度，失败显式报错；阶段只覆盖发球本身。
- 上传流式写盘并在传输中限 200MB，后缀白名单，threadpool + 信号量，不回显异常，outputs TTL 清理，不再缓存全部帧，ffmpeg 失败有日志且清理临时文件，前端读 detail。
- 锁依赖（numpy<2，单一 cv2），Dockerfile 修复（libgl1、非 root、HEALTHCHECK、.dockerignore、预下载模型）。
- 删除死代码 coach.py 等；评语区分偏大/偏小；README 去掉夸大说法。

### Added
- `config/default.toml` + `app/settings.py`：五阶段权重、评分档位阈值、分期检测参数全部可配置；
  缺省值与此前硬编码完全一致，可用环境变量 `TENNIS_CONFIG` 指向自定义 TOML 局部覆盖。
- `analyze_landmark_sequence()`：把 MediaPipe 之后的纯计算（分期、评分、角度对比）独立出来，不依赖视频与模型。
- 回归测试：合成关键点 JSON 夹具（`tests/fixtures/`）+ 期望得分，评分漂移会使测试失败；另有单元测试与配置测试。
- GitHub Actions：ubuntu/windows x Python 3.11-3.13 矩阵，跑 flake8 + pytest（只装 `requirements-dev.txt`）。

### Changed
- `cv2` / `mediapipe` 改为在视频函数内惰性导入，评分代码仅需 numpy 即可导入。
- 清理 `app/main.py` 中未使用的导入。
- Dockerfile 复制 `config/`。

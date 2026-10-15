# Changelog

## [Unreleased]

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

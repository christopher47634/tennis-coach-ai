(function () {
  'use strict';

  /* ========== 中文关节角度名称映射 ========== */
  var ANGLE_NAMES = {
    right_elbow_flexion: '右肘弯曲角',
    left_elbow_flexion: '左肘弯曲角',
    right_shoulder_abduction: '右肩外展角',
    left_shoulder_abduction: '左肩外展角',
    right_knee_flexion: '右膝弯曲角',
    left_knee_flexion: '左膝弯曲角',
    trunk_lateral_tilt: '躯干侧倾角',
    hip_shoulder_separation: '髋肩分离角',
    wrist_extension: '手腕伸展角',
    right_shoulder_rotation: '右肩旋转角',
    left_shoulder_rotation: '左肩旋转角',
    right_hip_flexion: '右髋弯曲角',
    left_hip_flexion: '左髋弯曲角',
    right_ankle_flexion: '右踝弯曲角',
    left_ankle_flexion: '左踝弯曲角'
  };

  /* ========== 评级中文映射 ========== */
  var RATING_LABELS = {
    good: '优秀',
    moderate: '中等',
    needs_work: '需改进'
  };

  /* ========== 阶段颜色 ========== */
  var PHASE_COLORS = ['#00ff88', '#00ccff', '#ffaa00', '#ff4488', '#aa66ff'];
  var PHASE_BG_COLORS = [
    'rgba(0,255,136,0.15)',
    'rgba(0,204,255,0.15)',
    'rgba(255,170,0,0.15)',
    'rgba(255,68,136,0.15)',
    'rgba(170,102,255,0.15)'
  ];

  /* ========== 骨架连线 ========== */
  var SKELETON_CONNECTIONS = [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],  // arms
    [11, 23], [12, 24], [23, 24],                        // torso
    [23, 25], [25, 27], [24, 26], [26, 28],              // legs
    [0, 11], [0, 12],                                     // neck
    [0, 1], [1, 2], [2, 3], [3, 7],                      // face right
    [0, 4], [4, 5], [5, 6], [6, 8],                      // face left
    [9, 10],                                               // mouth
    [15, 17], [15, 19], [15, 21],                          // right hand
    [16, 18], [16, 20], [16, 22],                          // left hand
    [27, 29], [27, 31],                                    // right foot
    [28, 30], [28, 32]                                     // left foot
  ];

  /* ========== 关节→关键landmark索引（用于颜色标注） ========== */
  var JOINT_HIGHLIGHT_LANDMARKS = {
    right_elbow_flexion:      14,  // right elbow
    left_elbow_flexion:       13,  // left elbow
    right_shoulder_abduction: 12,  // right shoulder
    left_shoulder_abduction:  11,  // left shoulder
    right_knee_flexion:       26,  // right knee
    left_knee_flexion:        25,  // left knee
    trunk_lateral_tilt:       12,  // right shoulder (proxy)
    hip_shoulder_separation:  24,  // right hip
    wrist_extension:          16,  // right wrist
    right_shoulder_rotation:  12,
    left_shoulder_rotation:   11,
    right_hip_flexion:        24,
    left_hip_flexion:         23,
    right_ankle_flexion:      28,
    left_ankle_flexion:       27
  };

  /* ========== 角度标注位置（在骨架上标注的角度及显示位置偏移） ========== */
  var ANGLE_LABEL_CONFIG = [
    { joint: 'right_elbow_flexion',      lm: [12, 14, 16], offset: { x:  20, y: -10 } },
    { joint: 'left_elbow_flexion',       lm: [11, 13, 15], offset: { x: -20, y: -10 } },
    { joint: 'right_shoulder_abduction', lm: [14, 12, 24], offset: { x:  20, y:   0 } },
    { joint: 'left_shoulder_abduction',  lm: [13, 11, 23], offset: { x: -20, y:   0 } },
    { joint: 'right_knee_flexion',       lm: [24, 26, 28], offset: { x:  20, y:   0 } },
    { joint: 'left_knee_flexion',        lm: [23, 25, 27], offset: { x: -20, y:   0 } },
    { joint: 'wrist_extension',          lm: [14, 16, 18], offset: { x:  20, y: -10 } }
  ];

  /* ========== 评分评级 ========== */
  function getRating(score) {
    if (score >= 85) return { text: '优秀', color: '#00ff88' };
    if (score >= 70) return { text: '良好', color: '#00ff88' };
    if (score >= 55) return { text: '中等', color: '#ffaa00' };
    if (score >= 40) return { text: '需改进', color: '#ffaa00' };
    return { text: '待提高', color: '#ff4444' };
  }

  /* ========== DOM 引用 ========== */
  var dropZone      = document.getElementById('drop-zone');
  var videoInput    = document.getElementById('video-input');
  var fileName      = document.getElementById('file-name');
  var analyzeBtn    = document.getElementById('analyze-btn');
  var uploadSection = document.getElementById('upload-section');
  var loadingSection= document.getElementById('loading-section');
  var errorSection  = document.getElementById('error-section');
  var errorMessage  = document.getElementById('error-message');
  var resultsSection= document.getElementById('results-section');
  var overallScoreEl= document.getElementById('overall-score');
  var scoreRingFill = document.getElementById('score-ring-fill');
  var scoreRatingEl = document.getElementById('score-rating');
  var phaseTimeline = document.getElementById('phase-timeline');
  var phaseLabels   = document.getElementById('phase-labels');
  var detailTitle   = document.getElementById('detail-title');
  var detailAngles  = document.getElementById('detail-angles');
  var detailCommentary = document.getElementById('detail-commentary');
  var overlayVideo  = document.getElementById('overlay-video');
  var resetBtn      = document.getElementById('reset-btn');
  var errorRetryBtn = document.getElementById('error-retry-btn');

  // New DOM refs for dual screen
  var refCanvas      = document.getElementById('ref-canvas');
  var refPhaseBadge  = document.getElementById('ref-phase-badge');
  var syncTimeline   = document.getElementById('sync-timeline');
  var syncProgress   = document.getElementById('sync-progress');
  var syncThumb      = document.getElementById('sync-thumb');
  var syncCurrent    = document.getElementById('sync-current');
  var syncDuration   = document.getElementById('sync-duration');
  var angleCompareBody = document.getElementById('angle-compare-body');

  var refCtx = refCanvas.getContext('2d');

  var selectedFile = null;
  var analysisData = null;
  var selectedPhaseIdx = 0;
  var videoFrameRate = 30;

  // Reference skeleton animation state
  var refFrames = [];
  var refFps = 30;
  var refTotalFrames = 120;
  var refAnimId = null;
  var currentRefFrameIdx = 0;
  var angleComparison = {};
  var landmarkColorMap = {}; // landmark_idx -> color

  /* ========== 文件选择 ========== */
  function selectFile(file) {
    if (!file) return;
    selectedFile = file;
    fileName.textContent = file.name;
    analyzeBtn.disabled = false;
  }

  videoInput.addEventListener('change', function () {
    selectFile(this.files[0]);
  });

  // 拖拽上传
  dropZone.addEventListener('dragover', function (e) {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });
  dropZone.addEventListener('dragleave', function () {
    dropZone.classList.remove('dragover');
  });
  dropZone.addEventListener('drop', function (e) {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) selectFile(e.dataTransfer.files[0]);
  });
  dropZone.addEventListener('click', function () {
    videoInput.click();
  });

  /* ========== 分析请求 ========== */
  analyzeBtn.addEventListener('click', async function () {
    if (!selectedFile) return;

    uploadSection.classList.add('hidden');
    errorSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    loadingSection.classList.remove('hidden');

    var form = new FormData();
    form.append('video', selectedFile);

    try {
      var res = await fetch('/api/analyze', { method: 'POST', body: form });
      var data = await res.json();

      if (!res.ok || data.error) {
        showError(data.error || '分析失败，请重试。');
        return;
      }
      showResults(data);
    } catch (err) {
      showError('网络错误 — 请确认服务器是否正在运行？');
    }
  });

  /* ========== 显示结果 ========== */
  function showResults(data) {
    loadingSection.classList.add('hidden');
    resultsSection.classList.remove('hidden');
    analysisData = data;
    selectedPhaseIdx = 0;

    videoFrameRate = 30;

    // --- 总评分动画 ---
    var score = data.overall_score || 0;
    var rating = getRating(score);
    animateScore(score, rating);

    // --- 构建阶段时间轴 ---
    buildTimeline(data.phases || [], data.total_frames || 1);

    // --- 显示第一阶段详情 ---
    if (data.phases && data.phases.length > 0) {
      selectPhase(0);
    } else {
      detailTitle.textContent = '暂无阶段数据';
      detailAngles.innerHTML = '';
      detailCommentary.textContent = '';
    }

    // --- 视频 ---
    if (data.overlay_video_url) {
      overlayVideo.src = data.overlay_video_url;
      overlayVideo.style.display = '';
      overlayVideo.addEventListener('loadedmetadata', function onMeta() {
        overlayVideo.removeEventListener('loadedmetadata', onMeta);
        if (data.total_frames && overlayVideo.duration) {
          videoFrameRate = data.total_frames / overlayVideo.duration;
        }
        buildTimeline(data.phases || [], data.total_frames || 1);
        syncDuration.textContent = formatTime(overlayVideo.duration);
      });
    } else {
      overlayVideo.style.display = 'none';
    }

    // --- 视频时间更新 ---
    overlayVideo.addEventListener('timeupdate', onVideoTimeUpdate);
    overlayVideo.addEventListener('timeupdate', onSyncUpdate);

    // --- 角度对比表 ---
    angleComparison = data.angle_comparison || {};
    buildAngleComparisonTable(angleComparison);

    // --- 参考骨架 ---
    var refSkel = data.reference_skeleton || {};
    refFrames = refSkel.frames || [];
    refFps = refSkel.fps || 30;
    refTotalFrames = refSkel.total_frames || (refFrames.length || 120);

    // Build landmark color map
    buildLandmarkColorMap(angleComparison);

    // Size canvas
    sizeRefCanvas();
    // Draw first frame
    if (refFrames.length > 0) {
      drawRefFrame(0);
    }

    resultsSection.scrollIntoView({ behavior: 'smooth' });
  }

  /* ========== 评分动画 ========== */
  function animateScore(targetScore, rating) {
    var circumference = 2 * Math.PI * 52;
    scoreRingFill.style.transition = 'none';
    scoreRingFill.style.strokeDashoffset = circumference;
    overallScoreEl.textContent = '0';

    void scoreRingFill.offsetWidth;

    scoreRingFill.style.transition = 'stroke-dashoffset 1.2s ease, stroke 0.5s';
    scoreRingFill.style.strokeDashoffset = circumference - (targetScore / 100) * circumference;
    scoreRingFill.style.stroke = rating.color;
    overallScoreEl.style.color = rating.color;
    scoreRatingEl.textContent = rating.text;
    scoreRatingEl.style.color = rating.color;

    var duration = 1200;
    var startTime = null;
    function tick(ts) {
      if (!startTime) startTime = ts;
      var elapsed = ts - startTime;
      var progress = Math.min(elapsed / duration, 1);
      var eased = 1 - Math.pow(1 - progress, 3);
      overallScoreEl.textContent = Math.round(eased * targetScore);
      if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  /* ========== 构建阶段时间轴 ========== */
  function buildTimeline(phases, totalFrames) {
    phaseTimeline.innerHTML = '';
    phaseLabels.innerHTML = '';

    if (!phases.length) return;

    phases.forEach(function (phase, i) {
      var widthPct = ((phase.end_frame - phase.start_frame) / totalFrames) * 100;
      var seg = document.createElement('div');
      seg.className = 'phase-segment';
      seg.style.width = widthPct + '%';
      seg.style.background = PHASE_COLORS[i % PHASE_COLORS.length];
      seg.setAttribute('data-idx', i);

      if (widthPct > 8) {
        seg.innerHTML = '<span>' + phase.name + '</span>';
        if (widthPct > 15) {
          seg.innerHTML += ' <span class="phase-score-badge">' + phase.score + '分</span>';
        }
      }

      seg.addEventListener('click', function (e) {
        e.stopPropagation();
        selectPhase(i);
        jumpVideoToPhase(i);
      });

      phaseTimeline.appendChild(seg);

      var label = document.createElement('div');
      label.className = 'phase-label';
      label.style.width = widthPct + '%';
      label.textContent = phase.name;
      label.setAttribute('data-idx', i);
      label.addEventListener('click', function () {
        selectPhase(i);
        jumpVideoToPhase(i);
      });
      phaseLabels.appendChild(label);
    });
  }

  /* ========== 选中阶段 ========== */
  function selectPhase(idx) {
    selectedPhaseIdx = idx;
    var phases = analysisData.phases;
    var phase = phases[idx];

    // 更新时间轴高亮
    var segments = phaseTimeline.querySelectorAll('.phase-segment');
    var labels = phaseLabels.querySelectorAll('.phase-label');
    segments.forEach(function (seg, i) {
      seg.classList.toggle('active', i === idx);
    });
    labels.forEach(function (lbl, i) {
      lbl.classList.toggle('active', i === idx);
    });

    // 更新详情标题
    var phaseRating = getRating(phase.score);
    detailTitle.innerHTML = phase.name +
      ' <span style="color:' + phaseRating.color + ';font-size:0.9em;">(' + phase.score + '分 · ' + phaseRating.text + ')</span>';

    // 角度表格
    detailAngles.innerHTML = '';
    if (phase.angles) {
      Object.keys(phase.angles).forEach(function (key) {
        var val = phase.angles[key];
        var name = ANGLE_NAMES[key] || key;
        var item = document.createElement('div');
        item.className = 'angle-item';
        item.innerHTML =
          '<span class="angle-name">' + name + '</span>' +
          '<span class="angle-value">' + val.toFixed(1) + '°</span>';
        detailAngles.appendChild(item);
      });
    }

    // 评语
    detailCommentary.textContent = phase.commentary || '暂无评语';

    // 更新视频进度条高亮
    updateVideoProgressHighlight(idx);
  }

  /* ========== 跳转视频到某阶段 ========== */
  function jumpVideoToPhase(idx) {
    var phase = analysisData.phases[idx];
    if (!phase || !overlayVideo.duration || !analysisData.total_frames) return;
    var fps = analysisData.total_frames / overlayVideo.duration;
    var startTime = phase.start_frame / fps;
    overlayVideo.currentTime = startTime;
    overlayVideo.play();
  }

  /* ========== 视频时间更新 → 自动切换阶段高亮 ========== */
  function onVideoTimeUpdate() {
    if (!analysisData || !analysisData.phases || !overlayVideo.duration) return;
    var currentFrame = (overlayVideo.currentTime / overlayVideo.duration) * (analysisData.total_frames || 1);

    var phases = analysisData.phases;
    for (var i = 0; i < phases.length; i++) {
      if (currentFrame >= phases[i].start_frame && currentFrame <= phases[i].end_frame) {
        if (selectedPhaseIdx !== i) {
          selectPhase(i);
        }
        break;
      }
    }
  }

  /* ========== 同步更新：视频 → 参考骨架 + 时间轴 ========== */
  function onSyncUpdate() {
    if (!overlayVideo.duration) return;

    var progress = overlayVideo.currentTime / overlayVideo.duration;
    progress = Math.max(0, Math.min(1, progress));

    // Update sync timeline
    syncProgress.style.width = (progress * 100) + '%';
    syncThumb.style.left = (progress * 100) + '%';
    syncCurrent.textContent = formatTime(overlayVideo.currentTime);

    // Update reference skeleton frame
    if (refFrames.length > 0) {
      var frameIdx = Math.floor(progress * (refTotalFrames - 1));
      frameIdx = Math.max(0, Math.min(frameIdx, refTotalFrames - 1));
      if (frameIdx !== currentRefFrameIdx) {
        currentRefFrameIdx = frameIdx;
        drawRefFrame(frameIdx);
      }
    }
  }

  /* ========== 同步时间轴交互 ========== */
  syncTimeline.addEventListener('click', function (e) {
    if (!overlayVideo.duration) return;
    var rect = syncTimeline.getBoundingClientRect();
    var ratio = (e.clientX - rect.left) / rect.width;
    ratio = Math.max(0, Math.min(1, ratio));
    overlayVideo.currentTime = ratio * overlayVideo.duration;
    overlayVideo.play();
  });

  /* ========== 格式化时间 ========== */
  function formatTime(seconds) {
    var m = Math.floor(seconds / 60);
    var s = Math.floor(seconds % 60);
    return m + ':' + (s < 10 ? '0' : '') + s;
  }

  /* ========== 角度对比表 ========== */
  function buildAngleComparisonTable(comparison) {
    angleCompareBody.innerHTML = '';
    if (!comparison || !Object.keys(comparison).length) {
      var emptyRow = document.createElement('tr');
      emptyRow.innerHTML = '<td colspan="5" style="text-align:center;color:var(--text-muted);padding:1.5rem;">暂无对比数据</td>';
      angleCompareBody.appendChild(emptyRow);
      return;
    }

    // Sort: needs_work first, then moderate, then good
    var ratingOrder = { needs_work: 0, moderate: 1, good: 2 };
    var keys = Object.keys(comparison).sort(function (a, b) {
      return (ratingOrder[comparison[a].rating] || 0) - (ratingOrder[comparison[b].rating] || 0);
    });

    keys.forEach(function (joint) {
      var c = comparison[joint];
      var name = ANGLE_NAMES[joint] || joint;
      var ratingText = RATING_LABELS[c.rating] || c.rating;

      // Bar fill: good=100%, moderate=60%, needs_work=30%
      var barPct = c.rating === 'good' ? 100 : c.rating === 'moderate' ? 60 : 30;

      var tr = document.createElement('tr');
      tr.innerHTML =
        '<td class="joint-name">' + name + '</td>' +
        '<td class="val-user" style="color:' + c.color + '">' + c.user_value.toFixed(1) + '°</td>' +
        '<td class="val-expert">' + c.expert_value.toFixed(1) + '°</td>' +
        '<td class="val-deviation" style="color:' + c.color + '">±' + c.deviation.toFixed(1) + '°</td>' +
        '<td>' +
          '<div class="rating-cell">' +
            '<div class="rating-bar-track"><div class="rating-bar-fill" style="width:' + barPct + '%;background:' + c.color + '"></div></div>' +
            '<span class="rating-label" style="color:' + c.color + '">' + ratingText + '</span>' +
          '</div>' +
        '</td>';
      angleCompareBody.appendChild(tr);
    });
  }

  /* ========== 骨架颜色映射 ========== */
  function buildLandmarkColorMap(comparison) {
    landmarkColorMap = {};
    for (var joint in comparison) {
      var lmIdx = JOINT_HIGHLIGHT_LANDMARKS[joint];
      if (lmIdx !== undefined) {
        landmarkColorMap[lmIdx] = comparison[joint].color;
      }
    }
  }

  /* ========== Canvas 尺寸 ========== */
  function sizeRefCanvas() {
    var container = refCanvas.parentElement;
    var w = container.clientWidth;
    var h = container.clientHeight || Math.round(w * 10 / 16);
    // Set actual pixel size for sharpness
    var dpr = window.devicePixelRatio || 1;
    refCanvas.width = w * dpr;
    refCanvas.height = h * dpr;
    refCanvas.style.width = w + 'px';
    refCanvas.style.height = h + 'px';
    refCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /* ========== 绘制参考骨架帧 ========== */
  function drawRefFrame(frameIdx) {
    if (!refFrames || frameIdx < 0 || frameIdx >= refFrames.length) return;

    var frame = refFrames[frameIdx];
    var landmarks = frame.landmarks;
    if (!landmarks || landmarks.length < 33) return;

    var w = refCanvas.width / (window.devicePixelRatio || 1);
    var h = refCanvas.height / (window.devicePixelRatio || 1);

    // Clear
    refCtx.clearRect(0, 0, w, h);

    // Dark background with subtle gradient
    var grad = refCtx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, '#0d1117');
    grad.addColorStop(1, '#0a0e14');
    refCtx.fillStyle = grad;
    refCtx.fillRect(0, 0, w, h);

    // Subtle grid
    refCtx.strokeStyle = 'rgba(255,255,255,0.03)';
    refCtx.lineWidth = 1;
    for (var gx = 0; gx < w; gx += 30) {
      refCtx.beginPath();
      refCtx.moveTo(gx, 0);
      refCtx.lineTo(gx, h);
      refCtx.stroke();
    }
    for (var gy = 0; gy < h; gy += 30) {
      refCtx.beginPath();
      refCtx.moveTo(0, gy);
      refCtx.lineTo(w, gy);
      refCtx.stroke();
    }

    // Convert normalized coords to canvas pixels
    // Landmarks are in [0,1] range; mirror X for natural view
    function toCanvas(lm) {
      return {
        x: (1 - lm.x) * w,
        y: lm.y * h
      };
    }

    // Draw connections
    SKELETON_CONNECTIONS.forEach(function (conn) {
      var a = landmarks[conn[0]];
      var b = landmarks[conn[1]];
      if (!a || !b) return;
      var pa = toCanvas(a);
      var pb = toCanvas(b);

      // Determine line color: if either endpoint is a highlighted joint, use that color
      var colorA = landmarkColorMap[conn[0]];
      var colorB = landmarkColorMap[conn[1]];
      var lineColor = colorA || colorB || 'rgba(0,255,136,0.35)';

      refCtx.beginPath();
      refCtx.moveTo(pa.x, pa.y);
      refCtx.lineTo(pb.x, pb.y);
      refCtx.strokeStyle = lineColor;
      refCtx.lineWidth = 2;
      refCtx.stroke();
    });

    // Draw landmarks
    for (var i = 0; i < 33; i++) {
      var lm = landmarks[i];
      if (!lm) continue;
      var p = toCanvas(lm);

      var isHighlighted = landmarkColorMap[i] !== undefined;
      var dotColor = landmarkColorMap[i] || 'rgba(0,255,136,0.7)';
      var radius = isHighlighted ? 5 : 3;

      // Glow for highlighted joints
      if (isHighlighted) {
        refCtx.beginPath();
        refCtx.arc(p.x, p.y, radius + 4, 0, Math.PI * 2);
        refCtx.fillStyle = dotColor.replace(')', ',0.2)').replace('rgb', 'rgba');
        // Handle hex colors for glow
        refCtx.globalAlpha = 0.2;
        refCtx.fillStyle = dotColor;
        refCtx.fill();
        refCtx.globalAlpha = 1;
      }

      refCtx.beginPath();
      refCtx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      refCtx.fillStyle = dotColor;
      refCtx.fill();

      // White border for highlighted
      if (isHighlighted) {
        refCtx.strokeStyle = '#fff';
        refCtx.lineWidth = 1.5;
        refCtx.stroke();
      }
    }

    // Draw angle labels on skeleton
    if (frame.angles) {
      drawAngleLabels(landmarks, frame.angles, w, h);
    }

    // Update phase badge
    if (frame.phase) {
      var phaseIdx = getPhaseIndex(frame.phase);
      refPhaseBadge.textContent = frame.phase;
      refPhaseBadge.style.color = PHASE_COLORS[phaseIdx] || 'var(--accent)';
      refPhaseBadge.style.borderColor = PHASE_COLORS[phaseIdx] || 'var(--border)';
    }
  }

  /* ========== 在骨架上标注角度值 ========== */
  function drawAngleLabels(landmarks, angles, w, h) {
    function toCanvas(lm) {
      return { x: (1 - lm.x) * w, y: lm.y * h };
    }

    ANGLE_LABEL_CONFIG.forEach(function (cfg) {
      var angleVal = angles[cfg.joint];
      if (angleVal === undefined) return;

      // Position: midpoint of the 3 landmarks, offset slightly
      var lm1 = landmarks[cfg.lm[0]];
      var lm2 = landmarks[cfg.lm[1]]; // center joint
      var lm3 = landmarks[cfg.lm[2]];
      if (!lm1 || !lm2 || !lm3) return;

      var center = toCanvas(lm2);
      var labelX = center.x + cfg.offset.x;
      var labelY = center.y + cfg.offset.y;

      var color = angleComparison[cfg.joint] ? angleComparison[cfg.joint].color : '#00ff88';
      var text = Math.round(angleVal) + '°';

      // Background pill
      refCtx.font = '600 11px -apple-system, "PingFang SC", sans-serif';
      var metrics = refCtx.measureText(text);
      var tw = metrics.width + 10;
      var th = 18;

      refCtx.fillStyle = 'rgba(0,0,0,0.75)';
      roundRect(refCtx, labelX - tw / 2, labelY - th / 2, tw, th, 9);
      refCtx.fill();

      refCtx.strokeStyle = color;
      refCtx.lineWidth = 1;
      roundRect(refCtx, labelX - tw / 2, labelY - th / 2, tw, th, 9);
      refCtx.stroke();

      // Text
      refCtx.fillStyle = color;
      refCtx.textAlign = 'center';
      refCtx.textBaseline = 'middle';
      refCtx.fillText(text, labelX, labelY);

      // Connector line from joint to label
      refCtx.beginPath();
      refCtx.moveTo(center.x, center.y);
      refCtx.lineTo(labelX, labelY);
      refCtx.strokeStyle = color;
      refCtx.globalAlpha = 0.3;
      refCtx.lineWidth = 1;
      refCtx.stroke();
      refCtx.globalAlpha = 1;
    });
  }

  /* ========== 圆角矩形路径 ========== */
  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  /* ========== 阶段名称→索引 ========== */
  function getPhaseIndex(phaseName) {
    var names = ['准备期', '蓄力期', '加速期', '击球期', '随挥期'];
    var idx = names.indexOf(phaseName);
    return idx >= 0 ? idx : 0;
  }

  /* ========== 视频进度条（阶段色带） ========== */
  function buildVideoProgressBar(phases, totalFrames) {
    // No longer needed - we have the sync timeline now
    // Keep function stub for compatibility
  }

  function updateVideoProgressHighlight(idx) {
    // No longer needed
  }

  /* ========== 窗口resize → 重新绘制 ========== */
  window.addEventListener('resize', function () {
    if (refFrames.length > 0) {
      sizeRefCanvas();
      drawRefFrame(currentRefFrameIdx);
    }
  });

  /* ========== 错误 ========== */
  function showError(msg) {
    loadingSection.classList.add('hidden');
    errorSection.classList.remove('hidden');
    errorMessage.textContent = msg;
  }

  /* ========== 重置 ========== */
  function resetAll() {
    selectedFile = null;
    analysisData = null;
    selectedPhaseIdx = 0;
    currentRefFrameIdx = 0;
    refFrames = [];
    angleComparison = {};
    landmarkColorMap = {};
    videoInput.value = '';
    fileName.textContent = '';
    analyzeBtn.disabled = true;
    resultsSection.classList.add('hidden');
    errorSection.classList.add('hidden');
    loadingSection.classList.add('hidden');
    uploadSection.classList.remove('hidden');
    overlayVideo.removeEventListener('timeupdate', onVideoTimeUpdate);
    overlayVideo.removeEventListener('timeupdate', onSyncUpdate);
    overlayVideo.pause();
    overlayVideo.src = '';
    syncProgress.style.width = '0%';
    syncThumb.style.left = '0%';
    syncCurrent.textContent = '0:00';
    syncDuration.textContent = '0:00';
    angleCompareBody.innerHTML = '';
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  resetBtn.addEventListener('click', resetAll);
  errorRetryBtn.addEventListener('click', resetAll);
})();

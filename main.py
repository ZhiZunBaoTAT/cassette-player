"""
透明磁带音乐播放器 — Cassette Player  v3.0
─────────────────────────────────────────────────────────────
v3.0 新增：
  • 变径磁带轮：左轮随进度缩小，右轮随进度增大
  • 带体弧线：观察窗内带子用贝塞尔曲线模拟绷紧张力
  • 机身磨砂纹理：细点噪声叠加，塑料质感代替玻璃感
  • 歌曲切换过渡动画：快进效果（轮子高速旋转 0.4s + 标题滑入）
  • 暂停松弛动画：带体曲率加大 + 轮速缓慢衰减
  • 磁头指示 LED：播放绿/暂停橙/停止灰，低频能量驱动亮度呼吸
  • 拖拽进度时浮动时间气泡
  • 播放模式切换：顺序 / 单曲循环 / 随机（持久化）
  • 歌词滚动：自动读取同目录 .lrc 文件，标签区下方滚动显示
  • 播放列表侧边栏：左边缘滑入/滑出，显示所有歌曲可点击切换
  • 帧率自适应：监测 tick 耗时，低端机自动降频
  继承 v2.0 全部功能
"""

import sys, os, math, random, threading, io, time
from pathlib import Path

import numpy as np
from scipy.fft import rfft
from pydub import AudioSegment

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QFileDialog
from PyQt6.QtCore    import Qt, QTimer, QRectF, QPointF, QUrl, QSettings, QEasingCurve
from PyQt6.QtGui     import (QPainter, QColor, QBrush, QPen, QFont,
                              QLinearGradient, QPainterPath,
                              QPixmap, QImage)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

from mutagen import File as MutagenFile
from mutagen.mp3 import MP3


# ════════════════════════════════════════════════════════════
#  LRC 解析器
# ════════════════════════════════════════════════════════════

class LrcParser:
    """解析 [mm:ss.xx] 格式的歌词文件，返回 (ms, text) 列表"""

    @staticmethod
    def load(filepath):
        lrc_path = Path(filepath).with_suffix('.lrc')
        if not lrc_path.exists():
            return []
        lines = []
        try:
            with open(lrc_path, encoding='utf-8', errors='ignore') as f:
                for raw in f:
                    raw = raw.strip()
                    # 匹配 [mm:ss.xx] 或 [mm:ss]
                    import re
                    for m in re.finditer(r'\[(\d+):(\d+)(?:[.:](\d+))?\]', raw):
                        mins  = int(m.group(1))
                        secs  = int(m.group(2))
                        frac  = int(m.group(3) or 0)
                        ms    = (mins * 60 + secs) * 1000 + frac * 10
                        text  = raw[m.end():].strip()
                        # 去掉剩余时间标签
                        text  = re.sub(r'\[\d+:\d+(?:[.:]\d+)?\]', '', text).strip()
                        if text:
                            lines.append((ms, text))
        except Exception:
            pass
        lines.sort(key=lambda x: x[0])
        return lines

    @staticmethod
    def current_line(lines, pos_ms):
        """返回当前播放位置对应的歌词行索引"""
        if not lines:
            return -1
        idx = 0
        for i, (ms, _) in enumerate(lines):
            if ms <= pos_ms:
                idx = i
            else:
                break
        return idx


# ════════════════════════════════════════════════════════════
#  频谱解码器
# ════════════════════════════════════════════════════════════

class SpectrumDecoder:
    def __init__(self, fft_size=1024, target_rate=22050):
        self._fft_size    = fft_size
        self._target_rate = target_rate
        self._pcm         = None
        self._sample_rate = 0
        self._total_samples = 0
        self._ready       = False
        self._current_file = None
        self._lock        = threading.Lock()
        self._window      = None
        self._bin_map     = None
        self._bin_counts  = None
        self._bar_count   = 0

    @property
    def ready(self): return self._ready

    def is_current(self, fp): return self._current_file == fp

    def reset(self):
        with self._lock:
            self._pcm = None; self._sample_rate = 0
            self._total_samples = 0; self._ready = False
            self._current_file = None
            self._window = self._bin_map = self._bin_counts = None
            self._bar_count = 0

    def load_async(self, fp):
        if self.is_current(fp) and self._ready: return
        self.reset()
        threading.Thread(target=self._load, args=(fp,), daemon=True).start()

    def _load(self, fp):
        try:
            audio = AudioSegment.from_file(fp)
            if audio.frame_rate > self._target_rate:
                audio = audio.set_frame_rate(self._target_rate)
            audio   = audio.set_channels(1)
            sr      = audio.frame_rate
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            samples /= float(2 ** (audio.sample_width * 8 - 1))
            with self._lock:
                self._pcm = samples; self._sample_rate = sr
                self._total_samples = len(samples)
                self._current_file = fp; self._ready = True
                if self._window is None or len(self._window) != self._fft_size:
                    self._window = np.hanning(self._fft_size).astype(np.float32)
        except Exception as e:
            print(f"[Decoder] 失败: {fp} — {e}")

    def get_spectrum(self, pos_ms, bar_count):
        if not self._ready or self._pcm is None or bar_count < 1: return None
        if self._bin_map is None or self._bar_count != bar_count:
            self._build_bin_map(bar_count)
        sr    = self._sample_rate; total = self._total_samples; n = self._fft_size
        idx   = max(0, min(int(pos_ms / 1000.0 * sr), total - n))
        win   = (self._pcm[idx:idx+n] * self._window if idx+n <= total
                 else np.pad(self._pcm[idx:total] * self._window[:total-idx],
                             (0, n-(total-idx))))
        mag   = np.abs(rfft(win))
        sums  = np.bincount(self._bin_map, weights=mag, minlength=bar_count)
        with np.errstate(divide='ignore', invalid='ignore'):
            bars = sums / np.maximum(self._bin_counts, 1)
        bars  = np.sqrt(bars)
        mx    = bars.max()
        if mx > 0: bars /= mx
        return np.clip(bars, 0.05, 1.0).tolist()

    def get_waveform(self, pos_ms, n):
        if not self._ready or self._pcm is None: return None
        sr    = self._sample_rate; total = self._total_samples
        idx   = int(pos_ms / 1000.0 * sr)
        start = max(0, idx - n // 2); end = min(total, start + n)
        chunk = self._pcm[start:end]
        if len(chunk) < n:
            p = np.zeros(n, dtype=np.float32); p[:len(chunk)] = chunk; return p.tolist()
        return chunk.tolist()

    def get_bass_energy(self, pos_ms):
        if not self._ready or self._pcm is None: return 0.0
        sr    = self._sample_rate; total = self._total_samples; n = self._fft_size
        idx   = max(0, min(int(pos_ms / 1000.0 * sr), total - n))
        win   = (self._pcm[idx:idx+n] * self._window if idx+n <= total
                 else np.pad(self._pcm[idx:total] * self._window[:total-idx],
                             (0, n-(total-idx))))
        mag   = np.abs(rfft(win))
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        bass  = mag[freqs < 200]
        return float(bass.mean()) if len(bass) > 0 else 0.0

    def _build_bin_map(self, bar_count):
        sr     = self._sample_rate
        freqs  = np.fft.rfftfreq(self._fft_size, 1.0 / sr)
        n_bins = len(freqs)
        edges  = np.logspace(np.log10(30.0), np.log10(min(sr/2.0, 10000.0)), bar_count+1)
        bmap   = np.zeros(n_bins, dtype=np.int32)
        bcnt   = np.zeros(bar_count, dtype=np.int32)
        for bi in range(n_bins):
            b = max(0, min(int(np.searchsorted(edges, freqs[bi]))-1, bar_count-1))
            bmap[bi] = b; bcnt[b] += 1
        self._bin_map    = bmap
        self._bin_counts = np.maximum(bcnt, 1)
        self._bar_count  = bar_count


# ════════════════════════════════════════════════════════════
#  音频引擎
# ════════════════════════════════════════════════════════════

class AudioEngine:
    PLAY_ORDER  = 0   # 顺序
    PLAY_SINGLE = 1   # 单曲循环
    PLAY_RANDOM = 2   # 随机

    def __init__(self, parent=None):
        self._player  = QMediaPlayer(parent)
        self._audio   = QAudioOutput(parent)
        self._player.setAudioOutput(self._audio)
        self._audio.setVolume(0.8)
        self._decoder = SpectrumDecoder()
        self._playlist = []
        self._index    = -1
        self._playing  = False
        self._muted    = False
        self._play_mode = AudioEngine.PLAY_ORDER
        self._player.playbackStateChanged.connect(self._on_state)
        self._player.mediaStatusChanged.connect(self._on_media_status)
        self._auto_next_cb = None

    def _on_state(self, s):
        self._playing = (s == QMediaPlayer.PlaybackState.PlayingState)

    def _on_media_status(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            if self._auto_next_cb: self._auto_next_cb()

    # ── 属性 ──
    @property
    def playing(self):       return self._playing
    @property
    def current_index(self): return self._index
    @property
    def playlist(self):      return self._playlist
    @property
    def decoder_ready(self): return self._decoder.ready
    @property
    def muted(self):         return self._muted
    @property
    def volume(self):        return self._audio.volume()
    @property
    def play_mode(self):     return self._play_mode

    def position(self):  return self._player.position()
    def duration(self):  return self._player.duration()
    def seek(self, ms):  self._player.setPosition(ms)

    def set_volume(self, v):
        self._audio.setVolume(max(0.0, min(1.0, v)))

    def toggle_mute(self):
        self._muted = not self._muted
        self._audio.setMuted(self._muted)

    def cycle_play_mode(self):
        self._play_mode = (self._play_mode + 1) % 3

    def get_spectrum(self, pos, n):   return self._decoder.get_spectrum(pos, n)
    def get_bass_energy(self, pos):   return self._decoder.get_bass_energy(pos)

    def load_folder(self, folder):
        exts = {'.mp3','.flac','.wav','.ogg','.m4a','.aac'}
        self._playlist = sorted(str(f) for ext in exts
                                for f in Path(folder).rglob(f'*{ext}'))
        return len(self._playlist)

    def play_index(self, idx):
        if 0 <= idx < len(self._playlist):
            path = self._playlist[idx]
            self._player.setSource(QUrl.fromLocalFile(path))
            self._player.play()
            self._playing = True; self._index = idx
            if not self._decoder.is_current(path):
                self._decoder.load_async(path)
            return True
        return False

    def toggle(self):
        if self._playing: self._player.pause()
        else:             self._player.play()

    def stop(self):
        self._player.stop(); self._playing = False

    def next(self):
        if not self._playlist: return False
        if self._play_mode == AudioEngine.PLAY_SINGLE:
            return self.play_index(self._index)
        elif self._play_mode == AudioEngine.PLAY_RANDOM:
            idx = random.randint(0, len(self._playlist)-1)
            return self.play_index(idx)
        else:
            return self.play_index((self._index + 1) % len(self._playlist))

    def prev(self):
        if not self._playlist: return False
        if self._play_mode == AudioEngine.PLAY_RANDOM:
            idx = random.randint(0, len(self._playlist)-1)
            return self.play_index(idx)
        return self.play_index((self._index - 1) % len(self._playlist))

    def cleanup(self): self._decoder.reset()

    @staticmethod
    def get_metadata(fp):
        meta = {'title': Path(fp).stem, 'artist': 'Unknown', 'path': fp, 'cover': None}
        try:
            tag = MutagenFile(fp, easy=False)
            if tag is None: return meta
            for k in ('TIT2', '\xa9nam'):
                if k in tag: meta['title'] = str(tag[k]); break
            for k in ('TPE1', '\xa9ART'):
                if k in tag: meta['artist'] = str(tag[k]); break
            for k, v in tag.items():
                if 'APIC' in k or k in ('covr',):
                    data = v.data if hasattr(v,'data') else (v[0] if isinstance(v,list) else None)
                    if data:
                        img = QImage.fromData(bytes(data))
                        if not img.isNull():
                            meta['cover'] = QPixmap.fromImage(img)
                    break
        except Exception:
            pass
        return meta


# ════════════════════════════════════════════════════════════
#  磁带播放器主控件
# ════════════════════════════════════════════════════════════

class CassettePlayer(QWidget):

    def __init__(self):
        super().__init__()
        self.audio = AudioEngine(self)
        self.audio._auto_next_cb = self._auto_next

        self._settings = QSettings("CassettePlayer", "CassettePlayer")

        # ── 可视化 ──
        self._viz_style       = int(self._settings.value("viz_style", 0))
        self._viz_erase_pending = False

        # ── 频谱柱 ──
        self._bar_count   = 60
        self._bars        = [0.05] * self._bar_count
        self._bar_targets = [0.05] * self._bar_count
        self._particles   = []
        self._bar_frame   = 0
        self._hue_offset  = 0.0

        # ── 磁带轮（变径 + 差速 + 节拍）──
        self._left_angle   = 0.0
        self._right_angle  = 0.0
        self._reel_speed   = 1.0    # 速度倍率（过渡动画用）
        self._beat_boost   = 0.0
        self._prev_bass    = 0.0
        self._beat_cooldown = 0

        # ── 磁带观察窗跳动音符 ──
        self._tape_notes = []   # [{x, y, vy, life, ch, sz}]

        # ── 暂停松弛动画 ──
        # 0.0 = 正常绷紧，1.0 = 完全松弛
        self._slack        = 0.0

        # ── 歌曲切换过渡动画 ──
        # _transition > 0 表示正在过渡，值从 TRANS_FRAMES 递减到 0
        self._TRANS_FRAMES  = 14
        self._transition    = 0
        self._title_slide_x = 0.0   # 标题滑入偏移量（像素）

        # ── 磁头 LED 呼吸 ──
        self._led_phase    = 0.0    # 0~2π，呼吸相位

        # ── 进度条拖拽气泡 ──
        self._seek_bubble_x = None  # 拖拽中的 x 坐标
        self._seek_bubble_t = None  # 对应时间 ms

        # ── 播放模式（从设置恢复）──
        mode = int(self._settings.value("play_mode", 0))
        self.audio._play_mode = mode

        # ── 播放列表侧边栏 ──
        self._sidebar_open   = False
        self._sidebar_frac   = 0.0   # 0.0=关闭，1.0=完全展开
        self._sidebar_hover  = -1    # 鼠标悬停的列表项索引

        # ── 歌词 ──
        self._lrc_lines = []   # [(ms, text), ...]
        self._lrc_idx   = -1   # 当前行

        # ── 音量条拖拽 ──
        self._vol_dragging = False
        self._vol_bar_rect = None

        # ── 拖拽 / 缩放 ──
        self._drag_start    = None
        self._resize_corner = None
        self._seeking       = False

        # ── 歌曲信息 ──
        self._track_title  = "未播放"
        self._track_artist = "请打开音乐文件夹"
        self._cover_pixmap = None

        # ── 帧率自适应 ──
        self._tick_interval  = 30    # ms，动态调整
        self._tick_slow_count = 0    # 连续慢帧计数

        # ── 频谱可视化方法表（预建，避免每帧创建列表）──
        self._viz_methods = [
            self._draw_bars, self._draw_radar, self._draw_waveform,
            self._draw_mirror, self._draw_particles, self._draw_pulse,
            self._draw_tape_ripple,
        ]

        # ── 动画定时器 ──
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._tick)
        self._anim_timer.start(self._tick_interval)

        self._setup_ui()
        self._restore_state()

    def _setup_ui(self):
        self.setMinimumSize(500, 320)
        self.setMouseTracking(True)
        self.setStyleSheet("background: transparent;")
        # 与 MainWindow 一致：禁止系统预填充，防止首帧烙印
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._btn_play_text = "▶"
        self._btn_regions   = []
        self._btn_hover     = -1

    def resizeEvent(self, e):
        super().resizeEvent(e)

    def _reel_center_y(self):
        w  = self.width(); s = w / 680
        mg = int(18 * s)
        ly = mg + int(8 * s); lh = int(68 * s)
        wh = int(78 * s); wb = int(6 * s); ps = int(14 * s)
        cb = self.height() - mg
        ct = cb - wb - wh - ps
        return ly + lh + (ct - ly - lh) // 2

    # ════════════════════════════════════════════════════════
    #  动画循环（帧率自适应）
    # ════════════════════════════════════════════════════════

    def _tick(self):
        t0 = time.perf_counter()

        playing = self.audio.playing
        pos     = self.audio.position()
        dur     = self.audio.duration()
        frac    = pos / max(dur, 1)

        # ── 暂停松弛动画 ──
        if playing:
            self._slack = max(0.0, self._slack - 0.05)  # 快速绷紧
        else:
            self._slack = min(1.0, self._slack + 0.025)  # 缓慢松弛

        # ── 过渡动画（歌曲切换快进效果）──
        in_transition = self._transition > 0
        if in_transition:
            self._transition -= 1
            self._reel_speed = 8.0   # 高速旋转
            t_frac = 1.0 - self._transition / self._TRANS_FRAMES
            self._title_slide_x = (1.0 - t_frac) * self.width() * 0.25
        else:
            self._reel_speed = max(1.0, self._reel_speed * 0.85)
            self._title_slide_x = max(0.0, self._title_slide_x - 4)

        # ── 磁带轮旋转 ──
        if playing or in_transition:
            base   = 3.0 * self._reel_speed
            l_spd  = base * (1.0 - frac * 0.55)
            r_spd  = base * (1.0 + frac * 0.55)

            # 节拍抖动
            bass = 0.0
            if self.audio.decoder_ready:
                bass = self.audio.get_bass_energy(pos)
            if bass > self._prev_bass * 1.4 and bass > 0.15 and self._beat_cooldown == 0:
                self._beat_boost    = min(bass * 12, 8.0)
                self._beat_cooldown = 8
            if self._beat_cooldown > 0: self._beat_cooldown -= 1
            self._prev_bass = bass

            # ── 磁带音符粒子 ──
            note_chars = ['♪', '♫', '♩', '♬']
            for note in self._tape_notes:
                note['y'] += note['vy']
                note['life'] -= 0.015
            self._tape_notes = [n for n in self._tape_notes if n['life'] > 0]
            # 节拍强拍时生成音符
            if self._beat_boost > 1.5 and len(self._tape_notes) < 12:
                for _ in range(random.randint(1, 3)):
                    self._tape_notes.append({
                        'x': random.random(),
                        'y': 0.0,
                        'vy': -random.uniform(0.6, 2.0),
                        'life': random.uniform(0.7, 1.0),
                        'ch': random.choice(note_chars),
                        'sz': random.uniform(0.7, 1.3),
                    })
            # 播放中偶尔自然生成
            if playing and self._bar_frame % 10 == 0 and len(self._tape_notes) < 6:
                self._tape_notes.append({
                    'x': random.random(),
                    'y': 0.0,
                    'vy': -random.uniform(0.3, 1.2),
                    'life': random.uniform(0.5, 0.8),
                    'ch': random.choice(note_chars),
                    'sz': random.uniform(0.5, 1.0),
                })

            boost = self._beat_boost
            self._beat_boost *= 0.6

            self._left_angle  = (self._left_angle  + l_spd + boost) % 360
            self._right_angle = (self._right_angle + r_spd + boost) % 360
        else:
            # 暂停：轮速逐渐衰减
            self._reel_speed = max(0.0, self._reel_speed * 0.88)

        # ── 频谱 ──
        if playing:
            self._bar_frame += 1
            if self.audio.decoder_ready:
                spec = self.audio.get_spectrum(pos, self._bar_count)
                if spec:
                    for i in range(self._bar_count): self._bar_targets[i] = spec[i]
                else:
                    self._random_bars()
            else:
                if self._bar_frame % 4 == 0: self._random_bars()
        else:
            for i in range(self._bar_count): self._bar_targets[i] = 0.05

        for i in range(self._bar_count):
            self._bars[i] += (self._bar_targets[i] - self._bars[i]) * 0.18

        self._hue_offset = (self._hue_offset + 0.003) % 1.0

        # ── LED 呼吸相位 ──
        self._led_phase = (self._led_phase + 0.12) % (2 * math.pi)

        # ── 歌词更新 ──
        if self._lrc_lines:
            self._lrc_idx = LrcParser.current_line(self._lrc_lines, pos)

        # ── 侧边栏动画 ──
        target_frac = 1.0 if self._sidebar_open else 0.0
        self._sidebar_frac += (target_frac - self._sidebar_frac) * 0.18

        self.update()

        # ── 帧率自适应 ──
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if elapsed_ms > self._tick_interval * 0.8:
            self._tick_slow_count += 1
            if self._tick_slow_count >= 5:
                new_interval = min(50, self._tick_interval + 5)
                if new_interval != self._tick_interval:
                    self._tick_interval = new_interval
                    self._anim_timer.setInterval(new_interval)
                self._tick_slow_count = 0
        else:
            self._tick_slow_count = 0
            if self._tick_interval > 30:
                self._tick_interval = max(30, self._tick_interval - 1)
                self._anim_timer.setInterval(self._tick_interval)

    def _random_bars(self):
        n = self._bar_count
        for i in range(0, n, 3):
            self._bar_targets[i]              = random.uniform(0.2,  1.0)
            self._bar_targets[min(i+1, n-1)]  = random.uniform(0.12, 0.65)
            self._bar_targets[min(i+2, n-1)]  = random.uniform(0.05, 0.35)

    # ════════════════════════════════════════════════════════
    #  paintEvent
    # ════════════════════════════════════════════════════════

    def paintEvent(self, event):
        w, h = self.width(), self.height()
        s    = w / 680
        mg   = int(18 * s)
        cb   = h - mg
        bw   = w - mg * 2
        bh   = h - mg * 2

        img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(0)

        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # 机身 path
        body = QPainterPath()
        body.addRoundedRect(QRectF(mg, mg, bw, bh), 22, 22)

        # 机身主体
        p.fillPath(body, QColor(55, 60, 72, 115))
        p.setPen(QPen(QColor(170,180,200,150), 2)); p.drawPath(body)

        # 内发光
        p2 = QPainterPath()
        p2.addRoundedRect(QRectF(mg+3, mg+3, bw-6, bh-6), 20, 20)
        p.setPen(QPen(QColor(255,255,255,30), 1)); p.drawPath(p2)

        # 所有后续内容裁剪在机身圆角内（img 上 clip 安全）
        p.save()
        p.setClipPath(body)

        # 四边斜面
        self._draw_bevels(p, mg, cb, bw, bh, s)

        # 磨砂纹理
        self._draw_texture(p, mg, cb, bw, bh, s)

        # 标签区
        self._draw_label(p, w, h, s, mg, cb)

        # 歌词
        self._draw_lyrics(p, w, h, s, mg)

        # 磁带轮区
        reel_r  = int(44 * s)
        reel_y  = self._reel_center_y()
        reel_sp = int(170 * s)
        r1x     = w // 2 - reel_sp
        r2x     = w // 2 + reel_sp
        pos_frac = self.audio.position() / max(self.audio.duration(), 1)

        # 变径：左轮缩小，右轮增大
        l_r = int(reel_r * (1.0 - pos_frac * 0.42))
        r_r = int(reel_r * (0.58 + pos_frac * 0.42))

        self._draw_tape_window(p, w, reel_y, reel_r, reel_sp, s, pos_frac)
        self._draw_reel(p, r1x, reel_y, l_r, self._left_angle)
        self._draw_reel(p, r2x, reel_y, r_r, self._right_angle)
        # 磁头 LED
        self._draw_led(p, s)

        # 控制按钮
        self._draw_buttons(p, w, reel_y, reel_sp, s)

        # 播放模式按钮（在按钮区右侧）
        self._draw_play_mode(p, w, s)

        # 四角螺丝
        self._draw_screws(p, w, mg, cb, s)

        # 音量条
        self._draw_volume_bar(p, w, mg, bh, s)

        # 进度条
        wsx = r1x - reel_r; wex = r2x + reel_r
        wtw = wex - wsx; wh_ = int(78*s); wbo = int(6*s)
        base_y = cb - wbo
        self._draw_progress(p, wsx, wex, wtw, base_y, wh_, s, cb)

        # 频谱
        self._draw_spectrum(p, wsx, wex, base_y, wh_, s)

        # 侧边栏（最上层，仍在 clip 区内）
        if self._sidebar_frac > 0.01:
            self._draw_sidebar(p, w, h, s)

        p.restore()   # 结束机身 clip
        p.end()       # 结束 img QPainter

        # ── 贴到真实透明窗口 ──
        # CompositionMode_Source：直接覆写像素（含 alpha），
        # 不与旧内容混合，彻底替换每一帧，无残影无黑块
        real = QPainter(self)
        real.setCompositionMode(
            QPainter.CompositionMode.CompositionMode_Source)
        real.drawImage(0, 0, img)
        real.end()

    # ──────────────────────────────────────────────────────────
    #  机身四边斜面高光/阴影
    # ──────────────────────────────────────────────────────────

    def _draw_bevels(self, p, mg, cb, bw, bh, s):
        e = int(5 * s)
        tg = QLinearGradient(0, mg, 0, mg+e)
        tg.setColorAt(0, QColor(255,255,255,55)); tg.setColorAt(1, QColor(255,255,255,0))
        tp = QPainterPath(); tp.addRoundedRect(QRectF(mg, mg, bw, e), 20, 20)
        p.fillPath(tp, tg)
        bg = QLinearGradient(0, cb-e, 0, cb)
        bg.setColorAt(0, QColor(0,0,0,0)); bg.setColorAt(1, QColor(0,0,0,45))
        bp = QPainterPath(); bp.addRoundedRect(QRectF(mg, cb-e, bw, e), 0, 0)
        p.fillPath(bp, bg)
        lg = QLinearGradient(mg, 0, mg+e, 0)
        lg.setColorAt(0, QColor(255,255,255,40)); lg.setColorAt(1, QColor(255,255,255,0))
        p.fillRect(QRectF(mg, mg+e, e, bh-e*2), lg)
        rg = QLinearGradient(mg+bw-e, 0, mg+bw, 0)
        rg.setColorAt(0, QColor(0,0,0,0)); rg.setColorAt(1, QColor(0,0,0,35))
        p.fillRect(QRectF(mg+bw-e, mg+e, e, bh-e*2), rg)

    # ──────────────────────────────────────────────────────────
    #  磨砂纹理（细点噪声）
    # ──────────────────────────────────────────────────────────

    def _draw_texture(self, p, mg, cb, bw, bh, s):
        """在机身上叠加稀疏点噪声，模拟磨砂塑料质感"""
        p.setPen(Qt.PenStyle.NoPen)
        # 用固定种子确保噪声稳定（不随帧变化闪烁）
        rng = random.Random(42)
        step = max(4, int(5 / s))
        dot_alpha = 12
        for dx in range(0, int(bw), step):
            for dy in range(0, int(bh), step):
                if rng.random() > 0.35: continue
                x = mg + dx + rng.randint(0, step-1)
                y = mg + dy + rng.randint(0, step-1)
                # 确保在圆角机身内（粗略过滤角落）
                if x < mg+22 and y < mg+22: continue
                if x > mg+bw-22 and y < mg+22: continue
                brightness = rng.randint(180, 255)
                p.setBrush(QColor(brightness, brightness, brightness, dot_alpha))
                p.drawEllipse(QPointF(x, y), 0.8, 0.8)

    # ──────────────────────────────────────────────────────────
    #  标签区
    # ──────────────────────────────────────────────────────────

    def _draw_label(self, p, w, h, s, mg, cb):
        label_y = mg + int(10*s); label_h = int(64*s)
        slant   = int(10*s)
        tl_x = mg + int(26*s);  tr_x = w - mg - int(26*s)
        bl_x = tl_x + slant;    br_x = tr_x - slant
        top_y = label_y; bot_y = label_y + label_h; cr = int(8*s)

        def trap(tlx, trx, blx, brx, ty, by, r):
            pp = QPainterPath()
            pp.moveTo(tlx+r, ty); pp.lineTo(trx-r, ty)
            pp.arcTo(trx-2*r, ty, 2*r, 2*r, 90, -90)
            pp.lineTo(brx, by-r)
            pp.arcTo(brx-2*r, by-2*r, 2*r, 2*r, 0, -90)
            pp.lineTo(blx+r, by)
            pp.arcTo(blx, by-2*r, 2*r, 2*r, 270, -90)
            pp.lineTo(tlx, ty+r)
            pp.arcTo(tlx, ty, 2*r, 2*r, 180, -90)
            pp.closeSubpath(); return pp

        p.fillPath(trap(tl_x+int(2*s), tr_x-int(2*s), bl_x, br_x,
                        top_y+int(3*s), bot_y+int(4*s), cr), QColor(0,0,0,40))
        lp = trap(tl_x, tr_x, bl_x, br_x, top_y, bot_y, cr)
        p.fillPath(lp, QColor(72, 64, 50, 160))
        p.setPen(QPen(QColor(180,170,140,90),1)); p.drawPath(lp)
        p.fillPath(trap(tl_x+int(2*s), tr_x-int(2*s), bl_x+int(4*s), br_x-int(4*s),
                        top_y+int(1*s), top_y+int(8*s), int(5*s)), QColor(255,255,255,35))
        gg = QLinearGradient(0, top_y, 0, bot_y)
        gg.setColorAt(0, QColor(255,255,255,100)); gg.setColorAt(0.3, QColor(255,255,255,30))
        gg.setColorAt(1, QColor(255,255,255,0))
        p.fillPath(trap(tl_x, tr_x, bl_x, br_x, top_y, bot_y, cr), gg)

        # 装饰横线
        p.setPen(QPen(QColor(200,190,160,50), max(1,int(s))))
        for i in range(2):
            ly = top_y + int(22*s) + i*int(20*s)
            p.drawLine(int(bl_x+int(16*s)), ly, int(br_x-int(16*s)), ly)

        # 固定孔
        hole_r = int(4*s)
        for hx in (tl_x+int(14*s), tr_x-int(14*s)):
            hy = top_y + label_h//2
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(30,25,18,180)); p.drawEllipse(QPointF(hx,hy), hole_r, hole_r)
            p.setBrush(QColor(120,110,90,60)); p.drawEllipse(QPointF(hx-1,hy-1), hole_r-1, hole_r-1)

        # TYPE II / C-60
        p.setFont(QFont("Consolas", max(7, int(8*s))))
        p.setPen(QColor(200,190,160,60))
        p.drawText(QRectF(bl_x+int(20*s), bot_y-int(16*s), int(60*s), int(13*s)),
                   Qt.AlignmentFlag.AlignLeft, "TYPE Ⅱ  C-60")

        # 封面
        cs = int(label_h * 0.80)
        cx0 = tl_x + int(18*s); cy0 = top_y + (label_h - cs)//2
        if self._cover_pixmap and not self._cover_pixmap.isNull():
            sc = self._cover_pixmap.scaled(cs, cs,
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation)
            clip = QPainterPath(); clip.addRoundedRect(QRectF(cx0,cy0,cs,cs), 5, 5)
            p.save(); p.setClipPath(clip)
            p.drawPixmap(cx0, cy0, cs, cs, sc); p.restore()
            p.setPen(QPen(QColor(255,255,255,40),1)); p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(QRectF(cx0,cy0,cs,cs), 5, 5)
        else:
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(100,90,70,60))
            p.drawRoundedRect(QRectF(cx0,cy0,cs,cs), 5, 5)
            p.setPen(QColor(150,140,120,80))
            p.setFont(QFont("Segoe UI Emoji", max(10,int(16*s))))
            p.drawText(QRectF(cx0,cy0,cs,cs), Qt.AlignmentFlag.AlignCenter, "♫")

        # 歌名 / 艺术家（过渡时从右侧滑入）
        tx0 = cx0 + cs + int(8*s) + self._title_slide_x
        tw  = br_x - int(16*s) - tx0
        tf  = QFont("Microsoft YaHei", max(11, int(15*s))); tf.setBold(True)
        p.setFont(tf); p.setPen(QColor(240,235,220,220))
        p.save(); p.setClipRect(QRectF(cx0+cs+int(4*s), top_y, tr_x-cx0-cs-int(4*s), label_h))
        p.drawText(QRectF(tx0, top_y+int(6*s), tw, int(22*s)),
                   Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter, self._track_title)
        af = QFont("Microsoft YaHei", max(9, int(12*s)))
        p.setFont(af); p.setPen(QColor(200,190,170,180))
        p.drawText(QRectF(tx0, top_y+int(30*s), tw, int(20*s)),
                   Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter, self._track_artist)
        p.restore()

    # ──────────────────────────────────────────────────────────
    #  歌词显示
    # ──────────────────────────────────────────────────────────

    def _draw_lyrics(self, p, w, h, s, mg):
        if not self._lrc_lines or self._lrc_idx < 0:
            return
        reel_y  = self._reel_center_y()
        lyr_y   = mg + int(78*s) + int(4*s)   # 标签区下方
        lyr_h   = reel_y - lyr_y - int(8*s)
        if lyr_h < int(14*s): return

        lines_to_show = 3  # 上一行、当前行、下一行
        line_h = lyr_h / lines_to_show
        center_offset = 1  # 中间那行是当前行

        p.save()
        p.setClipRect(QRectF(mg + int(30*s), lyr_y, w - mg*2 - int(60*s), lyr_h))
        for offset in range(-1, 2):
            idx = self._lrc_idx + offset
            if idx < 0 or idx >= len(self._lrc_lines): continue
            _, text = self._lrc_lines[idx]
            ly = lyr_y + (offset + center_offset) * line_h + line_h/2
            if offset == 0:
                # 当前行：高亮
                font_sz = max(9, int(13*s)); fnt = QFont("Microsoft YaHei", font_sz)
                fnt.setBold(True); p.setFont(fnt)
                p.setPen(QColor(240, 220, 160, 220))
            else:
                font_sz = max(8, int(11*s)); p.setFont(QFont("Microsoft YaHei", font_sz))
                alpha   = 80 if abs(offset) == 2 else 120
                p.setPen(QColor(200, 190, 160, alpha))
            p.drawText(QRectF(mg+int(30*s), ly - line_h/2, w-mg*2-int(60*s), line_h),
                       Qt.AlignmentFlag.AlignCenter|Qt.AlignmentFlag.AlignVCenter, text)
        p.restore()

    # ──────────────────────────────────────────────────────────
    #  磁带观察窗（贝塞尔带体弧线 + 变径 + 松弛）
    # ──────────────────────────────────────────────────────────

    def _draw_tape_window(self, p, w, reel_y, reel_r, reel_sp, s, pos_frac):
        r1x = w // 2 - reel_sp; r2x = w // 2 + reel_sp
        win_l = r1x + int(reel_r * 0.6); win_r = r2x - int(reel_r * 0.6)
        win_t = reel_y - int(reel_r * 0.55); win_b = reel_y + int(reel_r * 0.55)
        win_w = win_r - win_l; win_h = win_b - win_t

        wp = QPainterPath()
        wp.addRoundedRect(QRectF(win_l, win_t, win_w, win_h), int(8*s), int(8*s))
        p.setPen(Qt.PenStyle.NoPen); p.fillPath(wp, QColor(15,12,8,200))

        p.setPen(QPen(QColor(255,255,255,25),1))
        p.drawLine(int(win_l+8*s), int(win_t+2), int(win_r-8*s), int(win_t+2))
        p.setPen(QPen(QColor(0,0,0,60),1))
        p.drawLine(int(win_l+8*s), int(win_b-2), int(win_r-8*s), int(win_b-2))

        # 带体：贝塞尔弧线（张力 → 松弛由 _slack 控制）
        tape_colors = [
            QColor(90, 55, 25, 220), QColor(110, 70, 35, 180),
            QColor(75, 45, 18, 200), QColor(130, 85, 45, 140),
            QColor(85, 50, 20, 210),
        ]
        n_lines = 5
        gap = win_h / (n_lines + 1)
        pad = int(4*s)

        # 随进度偏移：带子在窗口内向右靠拢（右轮卷入变多）
        x_shift = int(pos_frac * win_w * 0.12)

        for i, tc in enumerate(tape_colors):
            ly      = win_t + gap * (i + 1)
            lw      = max(1.0, 2.0 * s)
            # 松弛时控制点向下弯曲
            sag     = self._slack * int(reel_r * 0.18)
            x0      = win_l + pad + x_shift
            x1      = win_r - pad + x_shift
            ctrl_y  = ly + sag   # 弧线顶点下移

            path = QPainterPath()
            path.moveTo(x0, ly)
            path.quadTo((x0+x1)/2, ctrl_y, x1, ly)

            p.setPen(QPen(tc, lw)); p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
            # 高光
            p.setPen(QPen(QColor(200,160,100,35), max(1,int(s*0.8))))
            hl = QPainterPath()
            hl.moveTo(x0, ly - lw/2)
            hl.quadTo((x0+x1)/2, ctrl_y - lw/2, x1, ly - lw/2)
            p.drawPath(hl)

        # 跳动音符
        if self._tape_notes:
            nf = QFont("Segoe UI Symbol", max(8, int(11*s)))
            for note in self._tape_notes:
                nx = win_l + note['x'] * win_w
                ny = win_t - int(6*s) + note['y'] * int(10*s)
                a  = int(note['life'] * 210)
                p.setFont(nf)
                c = QColor(255, 220, 130, a) if note['life'] > 0.3 else QColor(255, 200, 100, a)
                p.setPen(c)
                p.drawText(QRectF(nx - int(8*s), ny - int(8*s),
                                  int(16*s), int(16*s)),
                           Qt.AlignmentFlag.AlignCenter, note['ch'])

        p.setPen(QPen(QColor(140,130,110,100), max(1,int(1.5*s))))
        p.setBrush(Qt.BrushStyle.NoBrush); p.drawPath(wp)

    # ──────────────────────────────────────────────────────────
    #  磁带轮
    # ──────────────────────────────────────────────────────────

    def _draw_reel(self, p, cx, cy, r, angle_deg):
        p.save(); p.translate(cx, cy)
        p.setPen(QPen(QColor(140,150,170,60),1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(0,0), r, r)
        p.setPen(QPen(QColor(120,130,150,140),2))
        p.setBrush(QColor(25,28,35,145))
        p.drawEllipse(QPointF(0,0), r-2, r-2)
        morandi = [QColor(185,150,145),QColor(155,170,150),QColor(145,155,175),
                   QColor(190,175,150),QColor(170,160,180)]
        ar = math.radians(angle_deg)
        for i in range(5):
            a = math.radians(i*72)+ar
            gx = math.cos(a)*(r-min(16,r-8)); gy = math.sin(a)*(r-min(16,r-8))
            c = morandi[i]
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(c.red(),c.green(),c.blue(),80))
            p.drawEllipse(QPointF(gx+1,gy+1), 5, 5)
            p.setBrush(QColor(c.red(),c.green(),c.blue(),180))
            p.drawEllipse(QPointF(gx,gy), 5, 5)
            p.setBrush(QColor(min(c.red()+40,255),min(c.green()+40,255),min(c.blue()+40,255),120))
            p.drawEllipse(QPointF(gx-1,gy-2), 3, 3)
        p.setPen(QPen(QColor(90,100,120,100),1)); p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(0,0), max(4,r-24), max(4,r-24))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(75,80,95,160)); p.drawEllipse(QPointF(0,0), 8, 8)
        p.setBrush(QColor(140,145,160,200)); p.drawEllipse(QPointF(0,0), 3, 3)
        p.restore()

    # ──────────────────────────────────────────────────────────
    #  磁头 LED 指示灯
    # ──────────────────────────────────────────────────────────

    def _draw_led(self, p, s):
        """左下角小长方形呼吸灯，位于可视化切换螺丝上方"""
        if not self.audio.playing and self.audio.current_index < 0:
            return   # 停止状态不显示
        mg  = int(18 * s)
        sx  = mg + int(28 * s)                     # 螺丝中心 x
        sy  = self.height() - mg - int(24 * s)      # 螺丝中心 y
        lw_ = max(5, int(6*s))                      # 长方形宽度
        lh_ = max(2, int(3*s))                      # 长方形高度
        lx  = sx - lw_ // 2                         # 居中对齐螺丝
        ly  = sy - int(18 * s)                      # 螺丝上方

        if self.audio.playing:
            breath = 0.35 + 0.65 * math.sin(self._led_phase)
            base   = QColor(80, 220, 100)
            alpha  = int(100 + breath * 155)
            label  = "PLAY"
        else:
            base  = QColor(255, 160, 50)
            alpha = 140
            label = "PAUSE"

        # 呼吸光晕
        glow_a = alpha // 5
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(base.red(), base.green(), base.blue(), glow_a))
        p.drawRoundedRect(QRectF(lx - lw_*0.3, ly - lh_*0.3,
                                  lw_ * 1.6, lh_ * 1.6), max(1, int(2*s)), max(1, int(2*s)))
        # 主体
        p.setBrush(QColor(base.red(), base.green(), base.blue(), alpha))
        p.drawRoundedRect(QRectF(lx, ly, lw_, lh_), max(1, int(1*s)), max(1, int(1*s)))
        # 高光
        p.setBrush(QColor(255, 255, 255, alpha // 3))
        p.drawRoundedRect(QRectF(lx + int(1*s), ly + int(1*s), lw_ // 2, lh_ // 2),
                          max(1, int(1*s)), max(1, int(1*s)))
        # 状态文字
        p.setFont(QFont("Consolas", max(6, int(7*s))))
        p.setPen(QColor(base.red(), base.green(), base.blue(), min(255, alpha + 30)))
        p.drawText(QRectF(lx + lw_ + int(4*s), ly - int(1*s),
                          int(32*s), lh_ + int(2*s)),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)

    # ──────────────────────────────────────────────────────────
    #  控制按钮
    # ──────────────────────────────────────────────────────────

    def _draw_buttons(self, p, w, reel_y, reel_sp, s):
        btn_s  = int(44*s); btn_sp = int(reel_sp*0.50)
        cx     = w // 2; btn_y = reel_y - btn_s//2
        p.setFont(QFont("Segoe UI Symbol", max(10, int(20*s))))
        btns = [
            (cx-btn_sp-btn_s//2, btn_y, btn_s, "⏮"),
            (cx-btn_s//2,        btn_y, btn_s, self._btn_play_text),
            (cx+btn_sp-btn_s//2, btn_y, btn_s, "⏭"),
        ]
        self._btn_regions = []
        for i, (bx,by_,bs,sym) in enumerate(btns):
            r = QRectF(bx, by_, bs, bs)
            self._btn_regions.append(r)
            p.setPen(QColor(255,255,255,255) if self._btn_hover==i else QColor(180,180,180,180))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, sym)

    # ──────────────────────────────────────────────────────────
    #  播放模式图标
    # ──────────────────────────────────────────────────────────

    def _draw_play_mode(self, p, w, s):
        """右下角循环模式图标，与左下角 LED 对称"""
        mode_icons = ["→", "↺", "⇄"]
        icon = mode_icons[self.audio.play_mode]
        mg  = int(18 * s)
        # 对齐右下方螺丝（静音切换），放在它的正上方，和 LED 对称
        sx  = w - mg - int(28 * s)
        sy  = self.height() - mg - int(24 * s)
        sz  = int(14 * s)
        cx  = sx
        cy  = sy - int(18 * s)
        self._mode_btn_rect = QRectF(cx - sz//2, cy - sz//2, sz, sz)
        hovered = (hasattr(self,'_mode_hover') and self._mode_hover)
        p.setFont(QFont("Segoe UI Symbol", max(7, int(11*s))))
        p.setPen(QColor(255,255,255,220) if hovered else QColor(150,160,150,160))
        p.drawText(self._mode_btn_rect, Qt.AlignmentFlag.AlignCenter, icon)

    # ──────────────────────────────────────────────────────────
    #  四角螺丝
    # ──────────────────────────────────────────────────────────

    def _draw_screws(self, p, w, mg, cb, s):
        sr = int(7*s); top_off = int(18*s); bot_off = int(26*s)
        self._screw_positions = [
            (mg+top_off,              mg+top_off+int(7*s)),
            (w-mg-top_off,            mg+top_off+int(7*s)),
            (mg+bot_off+int(2*s),     cb-bot_off+int(2*s)),
            (w-mg-bot_off-int(2*s),   cb-bot_off+int(2*s)),
        ]
        for idx, (sx, sy) in enumerate(self._screw_positions):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(165,170,180,170)); p.drawEllipse(QPointF(sx,sy), sr, sr)
            p.setBrush(QColor(135,140,150,190)); p.drawEllipse(QPointF(sx,sy), sr-int(3*s), sr-int(3*s))
            lw = int(2*s)
            if idx == 0:
                p.setPen(QPen(QColor(220,225,235,200), max(1,lw)))
                d = int(3*s)
                p.drawLine(int(sx-d),int(sy),int(sx+d),int(sy))
                p.drawLine(int(sx),int(sy-d),int(sx),int(sy+d))
            elif idx == 1:
                p.setPen(QPen(QColor(220,225,235,200), max(1,lw)))
                d = int(2*s)
                p.drawLine(int(sx-d),int(sy-d),int(sx+d),int(sy+d))
                p.drawLine(int(sx+d),int(sy-d),int(sx-d),int(sy+d))
            elif idx == 2:
                p.setPen(QPen(QColor(255,90,100,200), max(1,lw)))
                d = int(3*s)
                p.drawLine(int(sx-d),int(sy-d),int(sx),int(sy+d))
                p.drawLine(int(sx+d),int(sy-d),int(sx),int(sy+d))
            else:
                muted = self.audio.muted
                color = QColor(255,80,80,200) if muted else QColor(160,220,160,200)
                p.setPen(QPen(color, max(1,lw)))
                d = int(2*s)
                if muted:
                    p.drawLine(int(sx-d),int(sy-d),int(sx+d),int(sy+d))
                    p.drawLine(int(sx+d),int(sy-d),int(sx-d),int(sy+d))
                else:
                    p.drawLine(int(sx-d),int(sy-d),int(sx+d),int(sy))
                    p.drawLine(int(sx+d),int(sy),int(sx-d),int(sy+d))
                    p.drawLine(int(sx-d),int(sy-d),int(sx-d),int(sy+d))

    # ──────────────────────────────────────────────────────────
    #  音量条
    # ──────────────────────────────────────────────────────────

    def _draw_volume_bar(self, p, w, mg, bh, s):
        bw_ = int(6*s); bh_ = int(bh*0.45)
        bx  = w - mg - int(14*s); by_ = mg + (bh-bh_)//2
        vol = self.audio.volume
        self._vol_bar_rect = QRectF(bx-int(4*s), by_-int(4*s), bw_+int(8*s), bh_+int(8*s))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255,255,255,20))
        p.drawRoundedRect(QRectF(bx, by_, bw_, bh_), 3, 3)
        fh = int(bh_*vol)
        if fh > 0:
            fg = QLinearGradient(0, by_+bh_-fh, 0, by_+bh_)
            fg.setColorAt(0, QColor(180,220,180,200)); fg.setColorAt(1, QColor(100,180,100,160))
            p.setBrush(fg)
            p.drawRoundedRect(QRectF(bx, by_+bh_-fh, bw_, fh), 3, 3)
        ky = by_ + bh_ - fh
        p.setBrush(QColor(220,230,220,230))
        p.drawEllipse(QPointF(bx+bw_/2, ky), int(5*s), int(5*s))
        p.setFont(QFont("Consolas", max(7,int(8*s))))
        p.setPen(QColor(180,200,180,120))
        p.drawText(QRectF(bx-int(6*s), by_+bh_+int(3*s), bw_+int(12*s), int(14*s)),
                   Qt.AlignmentFlag.AlignCenter, f"{int(vol*100)}")

    # ──────────────────────────────────────────────────────────
    #  进度条（含拖拽气泡）
    # ──────────────────────────────────────────────────────────

    def _draw_progress(self, p, wsx, wex, wtw, base_y, wh_, s, cb):
        prog_y = base_y - wh_ - int(24*s)
        dur    = self.audio.duration(); pos_ms = self.audio.position()

        def _fmt(ms):
            if ms <= 0: return "00:00"
            sec = ms//1000; return f"{sec//60:02d}:{sec%60:02d}"

        p.setFont(QFont("Consolas", max(9,int(12*s))))
        p.setPen(QColor(200,200,200,160))
        p.drawText(QRectF(wsx-int(70*s), prog_y-int(12*s), int(65*s), int(20*s)),
                   Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter, _fmt(pos_ms))
        p.drawText(QRectF(wex+int(5*s), prog_y-int(12*s), int(65*s), int(20*s)),
                   Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter,
                   "-"+_fmt(max(0, dur-pos_ms)))

        ph = int(4*s)
        self._progress_rect = QRectF(wsx, prog_y, wtw, ph)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255,255,255,30))
        p.drawRoundedRect(self._progress_rect, 2, 2)

        frac = 0.0
        if dur > 0:
            frac = min(pos_ms/dur, 1.0)
            fw   = int(self._progress_rect.width()*frac)
            if fw > 0:
                p.setBrush(QColor(220,200,150,180))
                p.drawRoundedRect(QRectF(self._progress_rect.x(),
                                         self._progress_rect.y(), fw, ph), 2, 2)
            dot_x = self._progress_rect.x() + self._progress_rect.width()*frac
            dot_y = self._progress_rect.center().y()
            # 用纯矢量菱形+圆代替 Emoji，彻底消除字形 bbox 残留
            dr = max(5, int(6*s))
            p.setPen(Qt.PenStyle.NoPen)
            # 光晕
            p.setBrush(QColor(255, 100, 130, 60))
            p.drawEllipse(QPointF(dot_x, dot_y), dr+3, dr+3)
            # 菱形主体
            diamond = QPainterPath()
            diamond.moveTo(dot_x,        dot_y - dr)
            diamond.lineTo(dot_x + dr,   dot_y)
            diamond.lineTo(dot_x,        dot_y + dr)
            diamond.lineTo(dot_x - dr,   dot_y)
            diamond.closeSubpath()
            p.setBrush(QColor(255, 100, 130, 230))
            p.drawPath(diamond)
            # 高光
            p.setBrush(QColor(255, 200, 210, 160))
            p.drawEllipse(QPointF(dot_x - dr*0.25, dot_y - dr*0.35), dr*0.3, dr*0.3)

        # 拖拽气泡
        if self._seek_bubble_x is not None and self._seek_bubble_t is not None:
            bx  = self._seek_bubble_x
            by_ = prog_y - int(28*s)
            bw_ = int(48*s); bht = int(18*s)
            bp  = QPainterPath()
            bp.addRoundedRect(QRectF(bx-bw_//2, by_, bw_, bht), 4, 4)
            # 小三角
            bp.moveTo(bx-int(5*s), by_+bht)
            bp.lineTo(bx+int(5*s), by_+bht)
            bp.lineTo(bx, by_+bht+int(5*s))
            bp.closeSubpath()
            p.setPen(Qt.PenStyle.NoPen)
            p.fillPath(bp, QColor(40,40,50,200))
            p.setFont(QFont("Consolas", max(8,int(10*s))))
            p.setPen(QColor(220,210,180,240))
            p.drawText(QRectF(bx-bw_//2, by_, bw_, bht),
                       Qt.AlignmentFlag.AlignCenter, _fmt(self._seek_bubble_t))

    # ════════════════════════════════════════════════════════
    #  频谱可视化
    # ════════════════════════════════════════════════════════

    def _bar_color(self, i, t, n):
        if self._viz_style < 6:
            pal = [(185,150,145),(200,180,165),(155,170,150),(145,155,175),
                   (170,165,180),(180,160,155),(160,170,170),(190,175,160)]
            idx = int((i/n * len(pal) + self._hue_offset * len(pal)) % len(pal))
            r,g,b = pal[idx]; sc = 0.6 + t*0.4
            return QColor(min(255,int(r*sc)), min(255,int(g*sc)), min(255,int(b*sc)))
        else:
            hue = (i/n + self._hue_offset) % 1.0
            sat = 0.75; val = 0.75 + t*0.25; chroma = val*sat
            h6  = hue*6; hx = chroma*(1-abs(h6%2-1)); cm = val-chroma
            if   h6<1: rf,gf,bf = chroma,hx,0
            elif h6<2: rf,gf,bf = hx,chroma,0
            elif h6<3: rf,gf,bf = 0,chroma,hx
            elif h6<4: rf,gf,bf = 0,hx,chroma
            elif h6<5: rf,gf,bf = hx,0,chroma
            else:      rf,gf,bf = chroma,0,hx
            return QColor(min(255,int((rf+cm)*255)), min(255,int((gf+cm)*255)),
                          min(255,int((bf+cm)*255)))

    def _draw_spectrum(self, p, sx, ex, base_y, max_h, s):
        self._viz_erase_pending = False
        # 空风格：什么都不画（启动首帧用，让烙印是空的）
        if self._viz_style < 0:
            return
        tw = ex-sx; n = min(60, len(self._bars)); self._bar_count = n
        self._viz_methods[self._viz_style % 7](p, sx, ex, base_y, max_h, s, n, tw)

    def _draw_bars(self, p, sx, ex, base_y, max_h, s, n, tw):
        cw = tw/n; bw = max(2.0, cw*0.7)
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(n):
            t = self._bars[i]; bh = max(2, int(t*max_h))
            bx = sx+i*cw; by = base_y-bh; c = self._bar_color(i,t,n)
            for alpha,pad in [(30,3),(80,1),(240,0)]:
                p.setBrush(QColor(c.red(),c.green(),c.blue(),alpha))
                p.drawRoundedRect(QRectF(bx-pad, by-pad*2, bw+pad*2, bh+pad*3), 2, 2)
            p.setBrush(QColor(min(c.red()+80,255),min(c.green()+80,255),min(c.blue()+80,255),200))
            p.drawRoundedRect(QRectF(bx,by,bw,max(3,int(bh*0.25))), 2, 2)

    def _draw_radar(self, p, sx, ex, base_y, max_h, s, n, tw):
        cx=(sx+ex)/2; cy=base_y-max_h/2-int(10*s)
        mr=min(tw,max_h)/2+int(2*s); ir=mr*0.05; ast=2*math.pi/n
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(n):
            t=self._bars[i]; bl=int(ir+t*(mr-ir)); a=i*ast-math.pi/2
            c=self._bar_color(i,t,n); ex2=cx+math.cos(a)*bl; ey2=cy+math.sin(a)*bl
            bw_=max(1.5,tw/n*0.55)
            p.setBrush(QColor(c.red(),c.green(),c.blue(),50))
            p.drawEllipse(QPointF(ex2,ey2), bw_+4, bw_+4)
            p.setPen(QPen(QColor(c.red(),c.green(),c.blue(),230), bw_))
            p.drawLine(QPointF(cx+math.cos(a)*ir, cy+math.sin(a)*ir), QPointF(ex2,ey2))
            p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(220,220,240,200)); p.drawEllipse(QPointF(cx,cy), 1, 1)

    def _draw_waveform(self, p, sx, ex, base_y, max_h, s, n, tw):
        cy=base_y-max_h/2; hh=max_h/2; sc=max(200,int(tw))
        samples=self.audio._decoder.get_waveform(self.audio.position(), sc)
        if not samples or len(samples)<2: return
        pts=[QPointF(sx+i*tw/(sc-1), cy-v*hh*0.9) for i,v in enumerate(samples[:sc])]
        path=QPainterPath(); path.moveTo(pts[0].x(),cy)
        for pt in pts: path.lineTo(pt)
        path.lineTo(pts[-1].x(),cy); path.closeSubpath()
        mc=self._bar_color(n//2,0.6,1)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(mc.red(),mc.green(),mc.blue(),25)); p.drawPath(path)
        for i in range(len(pts)-1):
            c=self._bar_color(i,0.7,len(pts))
            p.setPen(QPen(QColor(c.red(),c.green(),c.blue(),220), max(1.5,2.5*s)))
            p.drawLine(pts[i],pts[i+1])

    def _draw_mirror(self, p, sx, ex, base_y, max_h, s, n, tw):
        cy=base_y-max_h/2; hh=max_h/2-int(3*s); cw=tw/n; bw=max(2*s,cw*0.65)
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(n):
            t=self._bars[i]; bh=max(2*s,int(t*hh)); bx=sx+i*cw; c=self._bar_color(i,t,n)
            p.setBrush(QColor(c.red(),c.green(),c.blue(),200))
            p.drawRoundedRect(QRectF(bx,cy-bh,bw,bh),1,1)
            p.setBrush(QColor(min(c.red()+60,255),min(c.green()+60,255),min(c.blue()+60,255),140))
            p.drawRoundedRect(QRectF(bx,cy-bh,bw,max(2*s,int(bh*0.3))),1,1)
            p.setBrush(QColor(c.red(),c.green(),c.blue(),120))
            p.drawRoundedRect(QRectF(bx,cy,bw,bh),1,1)
        p.setPen(QPen(QColor(255,255,255,40),1))
        p.drawLine(QPointF(sx,cy),QPointF(ex,cy))

    def _draw_particles(self, p, sx, ex, base_y, max_h, s, n, tw):
        cy=base_y-max_h/2; hh=max_h/2-int(6*s)
        if not self._particles or len(self._particles)!=40 or getattr(self,'_ptw',0)!=tw:
            self._particles=[{'frac':random.random(),'y':cy,'ty':cy,
                               'sz':random.uniform(2,5)*s,'hue':random.random()} for _ in range(40)]
            self._ptw=tw
        p.setPen(Qt.PenStyle.NoPen)
        for pi,pt in enumerate(self._particles):
            px=sx+pt['frac']*tw; bi=max(0,min(n-1,int(pt['frac']*n))); t=self._bars[bi]
            pt['ty']=cy-t*hh*(1.0 if pi%2 else -1.0)
            pt['y']+=(pt['ty']-pt['y'])*0.12; pt['sz']=max(2*s,(3+t*4)*s)
            c=self._bar_color(bi,t,n); sz=pt['sz']
            p.setBrush(QColor(c.red(),c.green(),c.blue(),40))
            p.drawEllipse(QPointF(px,pt['y']),sz+2*s,sz+2*s)
            p.setBrush(QColor(c.red(),c.green(),c.blue(),220))
            p.drawEllipse(QPointF(px,pt['y']),sz,sz)

    def _draw_pulse(self, p, sx, ex, base_y, max_h, s, n, tw):
        cx=(sx+ex)/2; cy=base_y-max_h/2-int(10*s); mr=min(tw,max_h)/2+int(2*s)
        bass=sum(self._bars[:10])/10; mid=sum(self._bars[10:30])/20
        high=sum(self._bars[30:])/max(1,n-30)
        rings=[(0.25*mr,bass,self._bar_color(0,0.8,n)),
               (0.55*mr,mid,self._bar_color(n//3,0.7,n)),
               (0.85*mr,high,self._bar_color(n*2//3,0.6,n))]
        p.setPen(Qt.PenStyle.NoPen)
        for br,energy,color in rings:
            r=max(6,int(br+energy*mr*0.35))
            p.setBrush(QColor(color.red(),color.green(),color.blue(),15))
            p.drawEllipse(QPointF(cx,cy),r+16,r+16)
            p.setBrush(QColor(color.red(),color.green(),color.blue(),40))
            p.drawEllipse(QPointF(cx,cy),r+8,r+8)
            p.setPen(QPen(QColor(color.red(),color.green(),color.blue(),max(80,color.alpha())),max(2.0,3.0*s)))
            p.setBrush(Qt.BrushStyle.NoBrush); p.drawEllipse(QPointF(cx,cy),r,r)
            p.setPen(Qt.PenStyle.NoPen)
        dc=self._bar_color(0,bass,1)
        p.setBrush(QColor(dc.red(),dc.green(),dc.blue(),int(120+bass*135)))
        p.drawEllipse(QPointF(cx,cy),max(1,int(2+bass*4)),max(1,int(2+bass*4)))

    def _draw_tape_ripple(self, p, sx, ex, base_y, max_h, s, n, tw):
        cy=base_y-max_h*0.5; amp=max_h*0.38; num_lines=5; gap=max_h*0.16
        pts_n=max(120,int(tw))
        for li in range(num_lines):
            lcy=cy+(li-num_lines//2)*gap; br_t=0.3+li*0.15
            c=self._bar_color(int(li/num_lines*n), br_t, n)
            path=QPainterPath()
            for xi in range(pts_n):
                xf=xi/(pts_n-1); x=sx+xf*tw; bi=max(0,min(n-1,int(xf*n)))
                energy=self._bars[bi]
                phase=self._hue_offset*2*math.pi*3+li*0.8
                y=lcy-math.sin(xf*math.pi*6+phase)*amp*energy*0.55
                if xi==0: path.moveTo(x,y)
                else:     path.lineTo(x,y)
            alpha=160+int(br_t*80); lw=max(1.5,(1.5+br_t)*s)
            p.setPen(QPen(QColor(c.red(),c.green(),c.blue(),alpha),lw))
            p.setBrush(Qt.BrushStyle.NoBrush); p.drawPath(path)
            p.setPen(QPen(QColor(255,220,160,30), max(1,int(s*0.8)))); p.drawPath(path)

    # ──────────────────────────────────────────────────────────
    #  播放列表侧边栏
    # ──────────────────────────────────────────────────────────

    def _draw_sidebar(self, p, w, h, s):
        """从左边缘滑入的半透明播放列表"""
        sb_w   = int(min(260*s, w*0.55))
        frac   = self._sidebar_frac
        x0     = int(-sb_w * (1.0 - frac))   # 左边起始 x（负值到 0）
        mg     = int(18*s)

        # 背景
        p.setPen(Qt.PenStyle.NoPen)
        grad = QLinearGradient(x0, 0, x0+sb_w, 0)
        grad.setColorAt(0,   QColor(20, 22, 30, 220))
        grad.setColorAt(0.85,QColor(30, 33, 42, 200))
        grad.setColorAt(1,   QColor(30, 33, 42,   0))
        p.fillRect(QRectF(x0, mg, sb_w, h-mg*2), grad)

        # 右边缘淡出
        edge_g = QLinearGradient(x0+sb_w-int(20*s), 0, x0+sb_w, 0)
        edge_g.setColorAt(0, QColor(0,0,0,0)); edge_g.setColorAt(1, QColor(0,0,0,0))
        p.fillRect(QRectF(x0+sb_w-int(20*s), mg, int(20*s), h-mg*2), edge_g)

        if not self.audio.playlist: return

        # 标题栏
        p.setFont(QFont("Microsoft YaHei", max(9,int(12*s))))
        p.setPen(QColor(200,190,160,180))
        p.drawText(QRectF(x0+int(10*s), mg+int(6*s), sb_w-int(20*s), int(22*s)),
                   Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter,
                   f"播放列表  ({len(self.audio.playlist)})")

        # 分隔线
        p.setPen(QPen(QColor(200,190,160,40),1))
        p.drawLine(int(x0+int(8*s)), int(mg+int(28*s)),
                   int(x0+sb_w-int(20*s)), int(mg+int(28*s)))

        # 歌曲列表
        item_h    = int(24*s)
        list_top  = mg + int(32*s)
        list_h    = h - mg*2 - int(32*s) - int(8*s)
        visible_n = max(1, list_h // item_h)
        cur       = self.audio.current_index
        # 居中显示当前歌曲
        scroll_start = max(0, cur - visible_n // 2)
        scroll_start = min(scroll_start, max(0, len(self.audio.playlist) - visible_n))

        p.setFont(QFont("Microsoft YaHei", max(8,int(11*s))))
        for vi in range(visible_n):
            idx = scroll_start + vi
            if idx >= len(self.audio.playlist): break
            iy     = list_top + vi * item_h
            is_cur = (idx == cur)
            hov    = (self._sidebar_hover == idx)
            name   = Path(self.audio.playlist[idx]).stem

            # 当前行高亮背景
            if is_cur:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(200,180,120,60))
                p.drawRoundedRect(QRectF(x0+int(6*s), iy, sb_w-int(26*s), item_h), 3, 3)
            elif hov:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(255,255,255,18))
                p.drawRoundedRect(QRectF(x0+int(6*s), iy, sb_w-int(26*s), item_h), 3, 3)

            col = QColor(240,220,160,220) if is_cur else QColor(190,185,175,160)
            p.setPen(col)

            # 序号
            p.setFont(QFont("Consolas", max(7,int(9*s))))
            p.drawText(QRectF(x0+int(8*s), iy, int(20*s), item_h),
                       Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter,
                       str(idx+1))
            # 歌名（截断）
            p.setFont(QFont("Microsoft YaHei", max(8,int(11*s))))
            p.drawText(QRectF(x0+int(30*s), iy, sb_w-int(50*s), item_h),
                       Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter,
                       name)
            # 当前播放指示
            if is_cur:
                p.setFont(QFont("Segoe UI Symbol", max(7,int(9*s))))
                p.drawText(QRectF(x0+int(6*s), iy, int(18*s), item_h),
                           Qt.AlignmentFlag.AlignCenter, "▶")

        # 保存侧边栏区域供点击检测
        self._sidebar_rect   = QRectF(x0, mg, sb_w-int(20*s), h-mg*2)
        self._sidebar_list_y = list_top
        self._sidebar_item_h = item_h
        self._sidebar_scroll = scroll_start
        self._sidebar_vis_n  = visible_n
        self._sidebar_x0     = x0

    # ════════════════════════════════════════════════════════
    #  操作
    # ════════════════════════════════════════════════════════

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择音乐文件夹")
        if folder:
            cnt = self.audio.load_folder(folder)
            if cnt > 0:
                self.audio.play_index(0); self._update_track_info()
                self._btn_play_text = "⏸"; self.update()
            else:
                self._track_title = "未找到音乐文件"; self._track_artist = folder

    def _play_pause(self):
        if not self.audio.playlist: self._open_folder(); return
        if not self.audio.playing and self.audio.current_index < 0:
            self.audio.play_index(0); self._update_track_info()
        else:
            self.audio.toggle()
        self._btn_play_text = "⏸" if self.audio.playing else "▶"
        self.update()

    def _next(self):
        if self.audio.playlist:
            self._start_transition()
            self.audio.next(); self._update_track_info()
            self._btn_play_text = "⏸"; self.update()

    def _prev(self):
        if self.audio.playlist:
            self._start_transition()
            self.audio.prev(); self._update_track_info()
            self._btn_play_text = "⏸"; self.update()

    def _auto_next(self):
        self._next()

    def _start_transition(self):
        """触发歌曲切换过渡动画"""
        self._transition    = self._TRANS_FRAMES
        self._title_slide_x = self.width() * 0.25

    def _cycle_viz(self):
        # -1(空) → 0 → 1 → ... → 6 → -1(空)，循环 8 档
        self._viz_style = -1 if self._viz_style >= 6 else self._viz_style + 1
        self.update()

    def _toggle_mute(self):
        self.audio.toggle_mute(); self.update()

    def _toggle_sidebar(self):
        self._sidebar_open = not self._sidebar_open

    def _cycle_play_mode(self):
        self.audio.cycle_play_mode()
        self._settings.setValue("play_mode", self.audio.play_mode)
        self.update()

    def _update_track_info(self):
        if 0 <= self.audio.current_index < len(self.audio.playlist):
            path = self.audio.playlist[self.audio.current_index]
            meta = AudioEngine.get_metadata(path)
            self._track_title  = meta['title']
            self._track_artist = meta['artist']
            self._cover_pixmap = meta['cover']
            self._lrc_lines    = LrcParser.load(path)
            self._lrc_idx      = -1
            self._save_state()

    def _save_state(self):
        if self.audio.playlist and self.audio.current_index >= 0:
            self._settings.setValue("last_folder",
                str(Path(self.audio.playlist[0]).parent))
            self._settings.setValue("last_index", self.audio.current_index)

    def _restore_state(self):
        folder = self._settings.value("last_folder")
        if folder and os.path.isdir(folder):
            cnt = self.audio.load_folder(folder)
            if cnt > 0:
                idx = self._settings.value("last_index", 0, type=int)
                self.audio.play_index(min(idx, cnt-1))
                self._update_track_info()
                self._btn_play_text = "⏸"; self.update(); return
        self._track_title = "未播放"; self._track_artist = "请打开音乐文件夹"

    # ════════════════════════════════════════════════════════
    #  鼠标 & 键盘
    # ════════════════════════════════════════════════════════

    def _corner_at(self, pos):
        z = 30; w, h = self.width(), self.height()
        if pos.x()<z     and pos.y()<z:   return 0
        if pos.x()>w-z   and pos.y()<z:   return 1
        if pos.x()<z     and pos.y()>h-z: return 2
        if pos.x()>w-z   and pos.y()>h-z: return 3
        return None

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton: return
        pos = event.position()

        # 侧边栏点击（列表项切歌）
        if (self._sidebar_frac > 0.3 and hasattr(self,'_sidebar_rect')
                and self._sidebar_rect.contains(pos)):
            # 关闭按钮区（标题行）
            ly = pos.y() - self._sidebar_list_y
            if ly >= 0:
                item_idx = self._sidebar_scroll + int(ly // self._sidebar_item_h)
                if 0 <= item_idx < len(self.audio.playlist):
                    self._start_transition()
                    self.audio.play_index(item_idx)
                    self._update_track_info()
                    self._btn_play_text = "⏸"; self.update()
            return

        # 播放模式按钮
        if hasattr(self,'_mode_btn_rect') and self._mode_btn_rect.contains(pos):
            self._cycle_play_mode(); return

        # 控制按钮
        for i, r in enumerate(self._btn_regions):
            if r.contains(pos):
                [self._prev, self._play_pause, self._next][i](); return

        # 进度条
        if (hasattr(self,'_progress_rect') and self.audio.duration()>0
                and self._progress_rect.contains(pos)):
            self._seeking = True
            frac = max(0.0,min(1.0,(pos.x()-self._progress_rect.x())/self._progress_rect.width()))
            ms   = int(self.audio.duration()*frac)
            self.audio.seek(ms)
            self._seek_bubble_x = pos.x(); self._seek_bubble_t = ms
            return

        # 音量条
        if self._vol_bar_rect and self._vol_bar_rect.contains(pos):
            self._vol_dragging = True; self._update_volume_from_pos(pos); return

        # 角落缩放
        corner = self._corner_at(pos)
        if corner is not None:
            self._resize_corner = corner
            self._resize_start  = event.globalPosition().toPoint()
            self._resize_min    = self.window().minimumSize()
            g = self.window().geometry()
            self._resize_ratio  = g.width()/g.height(); return

        # 螺丝
        if hasattr(self,'_screw_positions'):
            r = 10*(self.width()/680)
            for idx in range(4):
                sx,sy = self._screw_positions[idx]
                if ((pos.x()-sx)**2+(pos.y()-sy)**2)**0.5 <= r:
                    [self._open_folder, self.window().close,
                     self._cycle_viz,   self._toggle_mute][idx](); return

        self._drag_start = event.globalPosition().toPoint()

    def _update_volume_from_pos(self, pos):
        if not self._vol_bar_rect: return
        r    = self._vol_bar_rect
        frac = 1.0 - max(0.0, min(1.0,(pos.y()-r.y())/r.height()))
        self.audio.set_volume(frac); self.update()

    def mouseMoveEvent(self, event):
        pos = event.position()

        if self._seeking and event.buttons() & Qt.MouseButton.LeftButton:
            if hasattr(self,'_progress_rect') and self.audio.duration()>0:
                frac = max(0.0,min(1.0,(pos.x()-self._progress_rect.x())/self._progress_rect.width()))
                ms   = int(self.audio.duration()*frac)
                self.audio.seek(ms)
                self._seek_bubble_x = pos.x(); self._seek_bubble_t = ms
            return

        if self._vol_dragging and event.buttons() & Qt.MouseButton.LeftButton:
            self._update_volume_from_pos(pos); return

        if (self._resize_corner is not None
                and event.buttons() & Qt.MouseButton.LeftButton):
            np_ = event.globalPosition().toPoint()
            d   = np_ - self._resize_start
            g   = self.window().geometry()
            mw,mh = self._resize_min.width(), self._resize_min.height()
            ow,oh = g.width(), g.height()
            c     = self._resize_corner
            ratio = getattr(self,'_resize_ratio',680/420)

            # 原始目标尺寸（未钳位）
            raw_w = ow + (d.x() if c in (1,3) else -d.x())
            raw_h = oh + (d.y() if c in (2,3) else -d.y())

            # 按变化量更大的一维驱动，另一维跟随比例
            dw = abs(raw_w - ow)
            dh = abs(raw_h - oh)
            if dw > dh:
                w_ = raw_w
                h_ = w_ / ratio
            else:
                h_ = raw_h
                w_ = h_ * ratio

            # 钳位到最小尺寸
            w_ = max(mw, w_)
            h_ = max(mh, h_)

            x = g.x() + (ow - int(w_) if c in (0,2) else 0)
            y = g.y() + (oh - int(h_) if c in (0,1) else 0)
            self.window().setGeometry(int(x), int(y), int(w_), int(h_))
            self._resize_start = np_

        elif self._drag_start and event.buttons() & Qt.MouseButton.LeftButton:
            d = event.globalPosition().toPoint()-self._drag_start
            self.window().move(self.window().pos()+d)
            self._drag_start = event.globalPosition().toPoint()

        else:
            # 悬停检测
            hc = -1
            for i,r in enumerate(self._btn_regions):
                if r.contains(pos): hc = i; break
            if hc != self._btn_hover: self._btn_hover = hc; self.update()

            prev_mode_hover = getattr(self, '_mode_hover', False)
            self._mode_hover = (hasattr(self,'_mode_btn_rect')
                                and self._mode_btn_rect.contains(pos))
            if self._mode_hover != prev_mode_hover:
                self.update()

            # 侧边栏 hover
            if (self._sidebar_frac > 0.3 and hasattr(self,'_sidebar_list_y')
                    and hasattr(self,'_sidebar_rect') and self._sidebar_rect.contains(pos)):
                ly  = pos.y()-self._sidebar_list_y
                hov = (self._sidebar_scroll+int(ly//self._sidebar_item_h)
                       if ly >= 0 else -1)
                if hov != self._sidebar_hover:
                    self._sidebar_hover = hov; self.update()
                self.setCursor(Qt.CursorShape.PointingHandCursor); return
            else:
                self._sidebar_hover = -1

            if (hasattr(self,'_progress_rect') and self._progress_rect.contains(pos)
                    and self.audio.duration()>0):
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            elif self._vol_bar_rect and self._vol_bar_rect.contains(pos):
                self.setCursor(Qt.CursorShape.SizeVerCursor)
            elif self._btn_hover >= 0 or self._mode_hover:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                c = self._corner_at(pos)
                if   c in (0,3): self.setCursor(Qt.CursorShape.SizeFDiagCursor)
                elif c in (1,2): self.setCursor(Qt.CursorShape.SizeBDiagCursor)
                else:            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        self._seeking = False
        self._seek_bubble_x = None; self._seek_bubble_t = None
        self._vol_dragging = False
        self._resize_corner = None; self._drag_start = None

    def mouseDoubleClickEvent(self, event):
        """双击切换侧边栏"""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            # 不在进度条、按钮区域时才触发
            in_btn = any(r.contains(pos) for r in self._btn_regions)
            in_prog = hasattr(self,'_progress_rect') and self._progress_rect.contains(pos)
            if not in_btn and not in_prog:
                self._toggle_sidebar()

    def keyPressEvent(self, event):
        k = event.key()
        if   k == Qt.Key.Key_Space: self._play_pause()
        elif k == Qt.Key.Key_Right: self._next()
        elif k == Qt.Key.Key_Left:  self._prev()
        elif k == Qt.Key.Key_M:     self._toggle_mute()
        elif k == Qt.Key.Key_L:     self._toggle_sidebar()
        elif k == Qt.Key.Key_R:     self._cycle_play_mode()
        elif k == Qt.Key.Key_Up:
            self.audio.set_volume(min(1.0,self.audio.volume+0.05)); self.update()
        elif k == Qt.Key.Key_Down:
            self.audio.set_volume(max(0.0,self.audio.volume-0.05)); self.update()

    def closeEvent(self, event):
        self._save_state(); self.audio.stop()
        self._anim_timer.stop(); event.accept()


# ════════════════════════════════════════════════════════════
#  主窗口
# ════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(680, 420); self.setMinimumSize(500, 320)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        # WA_TranslucentBackground：允许透明
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # WA_NoSystemBackground：禁止 Qt/系统预填充背景（消除"烙印"根因）
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setStyleSheet("background: transparent;")
        self.player = CassettePlayer()
        self.setCentralWidget(self.player)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = app.palette()
    pal.setColor(pal.ColorRole.Window, QColor(10,12,20))
    app.setPalette(pal)
    w = MainWindow(); w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
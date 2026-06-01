"""
透明磁带音乐播放器 — Cassette Player
─────────────────────────────────────────
• 磁带轮旋转动画 + 频谱音浪可视化
• 上一首 / 播放暂停 / 下一首
• 支持 MP3 / FLAC / WAV / OGG / M4A / AAC
• 无边框透明窗口 + 四角拖拽缩放
• 状态记忆（上次文件夹 & 歌曲）
"""
import sys        # 系统相关，获取平台信息
import os          # 文件路径操作
import math        # 数学函数（sin/cos 用于旋转角度）
import random      # 随机数（频谱回退时使用）
import threading   # 后台线程（异步 PCM 解码）
from pathlib import Path  # 面向对象的文件路径处理

# ── 科学计算（FFT 音频频谱分析）───────────────
import numpy as np                          # PCM 数组 + numpy.fft.rfft
from scipy.fft import rfft, rfftfreq        # 实数 FFT + 频率轴
from pydub import AudioSegment              # 音频解码（通过 ffmpeg）

# ── PyQt6 GUI 组件 ────────────────────────────
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget,
                              QVBoxLayout, QHBoxLayout, QPushButton,
                              QFileDialog, QListWidget, QLabel, QSlider)
# ── PyQt6 核心（事件、定时器、几何、URL、设置）──
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF, QUrl, QSettings
# ── PyQt6 绘图（画笔、颜色、画刷、渐变、路径）──
from PyQt6.QtGui import (QPainter, QColor, QBrush, QPen, QFont,
                          QLinearGradient, QRadialGradient, QPainterPath,
                          QFontDatabase, QAction)
# ── PyQt6 多媒体（音频播放器 + 输出设备）───────
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

# ── mutagen：读取音频文件元数据（ID3 tags）─────
from mutagen import File as MutagenFile
from mutagen.mp3 import MP3


# ============================================================
#  频谱解码器 — 后台解码 PCM + FFT 频谱分析
# ============================================================

class SpectrumDecoder:
    def __init__(self, fft_size=1024, target_rate=22050):
        self._fft_size = fft_size
        self._target_rate = target_rate
        self._pcm = None
        self._sample_rate = 0
        self._total_samples = 0
        self._ready = False
        self._current_file = None
        self._lock = threading.Lock()
        self._window = None
        self._bin_map = None
        self._bin_counts = None
        self._bar_count = 0

    @property
    def ready(self):
        return self._ready

    def is_current(self, filepath):
        return self._current_file == filepath

    def reset(self):
        with self._lock:
            self._pcm = None
            self._sample_rate = 0
            self._total_samples = 0
            self._ready = False
            self._current_file = None
            self._window = None
            self._bin_map = None
            self._bin_counts = None
            self._bar_count = 0

    def load_async(self, filepath):
        if self.is_current(filepath) and self._ready:
            return
        self.reset()
        t = threading.Thread(target=self._load, args=(filepath,), daemon=True)
        t.start()

    def _load(self, filepath):
        try:
            audio = AudioSegment.from_file(filepath)
            if audio.frame_rate > self._target_rate:
                audio = audio.set_frame_rate(self._target_rate)
            audio = audio.set_channels(1)
            sr = audio.frame_rate
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            max_val = float(2 ** (audio.sample_width * 8 - 1))
            samples /= max_val
            with self._lock:
                self._pcm = samples
                self._sample_rate = sr
                self._total_samples = len(samples)
                self._current_file = filepath
                self._ready = True
                if self._window is None or len(self._window) != self._fft_size:
                    self._window = np.hanning(self._fft_size).astype(np.float32)
        except Exception as e:
            print(f"[SpectrumDecoder] 解码失败: {filepath} — {e}")

    def get_spectrum(self, position_ms, bar_count):
        if not self._ready or self._pcm is None:
            return None
        if bar_count < 1:
            return None
        if self._bin_map is None or self._bar_count != bar_count:
            self._build_bin_map(bar_count)
        sr = self._sample_rate
        total = self._total_samples
        fft_n = self._fft_size
        idx = int(position_ms / 1000.0 * sr)
        idx = max(0, min(idx, total - fft_n))
        if idx + fft_n <= total:
            window = self._pcm[idx:idx + fft_n] * self._window
        else:
            window = np.zeros(fft_n, dtype=np.float32)
            avail = total - idx
            window[:avail] = self._pcm[idx:total] * self._window[:avail]
        mag = np.abs(rfft(window))
        bar_sums = np.bincount(self._bin_map, weights=mag, minlength=bar_count)
        with np.errstate(divide='ignore', invalid='ignore'):
            bars = bar_sums / np.maximum(self._bin_counts, 1)
        bars = np.sqrt(bars)
        max_val = bars.max()
        if max_val > 0:
            bars = bars / max_val
        bars = np.clip(bars, 0.05, 1.0)
        return bars.tolist()

    def get_waveform(self, position_ms, sample_count):
        if not self._ready or self._pcm is None:
            return None
        sr = self._sample_rate
        total = self._total_samples
        idx = int(position_ms / 1000.0 * sr)
        start = max(0, idx - sample_count // 2)
        end = min(total, start + sample_count)
        chunk = self._pcm[start:end]
        if len(chunk) < sample_count:
            padded = np.zeros(sample_count, dtype=np.float32)
            padded[:len(chunk)] = chunk
            return padded.tolist()
        return chunk.tolist()

    def _build_bin_map(self, bar_count):
        sr = self._sample_rate
        fft_n = self._fft_size
        freqs = rfftfreq(fft_n, 1.0 / sr)
        n_bins = len(freqs)
        min_freq = 30.0
        max_freq = min(sr / 2.0, 10000.0)
        log_min = np.log10(min_freq)
        log_max = np.log10(max_freq)
        edges = np.logspace(log_min, log_max, bar_count + 1)
        bin_map = np.zeros(n_bins, dtype=np.int32)
        bin_counts = np.zeros(bar_count, dtype=np.int32)
        for bi in range(n_bins):
            f = freqs[bi]
            bar_idx = np.searchsorted(edges, f) - 1
            bar_idx = max(0, min(bar_idx, bar_count - 1))
            bin_map[bi] = bar_idx
            bin_counts[bar_idx] += 1
        bin_counts = np.maximum(bin_counts, 1)
        self._bin_map = bin_map
        self._bin_counts = bin_counts
        self._bar_count = bar_count


# ============================================================
#  音频引擎
# ============================================================

class AudioEngine:
    def __init__(self, parent=None):
        self._player = QMediaPlayer(parent)
        self._audio = QAudioOutput(parent)
        self._player.setAudioOutput(self._audio)
        self._audio.setVolume(0.8)
        self._decoder = SpectrumDecoder(fft_size=1024, target_rate=22050)
        self._playlist = []
        self._index = -1
        self._playing = False
        self._player.playbackStateChanged.connect(self._on_state_change)

    def _on_state_change(self, state):
        self._playing = (state == QMediaPlayer.PlaybackState.PlayingState)

    @property
    def decoder_ready(self):
        return self._decoder.ready

    def get_spectrum(self, position_ms, bar_count):
        return self._decoder.get_spectrum(position_ms, bar_count)

    def _start_decode(self, filepath):
        if not self._decoder.is_current(filepath):
            self._decoder.load_async(filepath)

    def cleanup(self):
        self._decoder.reset()

    @property
    def playing(self):
        return self._playing

    @property
    def current_index(self):
        return self._index

    @property
    def playlist(self):
        return self._playlist

    def position(self):
        return self._player.position()

    def duration(self):
        return self._player.duration()

    def seek(self, ms):
        self._player.setPosition(ms)

    def load_folder(self, folder_path):
        extensions = {'.mp3', '.flac', '.wav', '.ogg', '.m4a', '.aac'}
        self._playlist = []
        for ext in extensions:
            for f in Path(folder_path).rglob(f'*{ext}'):
                self._playlist.append(str(f))
        self._playlist.sort()
        return len(self._playlist)

    def play_index(self, index):
        if 0 <= index < len(self._playlist):
            path = self._playlist[index]
            self._player.setSource(QUrl.fromLocalFile(path))
            self._player.play()
            self._playing = True
            self._index = index
            self._start_decode(path)
            return True
        return False

    def toggle(self):
        if self._playing:
            self._player.pause()
        else:
            self._player.play()

    def stop(self):
        self._player.stop()
        self._playing = False

    def next(self):
        if self._playlist:
            nxt = (self._index + 1) % len(self._playlist)
            return self.play_index(nxt)
        return False

    def prev(self):
        if self._playlist:
            prv = (self._index - 1) % len(self._playlist)
            return self.play_index(prv)
        return False

    @staticmethod
    def get_metadata(filepath):
        try:
            if filepath.endswith('.mp3'):
                audio = MP3(filepath)
                tags = audio.tags
                if tags:
                    title = str(tags.get('TIT2', Path(filepath).stem))
                    artist = str(tags.get('TPE1', 'Unknown'))
                    return {'title': title, 'artist': artist, 'path': filepath}
            return {'title': Path(filepath).stem, 'artist': 'Unknown', 'path': filepath}
        except Exception:
            return {'title': Path(filepath).stem, 'artist': 'Unknown', 'path': filepath}


# ============================================================
#  磁带播放器主控件
# ============================================================

class CassettePlayer(QWidget):

    def __init__(self):
        super().__init__()
        self.audio = AudioEngine(self)
        self.rotation_angle = 0.0
        self._settings = QSettings("CassettePlayer", "CassettePlayer")

        self._viz_style = int(self._settings.value("viz_style", 0))
        # ▼ 修复：不再使用 _viz_clear 标志位，改用显式擦除区域
        self._viz_erase_pending = False   # 标记下一帧需要擦除频谱区再画新风格

        self._bar_count = 60
        self._bars = [0.05] * self._bar_count
        self._bar_targets = [0.05] * self._bar_count
        self._particles = []
        self._bar_frame = 0
        self._hue_offset = 0.0
        self._drag_start = None
        self._seeking = False

        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._tick)
        self._anim_timer.start(30)

        self._track_title = "未播放"
        self._track_artist = "请打开音乐文件夹"

        self._setup_ui()
        self._restore_state()

    def _setup_ui(self):
        self.setMinimumSize(500, 320)
        self.setMouseTracking(True)
        self.setStyleSheet("background: transparent;")
        self._btn_play_text = "▶"
        self._btn_regions = []
        self._btn_hover = -1
        self._file_list = []

    def resizeEvent(self, event):
        super().resizeEvent(event)

    def _reel_center_y(self):
        w = self.width()
        s = w / 680
        margin = int(18 * s)
        label_y = margin + int(8 * s)
        label_h = int(68 * s)
        waveform_max_h = int(78 * s)
        waveform_base_offset = int(6 * s)
        progress_space = int(14 * s)
        cassette_bottom = self.height() - margin
        content_bottom = cassette_bottom - waveform_base_offset - waveform_max_h - progress_space
        return label_y + label_h + (content_bottom - label_y - label_h) // 2

    # ================================================================
    #  动画循环
    # ================================================================

    def _tick(self):
        if self.audio.playing:
            self.rotation_angle += 3.0
            self._bar_frame += 1
            if self.audio.decoder_ready:
                spectrum = self.audio.get_spectrum(
                    self.audio.position(), self._bar_count)
                if spectrum is not None:
                    for i in range(self._bar_count):
                        self._bar_targets[i] = spectrum[i]
                else:
                    self._random_bars()
            else:
                if self._bar_frame % 4 == 0:
                    self._random_bars()
        else:
            for i in range(self._bar_count):
                self._bar_targets[i] = 0.05

        for i in range(self._bar_count):
            self._bars[i] += (self._bar_targets[i] - self._bars[i]) * 0.18

        self._hue_offset = (self._hue_offset + 0.003) % 1.0
        self.update()

    def _random_bars(self):
        n = self._bar_count
        for i in range(0, n, 3):
            self._bar_targets[i] = random.uniform(0.2, 1.0)
            self._bar_targets[min(i + 1, n - 1)] = random.uniform(0.12, 0.65)
            self._bar_targets[min(i + 2, n - 1)] = random.uniform(0.05, 0.35)

    # ================================================================
    #  绘制
    # ================================================================

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # ▼ 修复关键点1：透明窗口必须先用 CompositionMode_Clear 清空整个画布，
        #   否则上一帧的像素会直接保留（透明窗口不自动清除）。
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        p.fillRect(self.rect(), Qt.GlobalColor.transparent)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        w, h = self.width(), self.height()
        base_w = 680
        s = w / base_w
        margin = int(18 * s)
        cassette_bottom = h - margin
        bw = w - margin * 2
        bh = h - margin * 2

        # ── 玻璃磁带主体 ──
        path = QPainterPath()
        path.addRoundedRect(QRectF(margin, margin, bw, bh), 22, 22)
        p.fillPath(path, QColor(55, 60, 72, 110))
        p.setPen(QPen(QColor(170, 180, 200, 150), 2))
        p.drawPath(path)

        path2 = QPainterPath()
        path2.addRoundedRect(QRectF(margin + 3, margin + 3, bw - 6, bh - 6), 20, 20)
        p.setPen(QPen(QColor(255, 255, 255, 30), 1))
        p.drawPath(path2)

        # ── 标签区 ──
        label_y = margin + int(10 * s)
        label_h = int(64 * s)
        slant = int(10 * s)
        tl_x = margin + int(26 * s)
        tr_x = w - margin - int(26 * s)
        bl_x = tl_x + slant
        br_x = tr_x - slant
        top_y = label_y
        bottom_y = label_y + label_h
        cr = int(8 * s)

        def _rounded_trapezoid(tlx, trx, blx, brx, ty, by, radius):
            path = QPainterPath()
            path.moveTo(tlx + radius, ty)
            path.lineTo(trx - radius, ty)
            path.arcTo(trx - 2 * radius, ty, 2 * radius, 2 * radius, 90, -90)
            path.lineTo(brx, by - radius)
            path.arcTo(brx - 2 * radius, by - 2 * radius, 2 * radius, 2 * radius, 0, -90)
            path.lineTo(blx + radius, by)
            path.arcTo(blx, by - 2 * radius, 2 * radius, 2 * radius, 270, -90)
            path.lineTo(tlx, ty + radius)
            path.arcTo(tlx, ty, 2 * radius, 2 * radius, 180, -90)
            path.closeSubpath()
            return path

        shadow_path = _rounded_trapezoid(tl_x + int(2*s), tr_x - int(2*s), bl_x, br_x,
                                         top_y + int(3*s), bottom_y + int(4*s), cr)
        p.fillPath(shadow_path, QColor(0, 0, 0, 40))

        label_path = _rounded_trapezoid(tl_x, tr_x, bl_x, br_x, top_y, bottom_y, cr)
        p.fillPath(label_path, QColor(72, 64, 50, 160))
        p.setPen(QPen(QColor(180, 170, 140, 90), 1))
        p.drawPath(label_path)

        hl_path = _rounded_trapezoid(tl_x + int(2*s), tr_x - int(2*s),
                                     bl_x + int(4*s), br_x - int(4*s),
                                     top_y + int(1*s), top_y + int(8*s), int(5*s))
        p.fillPath(hl_path, QColor(255, 255, 255, 35))

        grad = QLinearGradient(0, top_y, 0, bottom_y)
        grad.setColorAt(0, QColor(255, 255, 255, 100))
        grad.setColorAt(0.3, QColor(255, 255, 255, 30))
        grad.setColorAt(1, QColor(255, 255, 255, 0))
        hl_global = _rounded_trapezoid(tl_x, tr_x, bl_x, br_x, top_y, bottom_y, cr)
        p.fillPath(hl_global, grad)

        p.setPen(QPen(QColor(200, 190, 160, 50), max(1, int(1 * s))))
        for i in range(2):
            ly = top_y + int(22 * s) + i * int(20 * s)
            p.drawLine(int(bl_x + int(16 * s)), ly, int(br_x - int(16 * s)), ly)

        font_s = max(12, int(16 * s))
        artist_font_s = max(10, int(13 * s))
        label_cx = (tl_x + tr_x) / 2

        title_font = QFont("Microsoft YaHei", font_s)
        title_font.setBold(True)
        p.setFont(title_font)
        p.setPen(QColor(240, 235, 220, 220))
        title_rect = QRectF(tl_x + 10, top_y + int(3 * s) + 4, tr_x - tl_x - 20, 24 * s)
        p.drawText(title_rect, Qt.AlignmentFlag.AlignCenter, self._track_title)

        artist_font = QFont("Microsoft YaHei", artist_font_s)
        p.setFont(artist_font)
        p.setPen(QColor(200, 190, 170, 180))
        artist_rect = QRectF(tl_x + 10, top_y + int(3 * s) + 28 * s, tr_x - tl_x - 20, 20 * s)
        p.drawText(artist_rect, Qt.AlignmentFlag.AlignCenter, self._track_artist)

        # ── 磁带轮 ──
        reel_r = int(44 * s)
        reel_y = self._reel_center_y()
        reel_spacing = int(170 * s)
        r1_x = w // 2 - reel_spacing
        r2_x = w // 2 + reel_spacing
        for cx in [r1_x, r2_x]:
            self._draw_reel(p, cx, reel_y, reel_r)

        # ── 控制按钮 ──
        btn_s = int(44 * s)
        btn_spacing = int(reel_spacing * 0.50)
        center_x = w // 2
        btn_y = reel_y - btn_s // 2
        btn_font = QFont("Segoe UI Symbol", max(10, int(20 * s)))
        p.setFont(btn_font)

        btns = [
            (center_x - btn_spacing - btn_s // 2, btn_y, btn_s, "⏮"),
            (center_x - btn_s // 2, btn_y, btn_s, self._btn_play_text),
            (center_x + btn_spacing - btn_s // 2, btn_y, btn_s, "⏭"),
        ]
        self._btn_regions = []
        for i, (bx, by_, bs, sym) in enumerate(btns):
            rect = QRectF(bx, by_, bs, bs)
            self._btn_regions.append(rect)
            hovered = (self._btn_hover == i)
            if hovered:
                p.setPen(QColor(255, 255, 255, 255))
            else:
                p.setPen(QColor(180, 180, 180, 180))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, sym)

        # ── 四角螺丝 ──
        screw_r = int(7 * s)
        top_off = int(18 * s)
        bot_off = int(26 * s)
        screw_positions = [
            (margin + top_off, margin + top_off + int(7 * s)),
            (w - margin - top_off, margin + top_off + int(7 * s)),
            (margin + bot_off + int(2 * s), cassette_bottom - bot_off + int(2 * s)),
            (w - margin - bot_off - int(2 * s), cassette_bottom - bot_off + int(2 * s)),
        ]
        self._screw_positions = screw_positions

        for idx, (sx, sy) in enumerate(screw_positions):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(165, 170, 180, 170))
            p.drawEllipse(QPointF(sx, sy), screw_r, screw_r)
            p.setBrush(QColor(135, 140, 150, 190))
            p.drawEllipse(QPointF(sx, sy), screw_r - int(3 * s), screw_r - int(3 * s))

            lw = int(2 * s)
            if idx == 0:
                p.setPen(QPen(QColor(220, 225, 235, 200), max(1, lw)))
                d = int(3 * s)
                p.drawLine(int(sx - d), int(sy), int(sx + d), int(sy))
                p.drawLine(int(sx), int(sy - d), int(sx), int(sy + d))
            elif idx == 1:
                p.setPen(QPen(QColor(220, 225, 235, 200), max(1, lw)))
                d = int(2 * s)
                p.drawLine(int(sx - d), int(sy - d), int(sx + d), int(sy + d))
                p.drawLine(int(sx + d), int(sy - d), int(sx - d), int(sy + d))
            elif idx == 2:
                p.setPen(QPen(QColor(255, 90, 100, 200), max(1, lw)))
                d = int(3 * s)
                p.drawLine(int(sx - d), int(sy - d), int(sx), int(sy + d))
                p.drawLine(int(sx + d), int(sy - d), int(sx), int(sy + d))
            else:
                p.setPen(QPen(QColor(100, 105, 115, 150), 1))
                p.drawLine(int(sx - d), int(sy), int(sx + d), int(sy))
                p.drawLine(int(sx), int(sy - d), int(sx), int(sy + d))

        # ── 进度条 ──
        wave_start_x = (w // 2 - reel_spacing) - reel_r
        wave_end_x = (w // 2 + reel_spacing) + reel_r
        wave_total_w = wave_end_x - wave_start_x

        waveform_max_h = int(78 * s)
        waveform_base_offset = int(6 * s)

        dur = self.audio.duration()
        pos_ms = self.audio.position()

        progress_y = cassette_bottom - waveform_base_offset - waveform_max_h - int(24 * s)

        def _fmt(ms):
            if ms <= 0:
                return "00:00"
            sec = ms // 1000
            return f"{sec // 60:02d}:{sec % 60:02d}"

        time_font = QFont("Consolas", max(9, int(12 * s)))
        p.setFont(time_font)
        p.setPen(QColor(200, 200, 200, 160))
        elapsed_text = _fmt(pos_ms)
        remain_text = "-" + _fmt(max(0, dur - pos_ms))
        p.drawText(QRectF(wave_start_x - int(70 * s), progress_y - int(12 * s),
                          int(65 * s), int(20 * s)),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   elapsed_text)
        p.drawText(QRectF(wave_end_x + int(5 * s), progress_y - int(12 * s),
                          int(65 * s), int(20 * s)),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   remain_text)

        progress_h = int(4 * s)
        progress_rect = QRectF(wave_start_x, progress_y, wave_total_w, progress_h)
        self._progress_rect = progress_rect

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 30))
        p.drawRoundedRect(progress_rect, 2, 2)

        if dur > 0:
            frac = min(pos_ms / dur, 1.0)
            filled_w = int(progress_rect.width() * frac)
            if filled_w > 0:
                filled_rect = QRectF(progress_rect.x(), progress_rect.y(),
                                     filled_w, progress_rect.height())
                p.setBrush(QColor(220, 200, 150, 180))
                p.drawRoundedRect(filled_rect, 2, 2)

        if dur > 0 and pos_ms >= 0:
            dot_x = progress_rect.x() + progress_rect.width() * frac
            dot_y = progress_rect.center().y()
            heart_font = QFont("Segoe UI Emoji", max(16, int(24 * s)))
            p.setFont(heart_font)
            p.setPen(QColor(255, 100, 130, 240))
            p.drawText(QRectF(dot_x - int(20 * s), dot_y - int(23 * s),
                              int(40 * s), int(40 * s)),
                       Qt.AlignmentFlag.AlignCenter, "❤")

        # ── 频谱（传入 wave 区域参数供擦除使用）──
        base_y = cassette_bottom - waveform_base_offset
        max_bar_h = waveform_max_h
        self._draw_spectrum(p, wave_start_x, wave_end_x, base_y, max_bar_h, s)

    # ================================================================
    #  频谱可视化
    # ================================================================

    def _morandi_color(self, i, t, bar_count):
        palette = [
            (185, 150, 145), (200, 180, 165), (155, 170, 150), (145, 155, 175),
            (170, 165, 180), (180, 160, 155), (160, 170, 170), (190, 175, 160),
        ]
        idx = int((i / bar_count * len(palette) + self._hue_offset * len(palette)) % len(palette))
        br, bg, bb = palette[idx]
        scale = 0.6 + t * 0.4
        return QColor(min(255, int(br * scale)), min(255, int(bg * scale)), min(255, int(bb * scale)))

    def _rainbow_color(self, i, t, bar_count):
        hue = (i / bar_count + self._hue_offset) % 1.0
        val = 0.65 + t * 0.35
        chroma = val
        h6 = hue * 6
        hx = chroma * (1 - abs(h6 % 2 - 1))
        cm = val - chroma
        if h6 < 1:       rf, gf, bf = chroma, hx, 0
        elif h6 < 2:     rf, gf, bf = hx, chroma, 0
        elif h6 < 3:     rf, gf, bf = 0, chroma, hx
        elif h6 < 4:     rf, gf, bf = 0, hx, chroma
        elif h6 < 5:     rf, gf, bf = hx, 0, chroma
        else:            rf, gf, bf = chroma, 0, hx
        return QColor(int((rf + cm) * 255), int((gf + cm) * 255), int((bf + cm) * 255))

    def _bar_color(self, i, t, bar_count):
        if self._viz_style < 6:
            return self._morandi_color(i, t, bar_count)
        else:
            return self._rainbow_color(i, t, bar_count)

    def _draw_spectrum(self, p, sx, ex, base_y, max_h, s):
        total_w = ex - sx
        bar_count = min(60, len(self._bars))
        self._bar_count = bar_count

        # ▼ 修复关键点2：切换风格时，先用 CompositionMode_Clear 精确擦除频谱区域，
        #   再恢复正常混合模式绘制新风格，彻底消除残留。
        if self._viz_erase_pending:
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            erase_rect = QRectF(sx, base_y - max_h - int(4 * s),
                                total_w, max_h + int(8 * s))
            p.fillRect(erase_rect, Qt.GlobalColor.transparent)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            self._viz_erase_pending = False
            # 已擦除，本帧不继续绘制，让机身背景"透出"一帧即可消残留
            return

        styles = [
            self._draw_bars,
            self._draw_radar,
            self._draw_waveform,
            self._draw_mirror,
            self._draw_particles,
            self._draw_pulse,
        ]
        styles[self._viz_style % 6](p, sx, ex, base_y, max_h, s, bar_count, total_w)

    # ── 风格 0：柱状频谱 ──
    def _draw_bars(self, p, sx, ex, base_y, max_h, s, n, tw):
        cell_w = tw / n
        bar_w = max(2.0, cell_w * 0.7)
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(n):
            t = self._bars[i]
            bh = max(2, int(t * max_h))
            bx = sx + i * cell_w
            by = base_y - bh
            c = self._bar_color(i, t, n)
            p.setBrush(QColor(c.red(), c.green(), c.blue(), 30))
            p.drawRoundedRect(QRectF(bx - 3, by - 6, bar_w + 6, bh + 10), 6, 6)
            p.setBrush(QColor(c.red(), c.green(), c.blue(), 80))
            p.drawRoundedRect(QRectF(bx - 1, by - 3, bar_w + 2, bh + 5), 4, 4)
            p.setBrush(QColor(c.red(), c.green(), c.blue(), 240))
            p.drawRoundedRect(QRectF(bx, by, bar_w, bh), 2, 2)
            p.setBrush(QColor(min(c.red() + 80, 255), min(c.green() + 80, 255),
                              min(c.blue() + 80, 255), 200))
            p.drawRoundedRect(QRectF(bx, by, bar_w, max(3, int(bh * 0.25))), 2, 2)

    # ── 风格 1：圆形雷达 ──
    def _draw_radar(self, p, sx, ex, base_y, max_h, s, n, tw):
        cx = (sx + ex) / 2
        cy = base_y - max_h / 2 - int(10 * s)
        max_r = min(tw, max_h) / 2 + int(2 * s)
        inner_r = max_r * 0.05
        angle_step = 2 * math.pi / n
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(n):
            t = self._bars[i]
            bar_len = int(inner_r + t * (max_r - inner_r))
            a = i * angle_step - math.pi / 2
            bx = cx + math.cos(a) * inner_r
            by = cy + math.sin(a) * inner_r
            ex2 = cx + math.cos(a) * bar_len
            ey2 = cy + math.sin(a) * bar_len
            c = self._bar_color(i, t, n)
            bw = max(1.5, tw / n * 0.55)
            p.setBrush(QColor(c.red(), c.green(), c.blue(), 50))
            p.drawEllipse(QPointF(ex2, ey2), bw + 4, bw + 4)
            pen = QPen(QColor(c.red(), c.green(), c.blue(), 230), bw)
            p.setPen(pen)
            p.drawLine(QPointF(bx, by), QPointF(ex2, ey2))
            p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(220, 220, 240, 200))
        p.drawEllipse(QPointF(cx, cy), 1, 1)

    # ── 风格 2：波形曲线 ──
    def _draw_waveform(self, p, sx, ex, base_y, max_h, s, n, tw):
        cy = base_y - max_h / 2
        half_h = max_h / 2
        sample_count = max(200, int(tw))
        samples = self.audio._decoder.get_waveform(self.audio.position(), sample_count)
        if samples is None or len(samples) < 2:
            return
        pts = []
        for i, v in enumerate(samples[:sample_count]):
            x = sx + i * tw / (sample_count - 1)
            y = cy - v * half_h * 0.9
            pts.append(QPointF(x, y))
        path = QPainterPath()
        path.moveTo(pts[0].x(), cy)
        for pt in pts:
            path.lineTo(pt)
        path.lineTo(pts[-1].x(), cy)
        path.closeSubpath()
        mid_c = self._bar_color(n // 2, 0.6, 1)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(mid_c.red(), mid_c.green(), mid_c.blue(), 25))
        p.drawPath(path)
        seg_count = len(pts) - 1
        for i in range(seg_count):
            seg_c = self._bar_color(i, 0.7, seg_count)
            pen = QPen(QColor(seg_c.red(), seg_c.green(), seg_c.blue(), 220), max(1.5, 2.5 * s))
            p.setPen(pen)
            p.drawLine(pts[i], pts[i + 1])

    # ── 风格 3：镜像对称柱状 ──
    def _draw_mirror(self, p, sx, ex, base_y, max_h, s, n, tw):
        cy = base_y - max_h / 2
        half_h = max_h / 2 - int(3 * s)
        cell_w = tw / n
        bar_w = max(2 * s, cell_w * 0.65)
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(n):
            t = self._bars[i]
            bh = max(2 * s, int(t * half_h))
            bx = sx + i * cell_w
            c = self._bar_color(i, t, n)
            p.setBrush(QColor(c.red(), c.green(), c.blue(), 200))
            p.drawRoundedRect(QRectF(bx, cy - bh, bar_w, bh), 1, 1)
            p.setBrush(QColor(min(c.red() + 60, 255), min(c.green() + 60, 255),
                              min(c.blue() + 60, 255), 140))
            p.drawRoundedRect(QRectF(bx, cy - bh, bar_w, max(2 * s, int(bh * 0.3))), 1, 1)
            p.setBrush(QColor(c.red(), c.green(), c.blue(), 120))
            p.drawRoundedRect(QRectF(bx, cy, bar_w, bh), 1, 1)
        p.setPen(QPen(QColor(255, 255, 255, 40), 1))
        p.drawLine(QPointF(sx, cy), QPointF(ex, cy))

    # ── 风格 4：粒子漂浮 ──
    def _draw_particles(self, p, sx, ex, base_y, max_h, s, n, tw):
        cy = base_y - max_h / 2
        half_h = max_h / 2 - int(6 * s)
        if (not hasattr(self, '_particles') or len(self._particles) != 40
                or getattr(self, '_particles_tw', 0) != tw):
            self._particles = []
            self._particles_tw = tw
            import random as _rnd
            for _ in range(40):
                self._particles.append({
                    'frac': _rnd.random(),
                    'y': cy,
                    'target_y': cy,
                    'size': _rnd.uniform(2, 5) * s,
                    'hue': _rnd.random(),
                })
        p.setPen(Qt.PenStyle.NoPen)
        for pi, pt in enumerate(self._particles):
            px = sx + pt['frac'] * tw
            bi = max(0, min(n - 1, int(pt['frac'] * n)))
            t = self._bars[bi]
            pt['target_y'] = cy - t * half_h * (1.0 if (pi % 2) else -1.0)
            pt['y'] += (pt['target_y'] - pt['y']) * 0.12
            pt['hue'] = (pt['hue'] + random.uniform(0, 0.01)) % 1.0
            pt['size'] = max(2 * s, (3 + t * 4) * s)
            c = self._bar_color(bi, t, n)
            sz = pt['size']
            p.setBrush(QColor(c.red(), c.green(), c.blue(), 40))
            p.drawEllipse(QPointF(px, pt['y']), sz + 2 * s, sz + 2 * s)
            p.setBrush(QColor(c.red(), c.green(), c.blue(), 220))
            p.drawEllipse(QPointF(px, pt['y']), sz, sz)

    # ── 风格 5：圆环脉冲 ──
    def _draw_pulse(self, p, sx, ex, base_y, max_h, s, n, tw):
        cx = (sx + ex) / 2
        cy = base_y - max_h / 2 - int(10 * s)
        max_r = min(tw, max_h) / 2 + int(2 * s)
        bass = sum(self._bars[:10]) / 10
        mid = sum(self._bars[10:30]) / 20
        high = sum(self._bars[30:]) / max(1, n - 30)
        rings = [
            (0.25 * max_r, bass, self._bar_color(0, 0.8, n)),
            (0.55 * max_r, mid, self._bar_color(n // 3, 0.7, n)),
            (0.85 * max_r, high, self._bar_color(n * 2 // 3, 0.6, n)),
        ]
        p.setPen(Qt.PenStyle.NoPen)
        for base_r, energy, color in rings:
            r = max(6, int(base_r + energy * max_r * 0.35))
            p.setBrush(QColor(color.red(), color.green(), color.blue(), 15))
            p.drawEllipse(QPointF(cx, cy), r + 16, r + 16)
            p.setBrush(QColor(color.red(), color.green(), color.blue(), 40))
            p.drawEllipse(QPointF(cx, cy), r + 8, r + 8)
            pen = QPen(QColor(color.red(), color.green(), color.blue(),
                              max(80, color.alpha())), max(2.0, 3.0 * s))
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(cx, cy), r, r)
            p.setPen(Qt.PenStyle.NoPen)
        dot_c = self._bar_color(0, bass, 1)
        p.setBrush(QColor(dot_c.red(), dot_c.green(), dot_c.blue(), int(120 + bass * 135)))
        p.drawEllipse(QPointF(cx, cy), max(1, int(2 + bass * 4)), max(1, int(2 + bass * 4)))

    # ================================================================
    #  磁带轮绘制
    # ================================================================

    def _draw_reel(self, p, cx, cy, r):
        p.save()
        p.translate(cx, cy)

        p.setPen(QPen(QColor(140, 150, 170, 60), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(0, 0), r, r)

        p.setPen(QPen(QColor(120, 130, 150, 140), 2))
        p.setBrush(QColor(25, 28, 35, 140))
        p.drawEllipse(QPointF(0, 0), r - 2, r - 2)

        angle_rad = math.radians(self.rotation_angle)
        morandi = [
            QColor(185, 150, 145), QColor(155, 170, 150), QColor(145, 155, 175),
            QColor(190, 175, 150), QColor(170, 160, 180),
        ]
        for i in range(5):
            a = math.radians(i * 72) + angle_rad
            gx = math.cos(a) * (r - 16)
            gy = math.sin(a) * (r - 16)
            c = morandi[i]
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(c.red(), c.green(), c.blue(), 80))
            p.drawEllipse(QPointF(gx + 1, gy + 1), 6, 6)
            p.setBrush(QColor(c.red(), c.green(), c.blue(), 180))
            p.drawEllipse(QPointF(gx, gy), 6, 6)
            p.setBrush(QColor(min(c.red() + 40, 255), min(c.green() + 40, 255),
                              min(c.blue() + 40, 255), 120))
            p.drawEllipse(QPointF(gx - 1, gy - 2), 3, 3)

        p.setPen(QPen(QColor(90, 100, 120, 100), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(0, 0), r - 24, r - 24)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(75, 80, 95, 160))
        p.drawEllipse(QPointF(0, 0), 10, 10)
        p.setBrush(QColor(140, 145, 160, 200))
        p.drawEllipse(QPointF(0, 0), 4, 4)

        p.restore()

    # ================================================================
    #  操作
    # ================================================================

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择音乐文件夹")
        if folder:
            count = self.audio.load_folder(folder)
            if count > 0:
                self._file_list = self.audio.playlist
                self.audio.play_index(0)
                self._update_track_info()
                self._btn_play_text = "⏸"
                self.update()
            else:
                self._track_title = "未找到音乐文件"
                self._track_artist = folder

    def _play_pause(self):
        if not self.audio.playlist:
            self._open_folder()
            return
        if not self.audio.playing and self.audio.current_index < 0:
            self.audio.play_index(0)
            self._update_track_info()
        else:
            self.audio.toggle()
        self._btn_play_text = "⏸" if self.audio.playing else "▶"
        self.update()

    def _next(self):
        if self.audio.playlist:
            self.audio.next()
            self._update_track_info()
            self._btn_play_text = "⏸"
            self.update()

    def _prev(self):
        if self.audio.playlist:
            self.audio.prev()
            self._update_track_info()
            self._btn_play_text = "⏸"
            self.update()

    def _cycle_viz(self):
        """切换可视化风格：设置擦除标志，下一帧 paintEvent 会先清除频谱区再绘新风格"""
        self._viz_style = (self._viz_style + 1) % 12
        self._settings.setValue("viz_style", self._viz_style)
        # ▼ 修复关键点3：只需设置标志，paintEvent 的全画布 Clear 已保证干净，
        #   erase_pending 额外精确擦除频谱区，防止相邻帧出现一帧闪烁。
        self._viz_erase_pending = True
        self.update()

    def _update_track_info(self):
        if 0 <= self.audio.current_index < len(self.audio.playlist):
            path = self.audio.playlist[self.audio.current_index]
            meta = AudioEngine.get_metadata(path)
            self._track_title = meta['title']
            self._track_artist = meta['artist']
            self._save_state()

    def _save_state(self):
        if self.audio.playlist and self.audio.current_index >= 0:
            folder = str(Path(self.audio.playlist[0]).parent)
            self._settings.setValue("last_folder", folder)
            self._settings.setValue("last_index", self.audio.current_index)

    def _restore_state(self):
        folder = self._settings.value("last_folder")
        if folder and os.path.isdir(folder):
            count = self.audio.load_folder(folder)
            if count > 0:
                self._file_list = self.audio.playlist
                last_index = self._settings.value("last_index", 0, type=int)
                if last_index >= count:
                    last_index = 0
                self.audio.play_index(last_index)
                self._update_track_info()
                self._btn_play_text = "⏸"
                self.update()
                return
        self._track_title = "未播放"
        self._track_artist = "请打开音乐文件夹"

    # ================================================================
    #  鼠标 & 键盘事件
    # ================================================================

    def _corner_at(self, pos):
        z = 30
        w, h = self.width(), self.height()
        if pos.x() < z and pos.y() < z:         return 0
        if pos.x() > w - z and pos.y() < z:     return 1
        if pos.x() < z and pos.y() > h - z:     return 2
        if pos.x() > w - z and pos.y() > h - z: return 3
        return None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()

            if hasattr(self, '_btn_regions'):
                for i, r in enumerate(self._btn_regions):
                    if r.contains(pos):
                        if i == 0:   self._prev()
                        elif i == 1: self._play_pause()
                        elif i == 2: self._next()
                        return

            if hasattr(self, '_progress_rect') and self.audio.duration() > 0:
                pr = self._progress_rect
                if pr.contains(pos):
                    self._seeking = True
                    frac = (pos.x() - pr.x()) / pr.width()
                    frac = max(0.0, min(1.0, frac))
                    self.audio.seek(int(self.audio.duration() * frac))
                    return

            corner = self._corner_at(pos)
            if corner is not None:
                self._resize_corner = corner
                self._resize_start = event.globalPosition().toPoint()
                self._resize_min = self.window().minimumSize()
                g = self.window().geometry()
                self._resize_ratio = g.width() / g.height()
                return

            if hasattr(self, '_screw_positions'):
                r = 10 * (self.width() / 680)
                for idx in (0, 1, 2):
                    sx, sy = self._screw_positions[idx]
                    dist = ((pos.x() - sx) ** 2 + (pos.y() - sy) ** 2) ** 0.5
                    if dist <= r:
                        if idx == 0:   self._open_folder()
                        elif idx == 1: self.window().close()
                        else:          self._cycle_viz()
                        return

            self._drag_start = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if (hasattr(self, '_seeking') and self._seeking
                and event.buttons() & Qt.MouseButton.LeftButton):
            if hasattr(self, '_progress_rect') and self.audio.duration() > 0:
                pr = self._progress_rect
                pos = event.position()
                frac = (pos.x() - pr.x()) / pr.width()
                frac = max(0.0, min(1.0, frac))
                self.audio.seek(int(self.audio.duration() * frac))
                return

        if (hasattr(self, '_resize_corner') and self._resize_corner is not None
                and event.buttons() & Qt.MouseButton.LeftButton):
            new_pos = event.globalPosition().toPoint()
            delta = new_pos - self._resize_start
            g = self.window().geometry()
            mw, mh = self._resize_min.width(), self._resize_min.height()
            ow, oh = g.width(), g.height()
            c = self._resize_corner
            ratio = getattr(self, '_resize_ratio', 680 / 420)

            if c in (0, 2): w = max(mw, ow - delta.x())
            else:           w = max(mw, ow + delta.x())
            if c in (0, 1): h = max(mh, oh - delta.y())
            else:           h = max(mh, oh + delta.y())

            if abs(w - ow) > abs(h - oh):
                h = max(mh, w / ratio)
            else:
                w = max(mw, h * ratio)

            x, y = g.x(), g.y()
            if c in (0, 2): x = g.x() + ow - w
            if c in (0, 1): y = g.y() + oh - h

            self.window().setGeometry(int(x), int(y), int(w), int(h))
            self._resize_start = new_pos

        elif self._drag_start is not None and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.globalPosition().toPoint() - self._drag_start
            self.window().move(self.window().pos() + delta)
            self._drag_start = event.globalPosition().toPoint()

        else:
            pos = event.position()
            hover_changed = False
            if hasattr(self, '_btn_regions'):
                new_hover = -1
                for i, r in enumerate(self._btn_regions):
                    if r.contains(pos):
                        new_hover = i
                        break
                if new_hover != self._btn_hover:
                    self._btn_hover = new_hover
                    hover_changed = True
            if (hasattr(self, '_progress_rect')
                    and self._progress_rect.contains(pos)
                    and self.audio.duration() > 0):
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            elif hasattr(self, '_btn_regions') and self._btn_hover >= 0:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                c = self._corner_at(pos)
                if c in (0, 3):   self.setCursor(Qt.CursorShape.SizeFDiagCursor)
                elif c in (1, 2): self.setCursor(Qt.CursorShape.SizeBDiagCursor)
                else:             self.setCursor(Qt.CursorShape.ArrowCursor)
            if hover_changed:
                self.update()

    def mouseReleaseEvent(self, event):
        self._seeking = False
        self._resize_corner = None
        self._drag_start = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self._play_pause()
        elif event.key() == Qt.Key.Key_Right:
            self._next()
        elif event.key() == Qt.Key.Key_Left:
            self._prev()

    def closeEvent(self, event):
        self._save_state()
        self.audio.stop()
        self._anim_timer.stop()
        event.accept()


# ============================================================
#  主窗口
# ============================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(500, 320)
        self.setMinimumSize(500, 320)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        self.player = CassettePlayer()
        self.setCentralWidget(self.player)


# ============================================================
#  程序入口
# ============================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = app.palette()
    palette.setColor(palette.ColorRole.Window, QColor(10, 12, 20))
    app.setPalette(palette)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
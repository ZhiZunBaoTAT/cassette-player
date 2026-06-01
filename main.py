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
    """后台解码音频文件为 PCM，按播放位置返回 FFT 频谱柱高度。

       保持 QMediaPlayer 不变，用 pydub/ffmpeg 将同一文件解码为
       mono float32 PCM 数组。_tick() 通过 QMediaPlayer.position()
       定位采样点 → 加窗 → FFT → 对数频率映射 → 返回柱高列表。
    """

    def __init__(self, fft_size=1024, target_rate=22050):
        self._fft_size = fft_size              # FFT 窗口大小（采样点数）
        self._target_rate = target_rate         # 目标采样率（降采样节省内存）
        self._pcm = None                        # np.ndarray: mono float32
        self._sample_rate = 0                   # 实际采样率（Hz）
        self._total_samples = 0                 # PCM 总采样点数
        self._ready = False                     # 解码完成标志（线程安全）
        self._current_file = None               # 已解码的文件路径
        self._lock = threading.Lock()           # 保护状态切换
        # ── 预计算结构（延迟初始化）───────────
        self._window = None                     # Hann 窗系数
        self._bin_map = None                    # FFT bin → bar 索引
        self._bin_counts = None                 # 每根 bar 对应的 bin 数量
        self._bar_count = 0                     # 当前 bin_map 对应的柱数

    # ── 公开属性 ────────────────────────────
    @property
    def ready(self):
        """True 表示 PCM 已解码完毕，可以取频谱"""
        return self._ready

    def is_current(self, filepath):
        """已缓存的 PCM 是否对应此文件"""
        return self._current_file == filepath

    def reset(self):
        """清空 PCM 缓存，释放内存"""
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

    # ── 异步解码入口 ────────────────────────
    def load_async(self, filepath):
        """在后台线程中解码音频文件。非阻塞。"""
        if self.is_current(filepath) and self._ready:
            return  # 已缓存，无需重新解码
        self.reset()
        t = threading.Thread(target=self._load, args=(filepath,), daemon=True)
        t.start()

    # ── 实际解码（在后台线程中运行）─────────
    def _load(self, filepath):
        """pydub 解码 → mono → float32 → 存为 self._pcm"""
        try:
            audio = AudioSegment.from_file(filepath)
            # 降采样节省内存（22050Hz 对频谱可视化完全足够）
            if audio.frame_rate > self._target_rate:
                audio = audio.set_frame_rate(self._target_rate)
            audio = audio.set_channels(1)          # 单声道
            sr = audio.frame_rate
            # int16 → float32 归一化到 [-1, 1]
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            max_val = float(2 ** (audio.sample_width * 8 - 1))
            samples /= max_val
            with self._lock:
                self._pcm = samples
                self._sample_rate = sr
                self._total_samples = len(samples)
                self._current_file = filepath
                self._ready = True
                # Hann 窗（延迟初始化）
                if self._window is None or len(self._window) != self._fft_size:
                    self._window = np.hanning(self._fft_size).astype(np.float32)
        except Exception as e:
            print(f"[SpectrumDecoder] 解码失败: {filepath} — {e}")

    # ── 频谱查询（UI 线程调用）──────────────
    def get_spectrum(self, position_ms, bar_count):
        """取当前播放位置附近的 FFT 频谱，映射为 bar_count 根柱的高度 (0~1)。

           返回 list[float] 或 None（尚未就绪时）。
        """
        if not self._ready or self._pcm is None:
            return None
        if bar_count < 1:
            return None

        # 首次调用或柱数变化时重建 bin → bar 映射
        if self._bin_map is None or self._bar_count != bar_count:
            self._build_bin_map(bar_count)

        sr = self._sample_rate
        total = self._total_samples
        fft_n = self._fft_size

        # 位置 → 采样索引
        idx = int(position_ms / 1000.0 * sr)
        idx = max(0, min(idx, total - fft_n))

        # 提取窗口（末尾不足时零填充）
        if idx + fft_n <= total:
            window = self._pcm[idx:idx + fft_n] * self._window
        else:
            window = np.zeros(fft_n, dtype=np.float32)
            avail = total - idx
            window[:avail] = self._pcm[idx:total] * self._window[:avail]

        # FFT → 幅度谱
        mag = np.abs(rfft(window))
        # 按预计算映射聚合到柱
        bar_sums = np.bincount(self._bin_map, weights=mag,
                               minlength=bar_count)
        with np.errstate(divide='ignore', invalid='ignore'):
            bars = bar_sums / np.maximum(self._bin_counts, 1)
        # sqrt 压缩动态范围
        bars = np.sqrt(bars)
        # 归一化到 [0.05, 1.0]
        max_val = bars.max()
        if max_val > 0:
            bars = bars / max_val
        bars = np.clip(bars, 0.05, 1.0)
        return bars.tolist()

    def get_waveform(self, position_ms, sample_count):
        """返回当前位置附近的原始 PCM 采样点（时域波形用）。

           返回 list[float] 或 None（解码未就绪时）。
        """
        if not self._ready or self._pcm is None:
            return None
        sr = self._sample_rate
        total = self._total_samples
        idx = int(position_ms / 1000.0 * sr)
        start = max(0, idx - sample_count // 2)
        end = min(total, start + sample_count)
        chunk = self._pcm[start:end]
        # 不足时零填充
        if len(chunk) < sample_count:
            padded = np.zeros(sample_count, dtype=np.float32)
            padded[:len(chunk)] = chunk
            return padded.tolist()
        return chunk.tolist()

    # ── 预计算 FFT bin → bar 映射（对数频率）─
    def _build_bin_map(self, bar_count):
        """按对数频率刻度将 FFT bin 分配到 bar。

           频率范围 30Hz ~ min(Nyquist, 10kHz)，低频柱分到更多 bin，
           更符合人耳对低频的感知。
        """
        sr = self._sample_rate
        fft_n = self._fft_size
        freqs = rfftfreq(fft_n, 1.0 / sr)          # 每个正频率 bin 的中心频率
        n_bins = len(freqs)

        min_freq = 30.0
        max_freq = min(sr / 2.0, 10000.0)
        # 对数等间距切分 bar_count 段
        log_min = np.log10(min_freq)
        log_max = np.log10(max_freq)
        edges = np.logspace(log_min, log_max, bar_count + 1)

        bin_map = np.zeros(n_bins, dtype=np.int32)
        bin_counts = np.zeros(bar_count, dtype=np.int32)
        for bi in range(n_bins):
            f = freqs[bi]
            # 二分查找 f 落在哪个柱区间
            bar_idx = np.searchsorted(edges, f) - 1
            bar_idx = max(0, min(bar_idx, bar_count - 1))
            bin_map[bi] = bar_idx
            bin_counts[bar_idx] += 1
        # 确保每柱至少 1 个 bin（避免除零）
        bin_counts = np.maximum(bin_counts, 1)

        self._bin_map = bin_map
        self._bin_counts = bin_counts
        self._bar_count = bar_count


# ============================================================
#  音频引擎 — 负责所有音频播放逻辑（独立于 UI）
# ============================================================

class AudioEngine:
    """管理播放列表、播放控制、元数据读取"""

    def __init__(self, parent=None):
        # ── Qt 多媒体核心：播放器 + 音频输出 ──
        self._player = QMediaPlayer(parent)      # 媒体播放器实例
        self._audio = QAudioOutput(parent)       # 音频输出设备
        self._player.setAudioOutput(self._audio) # 绑定输出
        self._audio.setVolume(0.8)               # 默认音量 80%

        # ── 频谱解码器（后台 PCM 解码 + FFT）──
        self._decoder = SpectrumDecoder(fft_size=1024, target_rate=22050)

        # ── 播放列表状态 ──
        self._playlist = []   # 歌曲路径列表
        self._index = -1      # 当前播放索引（-1 = 无）
        self._playing = False # 是否正在播放

        # ── 监听播放状态变化（Qt 信号 → 本地回调）──
        self._player.playbackStateChanged.connect(self._on_state_change)

    def _on_state_change(self, state):
        """Qt 播放状态变化时更新内部标志"""
        self._playing = (state == QMediaPlayer.PlaybackState.PlayingState)

    # ── 频谱代理（委托给 SpectrumDecoder）─────
    @property
    def decoder_ready(self):
        """PCM 解码是否已完成"""
        return self._decoder.ready

    def get_spectrum(self, position_ms, bar_count):
        """获取当前播放位置的 FFT 频谱柱高列表"""
        return self._decoder.get_spectrum(position_ms, bar_count)

    def _start_decode(self, filepath):
        """后台异步解码音频文件为 PCM（非阻塞）"""
        if not self._decoder.is_current(filepath):
            self._decoder.load_async(filepath)

    def cleanup(self):
        """释放 PCM 缓存内存"""
        self._decoder.reset()

    # ── 属性（只读）────────────────────────────
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
        """当前播放位置（毫秒）"""
        return self._player.position()

    def duration(self):
        """当前歌曲总时长（毫秒）"""
        return self._player.duration()

    def seek(self, ms):
        """跳转到指定毫秒位置"""
        self._player.setPosition(ms)

    # ── 播放列表管理 ───────────────────────────
    def load_folder(self, folder_path):
        """递归扫描文件夹，收集支持的音频文件"""
        extensions = {'.mp3', '.flac', '.wav', '.ogg', '.m4a', '.aac'}
        self._playlist = []
        for ext in extensions:
            for f in Path(folder_path).rglob(f'*{ext}'):  # 递归匹配
                self._playlist.append(str(f))
        self._playlist.sort()  # 按路径排序
        return len(self._playlist)

    def play_index(self, index):
        """播放指定索引的歌曲"""
        if 0 <= index < len(self._playlist):
            path = self._playlist[index]
            # QUrl.fromLocalFile 将本地路径转为 Qt 可识别的 URL
            self._player.setSource(QUrl.fromLocalFile(path))
            self._player.play()
            self._playing = True
            self._index = index
            self._start_decode(path)             # 后台解码 PCM → FFT 频谱用
            return True
        return False

    # ── 播放控制 ───────────────────────────────
    def toggle(self):
        """播放 / 暂停切换"""
        if self._playing:
            self._player.pause()
        else:
            self._player.play()

    def stop(self):
        """停止播放"""
        self._player.stop()
        self._playing = False

    def next(self):
        """下一首（循环到列表头）"""
        if self._playlist:
            nxt = (self._index + 1) % len(self._playlist)
            return self.play_index(nxt)
        return False

    def prev(self):
        """上一首（循环到列表尾）"""
        if self._playlist:
            prv = (self._index - 1) % len(self._playlist)
            return self.play_index(prv)
        return False

    # ── 元数据 ─────────────────────────────────
    @staticmethod
    def get_metadata(filepath):
        """读取歌曲的标题（TIT2）和艺术家（TPE1）标签"""
        try:
            if filepath.endswith('.mp3'):
                audio = MP3(filepath)     # mutagen 解析 MP3
                tags = audio.tags          # ID3 标签字典
                if tags:
                    title = str(tags.get('TIT2', Path(filepath).stem))
                    artist = str(tags.get('TPE1', 'Unknown'))
                    return {'title': title, 'artist': artist, 'path': filepath}
            # 非 MP3 或无标签时用文件名作为标题
            return {'title': Path(filepath).stem, 'artist': 'Unknown', 'path': filepath}
        except Exception:
            return {'title': Path(filepath).stem, 'artist': 'Unknown', 'path': filepath}


# ============================================================
#  磁带播放器主控件 — 所有 UI 绘制 & 交互逻辑
# ============================================================

class CassettePlayer(QWidget):
    """磁带风格音乐播放器控件（继承 QWidget）"""

    # ── 构造 & 初始化 ──────────────────────────
    def __init__(self):
        super().__init__()
        self.audio = AudioEngine(self)          # 音频引擎实例
        self.rotation_angle = 0.0               # 磁带轮旋转角度（度）
        self._settings = QSettings("CassettePlayer", "CassettePlayer")  # 持久化存储

        # ── 可视化风格（0-5莫兰迪 6-11彩虹，%6得样式 //6得配色）──
        self._viz_style = int(self._settings.value("viz_style", 0))

        # ── 频谱柱数据（预分配容量，实际数量由布局决定）──
        self._bar_count = 60                    # 柱子上限
        self._bars = [0.05] * self._bar_count   # 当前高度（0~1）
        self._bar_targets = [0.05] * self._bar_count  # 目标高度
        self._particles = []                    # 粒子状态（风格 4 延迟初始化）
        self._bar_frame = 0                     # 动画帧计数
        self._hue_offset = 0.0                  # 色相偏移（流动彩虹）
        self._drag_start = None                 # 拖拽起始坐标
        self._seeking = False                  # 是否正在拖拽进度条

        # ── 动画定时器：每 30ms 触发 _tick，约 33fps ──
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._tick)
        self._anim_timer.start(30)

        # ── 歌曲信息（由 paintEvent 绘制到标签区）──
        self._track_title = "未播放"
        self._track_artist = "请打开音乐文件夹"

        # ── 初始化 UI 并恢复上次播放状态 ──
        self._setup_ui()
        self._restore_state()

    # ── UI 初始化（只调用一次）──────────────────
    def _setup_ui(self):
        self.setMinimumSize(500, 320)           # 窗口最小尺寸（磁带比例）
        self.setMouseTracking(True)             # 启用鼠标追踪（悬停光标变化）
        self.setStyleSheet("background: transparent;")  # 透明背景

        # ── 三个控制按钮（QPushButton）──────────
        # 按钮由 paintEvent 手绘（避免原生样式干扰）
        self._btn_play_text = "▶"         # 播放按钮文字（▶ / ⏸）
        self._btn_regions = []            # [(rect, action), ...] → 点击检测
        self._btn_hover = -1              # 当前悬停按钮索引（-1 = 无）
        self._file_list = []

    # ── 窗口缩放时重新布局 ─────────────────────
    def resizeEvent(self, event):
        """窗口缩放时触发重绘"""
        super().resizeEvent(event)

    # ── 磁带轮中心 Y 坐标计算 ──────────────────
    def _reel_center_y(self):
        """动态计算：在标签区和进度条+音浪区之间居中"""
        w = self.width()
        s = w / 680
        margin = int(18 * s)
        label_y = margin + int(8 * s)
        label_h = int(68 * s)
        waveform_max_h = int(78 * s)
        waveform_base_offset = int(6 * s)
        progress_space = int(14 * s)     # 进度条高度 + 间距
        cassette_bottom = self.height() - margin
        # reel 下方的可用空间顶部
        content_bottom = cassette_bottom - waveform_base_offset - waveform_max_h - progress_space
        return label_y + label_h + (content_bottom - label_y - label_h) // 2

    # ================================================================
    #  动画循环 — 每 30ms 执行一次
    # ================================================================

    def _tick(self):
        """更新磁带轮角度 + 频谱柱高度 + 色相偏移"""
        if self.audio.playing:
            # ── 播放中：磁带轮旋转 + 频谱跳动 ──
            self.rotation_angle += 3.0         # 每帧旋转 3°（约 100°/秒）
            self._bar_frame += 1               # 帧计数器递增

            # 优先使用真实 FFT 频谱，解码未完成时回退到随机
            if self.audio.decoder_ready:
                spectrum = self.audio.get_spectrum(
                    self.audio.position(), self._bar_count)
                if spectrum is not None:
                    for i in range(self._bar_count):
                        self._bar_targets[i] = spectrum[i]
                else:
                    self._random_bars()
            else:
                # 后台解码中（通常 < 1 秒），沿用旧随机逻辑
                if self._bar_frame % 4 == 0:
                    self._random_bars()
        else:
            # ── 暂停中：所有柱子缓慢衰减到接近 0 ──
            for i in range(self._bar_count):
                self._bar_targets[i] = 0.05

        # ── 平滑插值（lerp）：当前值 → 目标值，每帧移动 18% ──
        for i in range(self._bar_count):
            self._bars[i] += (self._bar_targets[i] - self._bars[i]) * 0.18

        # ── 色相偏移：每帧 +0.003，循环 0~1，实现彩虹流动 ──
        self._hue_offset = (self._hue_offset + 0.003) % 1.0

        self.update()  # 触发 paintEvent 重绘

    def _random_bars(self):
        """随机柱高（解码未完成时的回退方案）"""
        n = self._bar_count
        for i in range(0, n, 3):
            self._bar_targets[i] = random.uniform(0.2, 1.0)
            self._bar_targets[min(i + 1, n - 1)] = random.uniform(0.12, 0.65)
            self._bar_targets[min(i + 2, n - 1)] = random.uniform(0.05, 0.35)

    # ================================================================
    #  绘制 — paintEvent 在 update() 或窗口变化时自动调用
    # ================================================================

    def paintEvent(self, event):
        """绘制整个磁带 UI：机身 → 标签 → 磁带轮 → 螺丝 → 频谱"""
        p = QPainter(self)                            # 创建画笔
        p.setRenderHint(QPainter.RenderHint.Antialiasing)  # 抗锯齿

        w, h = self.width(), self.height()            # 当前控件尺寸

        # ── 缩放参数（以 680px 宽为基准）─────────
        base_w = 680
        s = w / base_w                               # 缩放比例
        margin = int(18 * s)                         # 机身外边距
        cassette_bottom = h - margin                 # 机身底部 Y
        bw = w - margin * 2                          # 机身宽度（扣除边距）
        bh = h - margin * 2                          # 机身高度（扣除边距）

        # ── 玻璃磁带主体（圆角矩形 + 半透明填充）──
        path = QPainterPath()                        # 创建矢量路径
        path.addRoundedRect(QRectF(margin, margin, bw, bh), 22, 22)  # 圆角半径 22
        p.fillPath(path, QColor(55, 60, 72, 110))    # 深灰蓝半透明填充
        p.setPen(QPen(QColor(170, 180, 200, 150), 2)) # 浅灰边框 2px
        p.drawPath(path)                             # 绘制路径

        # ── 内部发光（比机身小 3px 的亮框）───────
        path2 = QPainterPath()
        path2.addRoundedRect(QRectF(margin + 3, margin + 3, bw - 6, bh - 6), 20, 20)
        p.setPen(QPen(QColor(255, 255, 255, 30), 1)) # 极淡白色
        p.drawPath(path2)

        # ── 标签区（梯形 + 凸起效果）─────────────
        label_y = margin + int(10 * s)               # 标签顶部 Y
        label_h = int(64 * s)                        # 标签高度
        slant = int(10 * s)                          # 梯形内收量（上宽下窄）
        tl_x = margin + int(26 * s)                  # 标签左上 X
        tr_x = w - margin - int(26 * s)              # 标签右上 X
        bl_x = tl_x + slant                          # 标签左下 X（内收）
        br_x = tr_x - slant                          # 标签右下 X（内收）
        top_y = label_y
        bottom_y = label_y + label_h
        cr = int(8 * s)                              # 圆角半径

        # 局部函数：构建圆角梯形矢量路径
        def _rounded_trapezoid(tlx, trx, blx, brx, ty, by, radius):
            """返回 QPainterPath：上宽下窄的圆角梯形。
               四角用 arcTo 画 90° 圆弧，边用 lineTo 连直线。"""
            path = QPainterPath()
            # 起点：左上角弧线结束处
            path.moveTo(tlx + radius, ty)
            # 顶边 →
            path.lineTo(trx - radius, ty)
            # 右上角弧（从 90° 逆时针转 90° → 0°）
            path.arcTo(trx - 2 * radius, ty, 2 * radius, 2 * radius, 90, -90)
            # 右边 ↙（斜向内收）
            path.lineTo(brx, by - radius)
            # 右下角弧（0° → 270°）
            path.arcTo(brx - 2 * radius, by - 2 * radius, 2 * radius, 2 * radius, 0, -90)
            # 底边 ←
            path.lineTo(blx + radius, by)
            # 左下角弧（270° → 180°）
            path.arcTo(blx, by - 2 * radius, 2 * radius, 2 * radius, 270, -90)
            # 左边 ↗（斜向外扩）
            path.lineTo(tlx, ty + radius)
            # 左上角弧（180° → 90°）
            path.arcTo(tlx, ty, 2 * radius, 2 * radius, 180, -90)
            path.closeSubpath()  # 闭合路径
            return path

        # ① 标签底部阴影（向下偏移 3~4px）
        shadow_path = _rounded_trapezoid(tl_x + int(2*s), tr_x - int(2*s), bl_x, br_x,
                                         top_y + int(3*s), bottom_y + int(4*s), cr)
        p.fillPath(shadow_path, QColor(0, 0, 0, 40))  # 40/255 透明黑

        # ② 标签主体（暖棕色半透明）
        label_path = _rounded_trapezoid(tl_x, tr_x, bl_x, br_x, top_y, bottom_y, cr)
        p.fillPath(label_path, QColor(72, 64, 50, 160))
        p.setPen(QPen(QColor(180, 170, 140, 90), 1))
        p.drawPath(label_path)

        # ③ 顶部高光边（窄梯形，模拟光打在凸起边缘）
        hl_path = _rounded_trapezoid(tl_x + int(2*s), tr_x - int(2*s),
                                     bl_x + int(4*s), br_x - int(4*s),
                                     top_y + int(1*s), top_y + int(8*s), int(5*s))
        p.fillPath(hl_path, QColor(255, 255, 255, 35))

        # ④ 全局玻璃反光（与梯形精确重合的渐变）
        # QLinearGradient(x1, y1, x2, y2)：从 (x1,y1) 到 (x2,y2) 的线性渐变
        grad = QLinearGradient(0, top_y, 0, bottom_y)  # 垂直渐变
        grad.setColorAt(0, QColor(255, 255, 255, 100))   # 顶部亮白
        grad.setColorAt(0.3, QColor(255, 255, 255, 30))  # 30% 处骤减
        grad.setColorAt(1, QColor(255, 255, 255, 0))     # 底部全透
        hl_global = _rounded_trapezoid(tl_x, tr_x, bl_x, br_x, top_y, bottom_y, cr)
        p.fillPath(hl_global, grad)

        # ⑤ 标签装饰横线（两条淡色线，矢量缩放）
        p.setPen(QPen(QColor(200, 190, 160, 50), max(1, int(1 * s))))
        for i in range(2):
            ly = top_y + int(22 * s) + i * int(20 * s)
            p.drawLine(int(bl_x + int(16 * s)), ly, int(br_x - int(16 * s)), ly)

        # ⑥ 绘制歌曲信息文字（融入标签区）
        font_s = max(12, int(16 * s))              # 歌名字体大小
        artist_font_s = max(10, int(13 * s))        # 艺术家字体大小
        label_cx = (tl_x + tr_x) / 2               # 标签水平中心

        # ── 歌名 ──
        title_font = QFont("Microsoft YaHei", font_s)
        title_font.setBold(True)
        p.setFont(title_font)
        p.setPen(QColor(240, 235, 220, 220))       # 暖白色
        title_rect = QRectF(tl_x + 10, top_y + int(3 * s) + 4, tr_x - tl_x - 20, 24 * s)
        p.drawText(title_rect, Qt.AlignmentFlag.AlignCenter, self._track_title)

        # ── 艺术家 ──
        artist_font = QFont("Microsoft YaHei", artist_font_s)
        p.setFont(artist_font)
        p.setPen(QColor(200, 190, 170, 180))       # 淡暖色
        artist_rect = QRectF(tl_x + 10, top_y + int(3 * s) + 28 * s, tr_x - tl_x - 20, 20 * s)
        p.drawText(artist_rect, Qt.AlignmentFlag.AlignCenter, self._track_artist)

        # ── 磁带轮（左右两个旋转轮盘）───────────
        reel_r = int(44 * s)                       # 轮盘半径
        reel_y = self._reel_center_y()              # 轮盘中心 Y
        reel_spacing = int(170 * s)                 # 两轮中心间距
        r1_x = w // 2 - reel_spacing               # 左轮 X
        r2_x = w // 2 + reel_spacing               # 右轮 X
        for cx in [r1_x, r2_x]:
            self._draw_reel(p, cx, reel_y, reel_r)

        # ── 手绘控制按钮（⏮ ▶/⏸ ⏭）──────────────
        btn_s = int(44 * s)
        btn_spacing = int(reel_spacing * 0.50)
        center_x = w // 2
        btn_y = reel_y - btn_s // 2
        btn_font = QFont("Segoe UI Symbol", max(10, int(20 * s)))
        p.setFont(btn_font)

        # 三组按钮位置 (x, y, w, symbol)
        btns = [
            (center_x - btn_spacing - btn_s // 2, btn_y, btn_s, "⏮"),   # 上一首
            (center_x - btn_s // 2, btn_y, btn_s, self._btn_play_text),  # 播放/暂停
            (center_x + btn_spacing - btn_s // 2, btn_y, btn_s, "⏭"),   # 下一首
        ]
        self._btn_regions = []  # 重建点击区域
        for i, (bx, by_, bs, sym) in enumerate(btns):
            rect = QRectF(bx, by_, bs, bs)
            self._btn_regions.append(rect)
            # 悬停高亮
            hovered = (self._btn_hover == i)
            if hovered:
                p.setPen(QColor(255, 255, 255, 255))
            else:
                p.setPen(QColor(180, 180, 180, 180))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, sym)

        # ── 四角螺丝 ────────────────────────────
        screw_r = int(7 * s)                       # 螺丝外圈半径
        top_off = int(18 * s)                      # 上方螺丝距边缘（不挡标签）
        bot_off = int(26 * s)                      # 下方螺丝距边缘（内移）
        screw_positions = [
            (margin + top_off, margin + top_off + int(7 * s)),               # 左上：+
            (w - margin - top_off, margin + top_off + int(7 * s)),           # 右上：✕
            (margin + bot_off + int(2 * s), cassette_bottom - bot_off + int(2 * s)), # 左下：V
            (w - margin - bot_off - int(2 * s), cassette_bottom - bot_off + int(2 * s)), # 右下：一字槽
        ]
        self._screw_positions = screw_positions     # 存储 → 供点击检测

        for idx, (sx, sy) in enumerate(screw_positions):
            # 外圈（浅银灰）
            p.setPen(Qt.PenStyle.NoPen)            # 无边框
            p.setBrush(QColor(165, 170, 180, 170))  # 画刷填充
            p.drawEllipse(QPointF(sx, sy), screw_r, screw_r)
            # 内圈（稍暗）
            p.setBrush(QColor(135, 140, 150, 190))
            p.drawEllipse(QPointF(sx, sy), screw_r - int(3 * s), screw_r - int(3 * s))

            lw = int(2 * s)                        # 线条宽度
            if idx == 0:                           # 左上：十字 +
                p.setPen(QPen(QColor(220, 225, 235, 200), max(1, lw)))
                d = int(3 * s)
                p.drawLine(int(sx - d), int(sy), int(sx + d), int(sy))   # 横线
                p.drawLine(int(sx), int(sy - d), int(sx), int(sy + d))   # 竖线
            elif idx == 1:                         # 右上：叉号 ✕
                p.setPen(QPen(QColor(220, 225, 235, 200), max(1, lw)))
                d = int(2 * s)
                p.drawLine(int(sx - d), int(sy - d), int(sx + d), int(sy + d))  # 对角线
                p.drawLine(int(sx + d), int(sy - d), int(sx - d), int(sy + d))  # 反对角线
            elif idx == 2:                         # 左下：V 字形（visualization）
                p.setPen(QPen(QColor(255, 90, 100, 200), max(1, lw)))
                d = int(3 * s)
                p.drawLine(int(sx - d), int(sy - d), int(sx), int(sy + d))
                p.drawLine(int(sx + d), int(sy - d), int(sx), int(sy + d))
            else:                                  # 右下：一字槽（暗色）
                p.setPen(QPen(QColor(100, 105, 115, 150), 1))
                p.drawLine(int(sx - d), int(sy), int(sx + d), int(sy))
                p.drawLine(int(sx), int(sy - d), int(sx), int(sy + d))

        # ── 播放进度条（音浪上方）─────────────────
        # 先算音浪水平跨度 & 垂直参数（进度条 & 音浪共用）
        wave_start_x = (w // 2 - reel_spacing) - reel_r
        wave_end_x = (w // 2 + reel_spacing) + reel_r
        wave_total_w = wave_end_x - wave_start_x

        waveform_max_h = int(78 * s)
        waveform_base_offset = int(6 * s)

        dur = self.audio.duration()
        pos_ms = self.audio.position()

        # ── 进度条 Y 坐标（音浪上方留出间隙）────
        progress_y = cassette_bottom - waveform_base_offset - waveform_max_h - int(24 * s)

        # ── 时间标签 ────────────────────────────
        def _fmt(ms):
            """毫秒 → MM:SS"""
            if ms <= 0:
                return "00:00"
            sec = ms // 1000
            return f"{sec // 60:02d}:{sec % 60:02d}"

        time_font = QFont("Consolas", max(9, int(12 * s)))
        p.setFont(time_font)
        p.setPen(QColor(200, 200, 200, 160))
        elapsed_text = _fmt(pos_ms)
        remain_text = "-" + _fmt(max(0, dur - pos_ms))
        # 左侧已播放时间
        p.drawText(QRectF(wave_start_x - int(70 * s), progress_y - int(12 * s),
                          int(65 * s), int(20 * s)),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   elapsed_text)
        # 右侧剩余时间
        p.drawText(QRectF(wave_end_x + int(5 * s), progress_y - int(12 * s),
                          int(65 * s), int(20 * s)),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   remain_text)

        # ── 进度条轨道 ──────────────────────────
        progress_h = int(4 * s)
        progress_rect = QRectF(wave_start_x, progress_y, wave_total_w, progress_h)
        self._progress_rect = progress_rect

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 30))
        p.drawRoundedRect(progress_rect, 2, 2)

        # 已播放部分
        if dur > 0:
            frac = min(pos_ms / dur, 1.0)
            filled_w = int(progress_rect.width() * frac)
            if filled_w > 0:
                filled_rect = QRectF(progress_rect.x(), progress_rect.y(),
                                     filled_w, progress_rect.height())
                p.setBrush(QColor(220, 200, 150, 180))
                p.drawRoundedRect(filled_rect, 2, 2)

        # 进度滑块（爱心符号 ❤）
        if dur > 0 and pos_ms >= 0:
            dot_x = progress_rect.x() + progress_rect.width() * frac
            dot_y = progress_rect.center().y()
            heart_font = QFont("Segoe UI Emoji", max(16, int(24 * s)))
            p.setFont(heart_font)
            p.setPen(QColor(255, 100, 130, 240))
            p.drawText(QRectF(dot_x - int(20 * s), dot_y - int(23 * s),
                              int(40 * s), int(40 * s)),
                       Qt.AlignmentFlag.AlignCenter, "❤")

        # ── 频谱音浪（磁带机身内部底部）───────────
        base_y = cassette_bottom - waveform_base_offset
        max_bar_h = waveform_max_h
        self._draw_spectrum(p, wave_start_x, wave_end_x, base_y, max_bar_h, s)

    # ================================================================
    #  频谱可视化 — 6 种风格（由 _viz_style 选择）
    # ================================================================

    def _morandi_color(self, i, t, bar_count):
        """莫兰迪调色板：低饱和高级灰，亮度随柱高变化"""
        palette = [
            (185, 150, 145),  # 灰粉
            (200, 180, 165),  # 灰杏
            (155, 170, 150),  # 灰绿
            (145, 155, 175),  # 灰蓝
            (170, 165, 180),  # 灰紫
            (180, 160, 155),  # 灰玫
            (160, 170, 170),  # 灰青
            (190, 175, 160),  # 灰驼
        ]
        # 按位置循环取色，加上时间偏移使颜色缓慢流动
        idx = int((i / bar_count * len(palette) + self._hue_offset * len(palette)) % len(palette))
        br, bg, bb = palette[idx]
        # 亮度随柱高增强
        scale = 0.6 + t * 0.4
        r = min(255, int(br * scale))
        g = min(255, int(bg * scale))
        b = min(255, int(bb * scale))
        return QColor(r, g, b)

    def _rainbow_color(self, i, t, bar_count):
        """HSV 彩虹：色相按位置+时间偏移流动"""
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
        r = int((rf + cm) * 255)
        g = int((gf + cm) * 255)
        b = int((bf + cm) * 255)
        return QColor(r, g, b)

    def _bar_color(self, i, t, bar_count):
        """根据当前配色方案分发"""
        if self._viz_style < 6:
            return self._morandi_color(i, t, bar_count)
        else:
            return self._rainbow_color(i, t, bar_count)

    def _draw_spectrum(self, p, sx, ex, base_y, max_h, s):
        """根据 _viz_style 分发到对应绘制方法"""
        total_w = ex - sx
        bar_count = min(60, len(self._bars))
        self._bar_count = bar_count

        # ── 清除上一帧残留 ────────────────────
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(55, 60, 72, 110))
        p.drawRect(QRectF(sx, base_y - max_h, total_w, max_h))

        styles = [
            self._draw_bars,
            self._draw_radar,
            self._draw_waveform,
            self._draw_mirror,
            self._draw_particles,
            self._draw_pulse,
        ]
        styles[self._viz_style % 6](p, sx, ex, base_y, max_h, s, bar_count, total_w)

    # ── 风格 0：柱状频谱（经典）─────────────
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
            # 四层炫光
            p.setBrush(QColor(c.red(), c.green(), c.blue(), 30))
            p.drawRoundedRect(QRectF(bx - 3, by - 6, bar_w + 6, bh + 10), 6, 6)
            p.setBrush(QColor(c.red(), c.green(), c.blue(), 80))
            p.drawRoundedRect(QRectF(bx - 1, by - 3, bar_w + 2, bh + 5), 4, 4)
            p.setBrush(QColor(c.red(), c.green(), c.blue(), 240))
            p.drawRoundedRect(QRectF(bx, by, bar_w, bh), 2, 2)
            p.setBrush(QColor(min(c.red() + 80, 255),
                              min(c.green() + 80, 255),
                              min(c.blue() + 80, 255), 200))
            p.drawRoundedRect(QRectF(bx, by, bar_w, max(3, int(bh * 0.25))), 2, 2)

    # ── 风格 1：圆形雷达 ─────────────────────
    def _draw_radar(self, p, sx, ex, base_y, max_h, s, n, tw):
        cx = (sx + ex) / 2
        cy = base_y - max_h / 2 - int(10 * s) # 上提，避免溢出底部
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
            # 光晕
            p.setBrush(QColor(c.red(), c.green(), c.blue(), 50))
            p.drawEllipse(QPointF(ex2, ey2), bw + 4, bw + 4)
            # 主体线
            pen = QPen(QColor(c.red(), c.green(), c.blue(), 230), bw)
            p.setPen(pen)
            p.drawLine(QPointF(bx, by), QPointF(ex2, ey2))
            p.setPen(Qt.PenStyle.NoPen)
        # 中心点
        p.setBrush(QColor(220, 220, 240, 200))
        p.drawEllipse(QPointF(cx, cy), 1, 1)

    # ── 风格 2：波形曲线（炫彩时域）───────────
    def _draw_waveform(self, p, sx, ex, base_y, max_h, s, n, tw):
        cy = base_y - max_h / 2
        half_h = max_h / 2
        sample_count = max(200, int(tw))
        samples = self.audio._decoder.get_waveform(
            self.audio.position(), sample_count)
        if samples is None or len(samples) < 2:
            return
        pts = []
        for i, v in enumerate(samples[:sample_count]):
            x = sx + i * tw / (sample_count - 1)
            y = cy - v * half_h * 0.9
            pts.append(QPointF(x, y))
        # 波形填充（半透明彩虹底）
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
        # 描边：每段使用当前配色
        seg_count = len(pts) - 1
        for i in range(seg_count):
            seg_c = self._bar_color(i, 0.7, seg_count)
            pen = QPen(QColor(seg_c.red(), seg_c.green(), seg_c.blue(), 220),
                       max(1.5, 2.5 * s))
            p.setPen(pen)
            p.drawLine(pts[i], pts[i + 1])

    # ── 风格 3：镜像对称柱状 ─────────────────
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
            # 上半
            p.setBrush(QColor(c.red(), c.green(), c.blue(), 200))
            p.drawRoundedRect(QRectF(bx, cy - bh, bar_w, bh), 1, 1)
            p.setBrush(QColor(min(c.red() + 60, 255),
                              min(c.green() + 60, 255),
                              min(c.blue() + 60, 255), 140))
            p.drawRoundedRect(QRectF(bx, cy - bh, bar_w, max(2 * s, int(bh * 0.3))), 1, 1)
            # 下半（镜像）
            p.setBrush(QColor(c.red(), c.green(), c.blue(), 120))
            p.drawRoundedRect(QRectF(bx, cy, bar_w, bh), 1, 1)
        # 中缝横线
        p.setPen(QPen(QColor(255, 255, 255, 40), 1))
        p.drawLine(QPointF(sx, cy), QPointF(ex, cy))

    # ── 风格 4：粒子漂浮 ─────────────────────
    def _draw_particles(self, p, sx, ex, base_y, max_h, s, n, tw):
        cy = base_y - max_h / 2
        half_h = max_h / 2 - int(6 * s)
        # 窗口缩放或首次初始化时重建粒子
        if (not hasattr(self, '_particles') or len(self._particles) != 40
                or getattr(self, '_particles_tw', 0) != tw):
            self._particles = []
            self._particles_tw = tw
            import random as _rnd
            for _ in range(40):
                self._particles.append({
                    'frac': _rnd.random(),          # 用比例代替绝对坐标
                    'y': cy,
                    'target_y': cy,
                    'size': _rnd.uniform(2, 5) * s,
                    'hue': _rnd.random(),
                })
        p.setPen(Qt.PenStyle.NoPen)
        for pi, pt in enumerate(self._particles):
            # 根据 frac 还原当前 x 坐标
            px = sx + pt['frac'] * tw
            bi = max(0, min(n - 1, int(pt['frac'] * n)))
            t = self._bars[bi]
            pt['target_y'] = cy - t * half_h * (1.0 if (pi % 2) else -1.0)
            pt['y'] += (pt['target_y'] - pt['y']) * 0.12
            pt['hue'] = (pt['hue'] + random.uniform(0, 0.01)) % 1.0
            pt['size'] = max(2 * s, (3 + t * 4) * s)
            # 光晕
            c = self._bar_color(bi, t, n)
            sz = pt['size']
            p.setBrush(QColor(c.red(), c.green(), c.blue(), 40))
            p.drawEllipse(QPointF(px, pt['y']), sz + 2 * s, sz + 2 * s)
            p.setBrush(QColor(c.red(), c.green(), c.blue(), 220))
            p.drawEllipse(QPointF(px, pt['y']), sz, sz)

    # ── 风格 5：圆环脉冲 ─────────────────────
    def _draw_pulse(self, p, sx, ex, base_y, max_h, s, n, tw):
        cx = (sx + ex) / 2
        cy = base_y - max_h / 2 - int(10 * s) # 上提，与雷达同步
        max_r = min(tw, max_h) / 2 + int(2 * s)
        # 低频/中频/高频能量
        bass = sum(self._bars[:10]) / 10
        mid = sum(self._bars[10:30]) / 20
        high = sum(self._bars[30:]) / max(1, n - 30)
        rings = [
            (0.25 * max_r, bass, self._bar_color(0, 0.8, n)),     # 内圈
            (0.55 * max_r, mid, self._bar_color(n // 3, 0.7, n)),  # 中圈
            (0.85 * max_r, high, self._bar_color(n * 2 // 3, 0.6, n)),  # 外圈
        ]
        p.setPen(Qt.PenStyle.NoPen)
        for base_r, energy, color in rings:
            r = max(6, int(base_r + energy * max_r * 0.35))
            # 外光晕
            p.setBrush(QColor(color.red(), color.green(), color.blue(), 25))
            p.drawEllipse(QPointF(cx, cy), r + 10, r + 10)
            # 主体圆环
            pen = QPen(QColor(color.red(), color.green(), color.blue(),
                              max(60, color.alpha())), max(2.0, 3.0 * s))
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(cx, cy), r, r)
            p.setPen(Qt.PenStyle.NoPen)
        # 中心脉冲点
        dot_c = self._bar_color(0, bass, 1)
        p.setBrush(QColor(dot_c.red(), dot_c.green(), dot_c.blue(),
                          int(120 + bass * 135)))
        p.drawEllipse(QPointF(cx, cy), max(1, int(2 + bass * 4)),
                      max(1, int(2 + bass * 4)))

    # ================================================================
    #  磁带轮绘制
    # ================================================================

    def _draw_reel(self, p, cx, cy, r):
        """绘制一个磁带轮（含旋转齿轮和中心轴）
           p.save/translate/restore：临时移动坐标系到 (cx, cy)，画完后恢复"""
        p.save()                                   # 保存当前画笔状态
        p.translate(cx, cy)                        # 移动坐标系原点到此轮中心

        # ① 外圈光环（浅色细环）
        p.setPen(QPen(QColor(140, 150, 170, 60), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)          # 不填充
        p.drawEllipse(QPointF(0, 0), r, r)

        # ② 轮体（深色圆盘）
        p.setPen(QPen(QColor(120, 130, 150, 140), 2))
        p.setBrush(QColor(25, 28, 35, 140))
        p.drawEllipse(QPointF(0, 0), r - 2, r - 2)

        # ③ 5 个旋转齿轮（莫兰迪配色）
        angle_rad = math.radians(self.rotation_angle)
        morandi = [
            QColor(185, 150, 145),   # 灰粉
            QColor(155, 170, 150),   # 灰绿
            QColor(145, 155, 175),   # 灰蓝
            QColor(190, 175, 150),   # 灰杏
            QColor(170, 160, 180),   # 灰紫
        ]
        for i in range(5):
            a = math.radians(i * 72) + angle_rad
            gx = math.cos(a) * (r - 16)
            gy = math.sin(a) * (r - 16)
            c = morandi[i]
            # 阴影
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(c.red(), c.green(), c.blue(), 80))
            p.drawEllipse(QPointF(gx + 1, gy + 1), 6, 6)
            # 主体
            p.setBrush(QColor(c.red(), c.green(), c.blue(), 180))
            p.drawEllipse(QPointF(gx, gy), 6, 6)
            # 高光
            p.setBrush(QColor(min(c.red() + 40, 255),
                              min(c.green() + 40, 255),
                              min(c.blue() + 40, 255), 120))
            p.drawEllipse(QPointF(gx - 1, gy - 2), 3, 3)

        # ④ 内环（装饰圈）
        p.setPen(QPen(QColor(90, 100, 120, 100), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(0, 0), r - 24, r - 24)

        # ⑤ 中心轴（大圆 + 小亮点）
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(75, 80, 95, 160))
        p.drawEllipse(QPointF(0, 0), 10, 10)
        p.setBrush(QColor(140, 145, 160, 200))
        p.drawEllipse(QPointF(0, 0), 4, 4)

        p.restore()                                # 恢复坐标系

    # ================================================================
    #  操作 — 播放控制 & 状态管理
    # ================================================================

    def _open_folder(self):
        """打开文件夹对话框，加载音乐并播放第一首"""
        # QFileDialog.getExistingDirectory：系统原生文件夹选择对话框
        folder = QFileDialog.getExistingDirectory(self, "选择音乐文件夹")
        if folder:
            count = self.audio.load_folder(folder)
            if count > 0:
                self._file_list = self.audio.playlist  # 缓存播放列表引用
                self.audio.play_index(0)                # 播放第一首
                self._update_track_info()               # 更新标签文字
                self._btn_play_text = "⏸"
                self.update()
            else:
                self._track_title = "未找到音乐文件"
                self._track_artist = folder

    def _play_pause(self):
        """播放/暂停切换。若无播放列表则先打开文件夹"""
        if not self.audio.playlist:
            self._open_folder()                      # 空列表 → 提示选文件夹
            return
        if not self.audio.playing and self.audio.current_index < 0:
            self.audio.play_index(0)                 # 从未播放过 → 播放第一首
            self._update_track_info()
        else:
            self.audio.toggle()                      # 播放 ↔ 暂停
        self._btn_play_text = "⏸" if self.audio.playing else "▶"
        self.update()

    def _next(self):
        """下一首"""
        if self.audio.playlist:
            self.audio.next()
            self._update_track_info()
            self._btn_play_text = "⏸"
            self.update()

    def _prev(self):
        """上一首"""
        if self.audio.playlist:
            self.audio.prev()
            self._update_track_info()
            self._btn_play_text = "⏸"
            self.update()

    def _cycle_viz(self):
        """切换风格+配色（12 种组合）"""
        self._viz_style = (self._viz_style + 1) % 12
        self._settings.setValue("viz_style", self._viz_style)
        self.update()

    def _update_track_info(self):
        """根据当前索引更新歌名和艺术家标签"""
        if 0 <= self.audio.current_index < len(self.audio.playlist):
            path = self.audio.playlist[self.audio.current_index]
            meta = AudioEngine.get_metadata(path)    # 读取 ID3 标签
            self._track_title = meta['title']
            self._track_artist = meta['artist']
            self._save_state()                       # 自动保存状态

    def _save_state(self):
        """用 QSettings 持久化：当前文件夹路径 + 歌曲索引"""
        if self.audio.playlist and self.audio.current_index >= 0:
            folder = str(Path(self.audio.playlist[0]).parent)  # 取第一首所在文件夹
            self._settings.setValue("last_folder", folder)
            self._settings.setValue("last_index", self.audio.current_index)

    def _restore_state(self):
        """启动时恢复上次的播放状态"""
        folder = self._settings.value("last_folder")  # 读取设置
        if folder and os.path.isdir(folder):           # 文件夹仍存在
            count = self.audio.load_folder(folder)
            if count > 0:
                self._file_list = self.audio.playlist
                last_index = self._settings.value("last_index", 0, type=int)
                if last_index >= count:                # 防止索引越界
                    last_index = 0
                self.audio.play_index(last_index)
                self._update_track_info()
                self._btn_play_text = "⏸"
                self.update()
                return
        # 恢复失败：显示默认提示
        self._track_title = "未播放"
        self._track_artist = "请打开音乐文件夹"

    # ================================================================
    #  鼠标 & 键盘事件
    # ================================================================

    def _corner_at(self, pos):
        """判断鼠标坐标在哪个窗口角。
           z=30：四角 30×30px 为缩放热区。
           返回 0(TL), 1(TR), 2(BL), 3(BR), None(非角落)"""
        z = 30
        w, h = self.width(), self.height()
        if pos.x() < z and pos.y() < z:      return 0   # 左上 ↖
        if pos.x() > w - z and pos.y() < z:  return 1   # 右上 ↗
        if pos.x() < z and pos.y() > h - z:  return 2   # 左下 ↙
        if pos.x() > w - z and pos.y() > h - z: return 3  # 右下 ↘
        return None

    def mousePressEvent(self, event):
        """鼠标按下 → 螺丝 / 角缩放 / 拖拽 三选一"""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()                   # 相对于本控件的坐标

            # ① 手绘按钮点击
            if hasattr(self, '_btn_regions'):
                for i, r in enumerate(self._btn_regions):
                    if r.contains(pos):
                        if i == 0:
                            self._prev()
                        elif i == 1:
                            self._play_pause()
                        elif i == 2:
                            self._next()
                        return

            # ② 进度条点击 / 拖动跳转
            if hasattr(self, '_progress_rect') and self.audio.duration() > 0:
                pr = self._progress_rect
                if pr.contains(pos):
                    self._seeking = True         # 进入拖拽模式
                    frac = (pos.x() - pr.x()) / pr.width()
                    frac = max(0.0, min(1.0, frac))
                    self.audio.seek(int(self.audio.duration() * frac))
                    return

            # ② 角落缩放（优先于螺丝，避免误触）
            corner = self._corner_at(pos)
            if corner is not None:
                self._resize_corner = corner          # 记录哪个角
                self._resize_start = event.globalPosition().toPoint()  # 鼠标全局坐标
                self._resize_min = self.window().minimumSize() # 最小尺寸限制
                g = self.window().geometry()
                self._resize_ratio = g.width() / g.height()  # 锁定宽高比
                return

            # ③ 功能螺丝（缩小半径避免干扰拖拽）
            if hasattr(self, '_screw_positions'):
                r = 10 * (self.width() / 680)         # 响应半径随窗口缩放
                for idx in (0, 1, 2):                # 左上/右上/左下
                    sx, sy = self._screw_positions[idx]
                    dist = ((pos.x() - sx) ** 2 + (pos.y() - sy) ** 2) ** 0.5
                    if dist <= r:
                        if idx == 0:
                            self._open_folder()      # 左上：打开文件夹
                        elif idx == 1:
                            self.window().close()    # 右上：关闭窗口
                        else:
                            self._cycle_viz()        # 左下：切换可视化风格
                        return

            # ④ 否则：开始拖拽
            self._drag_start = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        """鼠标移动 → 进度条拖拽 / 缩放 / 窗口拖拽 / 光标切换"""
        # ── 进度条拖拽中 ──
        if (hasattr(self, '_seeking') and self._seeking
                and event.buttons() & Qt.MouseButton.LeftButton):
            if hasattr(self, '_progress_rect') and self.audio.duration() > 0:
                pr = self._progress_rect
                pos = event.position()
                frac = (pos.x() - pr.x()) / pr.width()
                frac = max(0.0, min(1.0, frac))
                self.audio.seek(int(self.audio.duration() * frac))
                return

        # ── 缩放中（增量式，锁定宽高比）──
        if (hasattr(self, '_resize_corner') and self._resize_corner is not None
                and event.buttons() & Qt.MouseButton.LeftButton):
            new_pos = event.globalPosition().toPoint()
            delta = new_pos - self._resize_start      # 上一帧到当前的增量
            g = self.window().geometry()              # 当前窗口几何
            mw, mh = self._resize_min.width(), self._resize_min.height()
            ow, oh = g.width(), g.height()            # 原始宽高
            c = self._resize_corner
            ratio = getattr(self, '_resize_ratio', 680 / 420)

            # 根据角落计算目标宽高
            if c in (0, 2):
                w = max(mw, ow - delta.x())           # 左边角：宽度缩小
            else:
                w = max(mw, ow + delta.x())           # 右边角：宽度增大
            if c in (0, 1):
                h = max(mh, oh - delta.y())           # 上边角：高度缩小
            else:
                h = max(mh, oh + delta.y())           # 下边角：高度增大

            # 宽高比锁定：取变化大的一方主导
            if abs(w - ow) > abs(h - oh):
                h = max(mh, w / ratio)
            else:
                w = max(mw, h * ratio)

            # 保持对角固定，计算新位置
            x, y = g.x(), g.y()
            if c in (0, 2):                           # 左边角 → 右边界固定
                x = g.x() + ow - w
            if c in (0, 1):                           # 上边角 → 下边界固定
                y = g.y() + oh - h

            self.window().setGeometry(int(x), int(y), int(w), int(h))
            self._resize_start = new_pos             # 更新参考点

        # ── 拖拽中 ──
        elif self._drag_start is not None and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.globalPosition().toPoint() - self._drag_start
            self.window().move(self.window().pos() + delta)  # 移动窗口
            self._drag_start = event.globalPosition().toPoint()

        # ── 悬停中（仅切换光标形状）──
        else:
            pos = event.position()
            # 按钮 hover 检测
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
            # 进度条上显示手型光标
            if (hasattr(self, '_progress_rect')
                    and self._progress_rect.contains(pos)
                    and self.audio.duration() > 0):
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            elif hasattr(self, '_btn_regions') and self._btn_hover >= 0:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                c = self._corner_at(pos)
                if c in (0, 3):
                    self.setCursor(Qt.CursorShape.SizeFDiagCursor)
                elif c in (1, 2):
                    self.setCursor(Qt.CursorShape.SizeBDiagCursor)
                else:
                    self.setCursor(Qt.CursorShape.ArrowCursor)
            if hover_changed:
                self.update()  # 重绘以显示/隐藏 hover 效果

    def mouseReleaseEvent(self, event):
        """鼠标释放 → 清除所有拖拽状态"""
        self._seeking = False
        self._resize_corner = None
        self._drag_start = None

    def keyPressEvent(self, event):
        """键盘快捷键：空格 = 播放/暂停，←→ = 切歌"""
        if event.key() == Qt.Key.Key_Space:
            self._play_pause()
        elif event.key() == Qt.Key.Key_Right:
            self._next()
        elif event.key() == Qt.Key.Key_Left:
            self._prev()

    def closeEvent(self, event):
        """窗口关闭前保存状态，停止音频和动画"""
        self._save_state()
        self.audio.stop()
        self._anim_timer.stop()
        event.accept()                             # 允许关闭


# ============================================================
#  主窗口 — 承载 CassettePlayer 的顶层容器
# ============================================================

class MainWindow(QMainWindow):
    """无边框透明主窗口"""

    def __init__(self):
        super().__init__()
        self.resize(500, 320)              # 启动时最小尺寸
        self.setMinimumSize(500, 320)      # 最小尺寸

        # FramelessWindowHint：去掉系统标题栏和边框 → 磁带形状即窗口
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        # WA_TranslucentBackground：允许窗口背景透明 → 桌面可见
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")

        # 将 CassettePlayer 设为中心控件（填满窗口）
        self.player = CassettePlayer()
        self.setCentralWidget(self.player)


# ============================================================
#  程序入口
# ============================================================

def main():
    """创建 Qt 应用 → 显示窗口 → 进入事件循环"""
    app = QApplication(sys.argv)           # Qt 应用实例（必须最先创建）

    # ── 全局暗色主题 ──
    app.setStyle("Fusion")                 # Fusion：跨平台一致的现代风格
    palette = app.palette()
    palette.setColor(palette.ColorRole.Window, QColor(10, 12, 20))  # 默认窗口暗色
    app.setPalette(palette)

    window = MainWindow()                  # 创建主窗口
    window.show()                          # 显示窗口
    sys.exit(app.exec())                   # 进入 Qt 事件循环（阻塞直到关闭）


if __name__ == "__main__":
    main()                                 # 直接运行时调用入口

import sys
import numpy as np
import sounddevice as sd
import pyqtgraph as pg
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QComboBox, QLabel, 
                               QSlider, QCheckBox, QGridLayout, QGroupBox, 
                               QFileDialog, QTextEdit, QMessageBox, QTabWidget,
                               QLineEdit)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor
import threading
import time
import random
import wave
import colorsys

class BrainwaveEngine:
    def __init__(self):
        self.sample_rate = 44100
        self.is_playing = False
        self.volume = 0.2
        self.target_volume = 0.2
        self.phase = 0
        self.noise_idx = 0
        self.buffer = np.zeros((2048, 2))
        self.lock = threading.Lock()

        self.noise_filter_state = 0.0
        self.notch_x1, self.notch_x2 = 0, 0
        self.notch_y1, self.notch_y2 = 0, 0
        self.cf_buffer = np.zeros(9 * 2)
        self.cf_lp = 0.0

        self.is_recording = False
        self.wav_file = None
        self.output_device = None

        self.states = {
            "Custom": 10.0, "Delta (Deep Sleep)": 2.0, "Theta (Meditation)": 6.0,
            "Schumann (Grounding)": 7.83, "Alpha (Relaxation)": 10.0,
            "Beta (Focus)": 20.0, "Gamma (Cognition)": 40.0,
            "Hyper-Gamma (Overload)": 60.0, "High Beta (Anxiety)": 30.0
        }
        self.tones = {
            "Custom": 432.0, "174 Hz (Pain Relief)": 174.0, "396 Hz (Root Chakra)": 396.0,
            "528 Hz (Solar Plexus)": 528.0, "639 Hz (Heart)": 639.0,
            "741 Hz (Throat)": 741.0, "852 Hz (Third Eye)": 852.0
        }
        
        self.current_state = "Alpha (Relaxation)"
        self.beat_freq = 10.0
        self.current_tone = "639 Hz (Heart)"
        self.carrier_freq = 432.0
        
        self.mode = "Binaural Beats"
        self.soundscape = "None"
        self.drone_mode = "None"
        self.bilateral_enabled = False
        self.breathing_enabled = False
        self.auto_drift_enabled = False
        self.auto_ascending_enabled = False
        self.crossfeed_enabled = False
        
        self.is_terror_mode = False
        self.photic_enabled = False
        self.ganzfeld_enabled = False
        self.glitch_enabled = False
        self.subliminal_enabled = False
        self.subliminal_text = ""
        self.visual_mode = "Mandala Engine"
        
        self.tinnitus_freq = 8000.0
        
        self.breath_patterns = {
            "4-7-8 Relaxation": (4, 7, 8, 0),
            "4-4-4-4 Box Breathing": (4, 4, 4, 4),
            "5-5 Coherent (HRV)": (5, 0, 5, 0)
        }
        self.current_breath_pattern = "4-7-8 Relaxation"
        
        self.startle_triggered = False
        self.startle_timer = 0
        
        self.brown_noise = self._generate_brown_noise(10)
        self.pink_noise = self._generate_pink_noise(10)

    def _generate_brown_noise(self, duration_sec):
        samples = int(self.sample_rate * duration_sec)
        white = np.random.randn(samples) * 0.1
        brown = np.zeros(samples)
        last = 0.0
        for i in range(samples):
            last = (last + 0.02 * white[i]) / 1.02
            brown[i] = last * 3.5
        return brown

    def _generate_pink_noise(self, duration_sec):
        samples = int(self.sample_rate * duration_sec)
        pink = np.zeros(samples)
        b0, b1, b2, b3 = 0, 0, 0, 0
        for i in range(samples):
            white = np.random.randn()
            b0 = 0.99886 * b0 + white * 0.0555179
            b1 = 0.99332 * b1 + white * 0.0750759
            b2 = 0.96900 * b2 + white * 0.1538520
            b3 = 0.86650 * b3 + white * 0.3104856
            pink[i] = (b0 + b1 + b2 + b3 + white * 0.5) * 0.11
        return pink

    def _apply_iir_filter(self, chunk, alpha=0.2):
        out = np.zeros_like(chunk)
        for i in range(len(chunk)):
            self.noise_filter_state = (1 - alpha) * self.noise_filter_state + alpha * chunk[i]
            out[i] = self.noise_filter_state
        return out

    def _apply_notch_filter(self, chunk):
        w0 = 2 * np.pi * self.tinnitus_freq / self.sample_rate
        Q = 8.0
        alpha = np.sin(w0) / (2 * Q)
        b0, b1, b2 = 1.0, -2.0 * np.cos(w0), 1.0
        a0, a1, a2 = 1.0 + alpha, -2.0 * np.cos(w0), 1.0 - alpha
        out = np.zeros_like(chunk)
        for i in range(len(chunk)):
            x = chunk[i]
            y = (b0*x + b1*self.notch_x1 + b2*self.notch_x2 - a1*self.notch_y1 - a2*self.notch_y2) / a0
            self.notch_x2, self.notch_x1 = self.notch_x1, x
            self.notch_y2, self.notch_y1 = self.notch_y1, y
            out[i] = y
        return out

    def _apply_crossfeed(self, wave):
        delay = 9
        for i in range(len(wave)):
            delayed_l = self.cf_buffer[delay]
            delayed_r = self.cf_buffer[delay + 1]
            self.cf_lp = 0.5 * self.cf_lp + 0.5 * (delayed_l + delayed_r) * 0.3
            wave[i, 0] += self.cf_lp * 0.5
            wave[i, 1] += self.cf_lp * 0.5
            self.cf_buffer[delay] = wave[i, 0]
            self.cf_buffer[delay + 1] = wave[i, 1]
            self.cf_buffer = np.roll(self.cf_buffer, -2)
        return wave

    def _get_soundscape(self, frames, t):
        if self.soundscape == "None": return np.zeros(frames)
        elif self.soundscape == "Ocean":
            lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t)
            chunk = self.brown_noise[self.noise_idx : self.noise_idx + frames]
            self.noise_idx = (self.noise_idx + frames) % len(self.brown_noise)
            return chunk * lfo * 1.5
        elif self.soundscape == "Pink Noise":
            chunk = self.pink_noise[self.noise_idx : self.noise_idx + frames]
            self.noise_idx = (self.noise_idx + frames) % len(self.pink_noise)
            return chunk * 1.2
        elif self.soundscape == "Infrasound (19 Hz)":
            return np.sin(2 * np.pi * 19 * t) * 3.0
        elif self.soundscape == "Vibroacoustic (40 Hz)":
            return np.sin(2 * np.pi * 40 * t) * 1.5
        elif self.soundscape == "Dread Heartbeat":
            beat_t = t % 1.3
            env = np.exp(-((beat_t % 0.1) * 30)**2) + np.exp(-(((beat_t+0.6) % 0.1) * 30)**2)
            mask = (np.sin(2 * np.pi * 0.7 * t) > 0.2).astype(float)
            chunk = self.brown_noise[self.noise_idx : self.noise_idx + frames]
            self.noise_idx = (self.noise_idx + frames) % len(self.brown_noise)
            return chunk * env * mask * 4.0
        elif self.soundscape == "3D Phantom Whispers":
            chunk = self.pink_noise[self.noise_idx : self.noise_idx + frames]
            self.noise_idx = (self.noise_idx + frames) % len(self.pink_noise)
            chunk = self._apply_iir_filter(chunk, 0.15)
            theta = 2 * np.pi * 0.3 * t
            left_gain = 0.5 + 0.5 * np.cos(theta)
            right_gain = 0.5 + 0.5 * np.cos(theta + np.pi)
            lfo = np.sin(2 * np.pi * 3.7 * t) * np.sin(2 * np.pi * 0.3 * t)
            left_audio = chunk * lfo * left_gain * 5.0
            right_audio = chunk * lfo * right_gain * 5.0
            return np.column_stack([left_audio, right_audio])
        elif self.soundscape == "Tinnitus Notch Therapy":
            chunk = self.pink_noise[self.noise_idx : self.noise_idx + frames]
            self.noise_idx = (self.noise_idx + frames) % len(self.pink_noise)
            chunk = self._apply_notch_filter(chunk)
            return chunk * 2.0
        return np.zeros((frames, 2))

    def _get_drone(self, t):
        if self.drone_mode == "None": return np.zeros(len(t))
        root = self.carrier_freq / 2.0 
        if self.is_terror_mode:
            wobble = 1.0 + 0.02 * np.sin(2 * np.pi * 0.13 * t) + 0.01 * np.sin(2 * np.pi * 1.7 * t)
        else:
            wobble = 1.0 + 0.005 * np.sin(2 * np.pi * 0.2 * t)
        root = root * wobble
        
        if self.drone_mode == "Calm Pad (Sus2)":
            return np.sin(2*np.pi*root*t) + 0.5*np.sin(2*np.pi*(root*1.122)*t) + 0.4*np.sin(2*np.pi*(root*1.5)*t)
        elif self.drone_mode == "Ambient (Maj7)":
            return np.sin(2*np.pi*root*t) + 0.4*np.sin(2*np.pi*(root*1.26)*t) + 0.3*np.sin(2*np.pi*(root*1.5)*t) + 0.2*np.sin(2*np.pi*(root*1.875)*t)
        elif self.drone_mode == "Tritone (Diabolus)":
            return np.sin(2*np.pi*root*t) + 0.8*np.sin(2*np.pi*(root*1.4142)*t)
        return np.zeros(len(t))

    def _get_subliminal_audio(self, t, frames):
        if not self.subliminal_enabled or not self.subliminal_text: return np.zeros((frames, 2))
        syllable_pulse = (np.sin(2 * np.pi * 3.2 * t) > 0.3).astype(float)
        chunk = self.pink_noise[self.noise_idx : self.noise_idx + frames]
        self.noise_idx = (self.noise_idx + frames) % len(self.pink_noise)
        voice_noise = self._apply_iir_filter(chunk, 0.3) * syllable_pulse
        masked = voice_noise * 0.05
        theta = 2 * np.pi * 0.1 * t
        left_gain = 0.5 + 0.5 * np.cos(theta)
        right_gain = 0.5 + 0.5 * np.cos(theta + np.pi)
        return np.column_stack([masked * left_gain, masked * right_gain])

    def _get_bilateral(self, t):
        pan_lfo = np.sin(2 * np.pi * 0.4 * t)
        left_env = np.maximum(0, pan_lfo)**2
        right_env = np.maximum(0, -pan_lfo)**2
        chime = np.sin(2 * np.pi * 1000 * t) * np.exp(-((t % 2.5) * 5))
        return chime * left_env, chime * right_env

    def _get_breathing_audio(self, t, frames):
        i, h1, e, h2 = self.breath_patterns[self.current_breath_pattern]
        total = i + h1 + e + h2
        cycle_t = t % total
        env = np.zeros(frames)
        inhale = cycle_t < i
        hold1 = (cycle_t >= i) & (cycle_t < i + h1)
        exhale = (cycle_t >= i + h1) & (cycle_t < i + h1 + e)
        env[inhale] = cycle_t[inhale] / i if i > 0 else 1
        env[exhale] = 1.0 - ((cycle_t[exhale] - (i + h1)) / e) if e > 0 else 0
        chunk = self.pink_noise[self.noise_idx : self.noise_idx + frames]
        self.noise_idx = (self.noise_idx + frames) % len(self.pink_noise)
        chunk = self._apply_iir_filter(chunk, 0.1)
        return chunk * env * 1.5

    def start_recording(self, filename):
        self.wav_file = wave.open(filename, 'w')
        self.wav_file.setnchannels(2)
        self.wav_file.setsampwidth(2) 
        self.wav_file.setframerate(self.sample_rate)
        self.is_recording = True

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.wav_file.close()
            self.wav_file = None

    def get_audio_callback(self):
        def callback(outdata, frames, time_info, status):
            if not self.is_playing:
                outdata.fill(0)
                return

            t = (np.arange(frames) + self.phase) / self.sample_rate
            self.phase += frames

            if self.mode == "Binaural Beats":
                left_freq = self.carrier_freq
                right_freq = self.carrier_freq + self.beat_freq
                modulation = 0.5 + 0.5 * np.sin(2 * np.pi * (self.beat_freq / 2) * t)
                left_channel = np.sin(2 * np.pi * left_freq * t) * modulation
                right_channel = np.sin(2 * np.pi * right_freq * t) * modulation
            else:
                pulse = 0.5 * (1 + np.sin(2 * np.pi * self.beat_freq * t))
                pulse = np.power(pulse, 4)
                wave = np.sin(2 * np.pi * self.carrier_freq * t)
                left_channel = wave * pulse
                right_channel = wave * pulse

            wave = np.column_stack([left_channel, right_channel]) * self.volume * 0.5

            duck_amount = 1.0
            if self.subliminal_enabled: duck_amount *= 0.6
            if self.is_terror_mode and self.startle_triggered: duck_amount *= 0.3

            if self.drone_mode != "None":
                drone = self._get_drone(t) * self.volume * 0.3 * duck_amount
                wave[:, 0] += drone
                wave[:, 1] += drone

            soundscape_audio = self._get_soundscape(frames, t) * self.volume * 0.4 * duck_amount
            if soundscape_audio.ndim == 1:
                wave[:, 0] += soundscape_audio
                wave[:, 1] += soundscape_audio
            else:
                wave += soundscape_audio

            if self.bilateral_enabled:
                bi_left, bi_right = self._get_bilateral(t)
                wave[:, 0] += bi_left * self.volume * 0.8
                wave[:, 1] += bi_right * self.volume * 0.8

            if self.breathing_enabled:
                breath = self._get_breathing_audio(t, frames) * self.volume * 0.6
                wave[:, 0] += breath
                wave[:, 1] += breath

            if self.subliminal_enabled:
                wave += self._get_subliminal_audio(t, frames) * self.volume * 2.0

            if self.is_terror_mode and self.startle_triggered:
                stinger = np.random.randn(frames) * np.exp(-np.arange(frames) / (self.sample_rate * 0.2))
                wave[:, 0] += stinger * 0.6
                wave[:, 1] += stinger * 0.6
                self.startle_timer += frames
                if self.startle_timer > self.sample_rate * 0.3:
                    self.startle_triggered = False
                    self.startle_timer = 0

            if self.is_terror_mode and self.glitch_enabled:
                if random.random() < 0.05:
                    glitch_len = random.randint(int(self.sample_rate*0.05), int(self.sample_rate*0.2))
                    if frames > glitch_len:
                        idx = random.randint(0, frames - glitch_len - 1)
                        wave[idx:idx+glitch_len] = wave[idx:idx+glitch_len][::-1].copy()

            if self.crossfeed_enabled:
                wave = self._apply_crossfeed(wave)

            wave = np.tanh(wave * 1.5) / 1.5

            if self.target_volume < self.volume:
                self.volume -= 0.001
                if self.volume < self.target_volume: self.volume = 0
                wave *= (self.volume / (self.volume + 0.001))

            outdata[:] = wave

            if self.is_recording and self.wav_file:
                int_data = (wave * 32767).astype(np.int16)
                self.wav_file.writeframes(int_data.tobytes())

            with self.lock:
                self.buffer = wave.copy()
                
        return callback

    def start(self):
        if not self.is_playing:
            self.is_playing = True
            self.stream = sd.OutputStream(
                device=self.output_device, 
                channels=2, samplerate=self.sample_rate, 
                dtype='float32', callback=self.get_audio_callback()
            )
            self.stream.start()

    def stop(self):
        if self.is_playing:
            self.is_playing = False
            self.stream.stop()
            self.stream.close()

class VisualizerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clinical Psychoacoustic Manipulation Suite Pro")
        self.resize(1200, 900)
        self.setStyleSheet("background-color: #0a0a0a; color: #eeeeee;")

        self.engine = BrainwaveEngine()
        self.session_start_time = None
        self.is_zen_mode = False
        self.is_terror_mode = False
        self.is_fullscreen_viz = False
        self.chroma_flow = False
        self.trails_enabled = False
        
        # Trail History Buffer
        self.trail_history = []

        # Mouse position for interactive Vortex
        self.mouse_pos = np.array([0.0, 0.0])

        self.presets = {
            "Custom": (None, None, None, None, False, False, False, False, False, False, False, False, False, "#00d2fc"),
            "Force Calm (Auto-Drift)": ("Custom", "432 Hz (Heart)", "Pink Noise", "Calm Pad (Sus2)", False, True, True, False, False, True, False, False, False, "#06d6a0"),
            "Deep Sleep": ("Delta (Deep Sleep)", "396 Hz (Root Chakra)", "Ocean", "Ambient (Maj7)", False, False, False, False, False, True, False, False, False, "#4361ee"),
            "ACUTE PARANOIA": ("High Beta (Anxiety)", "Custom", "3D Phantom Whispers", "Tritone (Diabolus)", False, False, False, True, True, False, False, True, False, "#8b0000"),
            "THE UNSEEN PRESENCE": ("Hyper-Gamma (Overload)", "Custom", "Infrasound (19 Hz)", "Tritone (Diabolus)", False, False, False, True, True, False, False, True, True, "#4a0000"),
            "DREAD & SORROW": ("Custom", "174 Hz (Pain Relief)", "Dread Heartbeat", "Tritone (Diabolus)", False, False, False, True, False, False, False, False, False, "#5e0000"),
            "SUBCONSCIOUS REPROGRAMMING": ("Theta (Meditation)", "528 Hz (Solar Plexus)", "Ocean", "Calm Pad (Sus2)", True, True, True, False, False, False, True, False, False, "#f72585"),
            "MORNING ACTIVATION (Ascend)": ("Custom", "741 Hz (Throat)", "Vibroacoustic (40 Hz)", "Calm Pad (Sus2)", False, False, False, False, False, False, False, False, True, "#ffaa00"),
            "TINNITUS RELIEF": ("Alpha (Relaxation)", "432 Hz (Heart)", "Tinnitus Notch Therapy", "Ambient (Maj7)", False, True, False, False, False, True, False, False, False, "#ffffff"),
        }

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(10)

        header_layout = QHBoxLayout()
        self.title = QLabel("Clinical Psychoacoustic Suite Pro")
        self.title.setStyleSheet("font-size: 22px; font-weight: bold; color: #00d2fc;")
        header_layout.addWidget(self.title)
        header_layout.addStretch()
        
        self.viz_fs_btn = QPushButton("Fullscreen Viz")
        self.viz_fs_btn.clicked.connect(self.toggle_fullscreen_viz)
        self.viz_fs_btn.setStyleSheet("QPushButton { background-color: #1a1a1a; padding: 8px; border-radius: 4px; }")
        header_layout.addWidget(self.viz_fs_btn)
        
        self.zen_ui_btn = QPushButton("Zen UI")
        self.zen_ui_btn.setCheckable(True)
        self.zen_ui_btn.clicked.connect(self.toggle_zen_ui)
        self.zen_ui_btn.setStyleSheet("QPushButton { background-color: #1a1a1a; padding: 8px; border-radius: 4px; } QPushButton:checked { background-color: #0f3460; }")
        header_layout.addWidget(self.zen_ui_btn)
        
        self.timer_label = QLabel("Session: 00:00")
        self.timer_label.setStyleSheet("font-size: 16px; color: #e94560; font-weight: bold;")
        header_layout.addWidget(self.timer_label)
        main_layout.addLayout(header_layout)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#050505')
        self.plot_widget.setAspectLocked(lock=True, ratio=1.0)
        self.plot_widget.hideAxis('left')
        self.plot_widget.hideAxis('bottom')
        self.plot_widget.setXRange(-1.0, 1.0)
        self.plot_widget.setYRange(-1.0, 1.0)
        # Track mouse for Vortex interactivity
        self.plot_widget.scene().sigMouseMoved.connect(self.update_mouse_pos)
        
        self.glow_curve = self.plot_widget.plot(pen=pg.mkPen(color=(0, 210, 252, 50), width=12))
        self.main_curve = self.plot_widget.plot(pen=pg.mkPen(color='#00d2fc', width=2))
        
        # Extra curves for multi-layer visualizers
        self.layer2_curve = self.plot_widget.plot(pen=pg.mkPen(color='#00d2fc', width=2))
        self.layer3_curve = self.plot_widget.plot(pen=pg.mkPen(color='#00d2fc', width=2))
        
        self.phyllo_scatter = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush(0, 210, 252, 200))
        self.plot_widget.addItem(self.phyllo_scatter)
        
        self.vortex_pos = np.random.uniform(-0.8, 0.8, (1000, 2))
        self.vortex_vel = np.zeros((1000, 2))
        self.vortex_scatter = pg.ScatterPlotItem(size=3, pen=pg.mkPen(None), brush=pg.mkBrush(0, 210, 252, 150))
        self.plot_widget.addItem(self.vortex_scatter)
        
        self.zen_ripples = []
        for i in range(5):
            circle = pg.QtWidgets.QGraphicsEllipseItem(-0.1, -0.1, 0.2, 0.2)
            circle.setPen(pg.mkPen(color=(255, 255, 255, 0), width=2))
            self.plot_widget.addItem(circle)
            self.zen_ripples.append(circle)
        
        self.breath_circle = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 50))
        self.plot_widget.addItem(self.breath_circle)
        
        self.breath_text = pg.TextItem("", color=(255, 255, 255, 200), anchor=(0.5, 0.5))
        self.breath_text.setPos(0, -0.8)
        self.plot_widget.addItem(self.breath_text)
        
        self.strobe_overlay = QWidget(self)
        self.strobe_overlay.setStyleSheet("background-color: transparent;")
        self.strobe_overlay.setGeometry(self.rect())
        self.strobe_overlay.lower()

        main_layout.addWidget(self.plot_widget)

        self.fft_widget = pg.PlotWidget()
        self.fft_widget.setBackground('#0a0a0a')
        self.fft_widget.setXRange(0, 2000)
        self.fft_widget.setYRange(-60, 10)
        self.fft_widget.setLabel('left', 'dB')
        self.fft_widget.setLabel('bottom', 'Hz')
        self.fft_curve = self.fft_widget.plot(pen=pg.mkPen(color='#e94560', width=2))
        main_layout.addWidget(self.fft_widget)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabWidget::pane { border: 1px solid #333; } QTabBar::tab { background: #1a1a1a; padding: 8px 20px; margin-right: 2px; }")
        
        tab1 = QWidget()
        t1_layout = QGridLayout(tab1)
        t1_layout.addWidget(QLabel("Preset:"), 0, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(self.presets.keys()))
        self.preset_combo.currentTextChanged.connect(self.apply_preset)
        self.preset_combo.setStyleSheet("background-color: #1a1a1a; padding: 5px; border-radius: 4px;")
        t1_layout.addWidget(self.preset_combo, 0, 1)
        
        t1_layout.addWidget(QLabel("Brainwave:"), 1, 0)
        self.state_combo = QComboBox()
        self.state_combo.addItems(list(self.engine.states.keys()))
        self.state_combo.currentTextChanged.connect(self.change_state)
        self.state_combo.setStyleSheet("background-color: #1a1a1a; padding: 5px; border-radius: 4px;")
        t1_layout.addWidget(self.state_combo, 1, 1)

        t1_layout.addWidget(QLabel("Therapy Tone:"), 2, 0)
        self.tone_combo = QComboBox()
        self.tone_combo.addItems(list(self.engine.tones.keys()))
        self.tone_combo.currentTextChanged.connect(self.change_tone)
        self.tone_combo.setStyleSheet("background-color: #1a1a1a; padding: 5px; border-radius: 4px;")
        t1_layout.addWidget(self.tone_combo, 2, 1)

        t1_layout.addWidget(QLabel("Custom Carrier (Hz):"), 3, 0)
        self.carrier_input = QLineEdit(str(self.engine.carrier_freq))
        self.carrier_input.returnPressed.connect(self.change_carrier_input)
        self.carrier_input.setStyleSheet("background-color: #1a1a1a; padding: 5px; border-radius: 4px;")
        t1_layout.addWidget(self.carrier_input, 3, 1)

        t1_layout.addWidget(QLabel("Target Hz:"), 4, 0)
        self.brainwave_slider = QSlider(Qt.Horizontal)
        self.brainwave_slider.setRange(1, 600)
        self.brainwave_slider.setValue(100)
        self.brainwave_slider.valueChanged.connect(self.change_brainwave_hz)
        t1_layout.addWidget(self.brainwave_slider, 4, 1)
        self.brainwave_label = QLabel("10.0 Hz")
        t1_layout.addWidget(self.brainwave_label, 4, 2)
        
        t1_layout.addWidget(QLabel("Visual Mode:"), 5, 0)
        self.visual_combo = QComboBox()
        self.visual_combo.addItems(["Mandala Engine", "Radial Spectrum", "Lissajous", "Hypno-Tunnel", "Particle Vortex", "Sacred Geometry", "Zen Ripples"])
        self.visual_combo.currentTextChanged.connect(self.change_visual_mode)
        self.visual_combo.setStyleSheet("background-color: #1a1a1a; padding: 5px; border-radius: 4px;")
        t1_layout.addWidget(self.visual_combo, 5, 1)
        
        self.chroma_check = QCheckBox("Chroma-Flow")
        self.chroma_check.stateChanged.connect(self.toggle_chroma)
        t1_layout.addWidget(self.chroma_check, 5, 2)
        
        self.trails_check = QCheckBox("Trails (Spirograph)")
        self.trails_check.stateChanged.connect(self.toggle_trails)
        t1_layout.addWidget(self.trails_check, 6, 1)

        self.tabs.addTab(tab1, "Entrainment")

        tab2 = QWidget()
        t2_layout = QGridLayout(tab2)
        t2_layout.addWidget(QLabel("Soundscape:"), 0, 0)
        self.soundscape_combo = QComboBox()
        self.soundscape_combo.addItems(["None", "Ocean", "Pink Noise", "Infrasound (19 Hz)", "Vibroacoustic (40 Hz)", "Dread Heartbeat", "3D Phantom Whispers", "Tinnitus Notch Therapy"])
        self.soundscape_combo.currentTextChanged.connect(self.change_soundscape)
        self.soundscape_combo.setStyleSheet("background-color: #1a1a1a; padding: 5px; border-radius: 4px;")
        t2_layout.addWidget(self.soundscape_combo, 0, 1)

        t2_layout.addWidget(QLabel("Drone:"), 1, 0)
        self.drone_combo = QComboBox()
        self.drone_combo.addItems(["None", "Calm Pad (Sus2)", "Ambient (Maj7)", "Tritone (Diabolus)"])
        self.drone_combo.currentTextChanged.connect(self.change_drone)
        self.drone_combo.setStyleSheet("background-color: #1a1a1a; padding: 5px; border-radius: 4px;")
        t2_layout.addWidget(self.drone_combo, 1, 1)

        t2_layout.addWidget(QLabel("Audio Mode:"), 2, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Binaural Beats", "Isochronic Tones"])
        self.mode_combo.currentTextChanged.connect(self.change_mode)
        self.mode_combo.setStyleSheet("background-color: #1a1a1a; padding: 5px; border-radius: 4px;")
        t2_layout.addWidget(self.mode_combo, 2, 1)

        t2_layout.addWidget(QLabel("Tinnitus Freq (Hz):"), 3, 0)
        self.tinnitus_slider = QSlider(Qt.Horizontal)
        self.tinnitus_slider.setRange(1000, 12000)
        self.tinnitus_slider.setValue(8000)
        self.tinnitus_slider.valueChanged.connect(self.change_tinnitus_freq)
        t2_layout.addWidget(self.tinnitus_slider, 3, 1)
        self.tinnitus_label = QLabel("8000 Hz")
        t2_layout.addWidget(self.tinnitus_label, 3, 2)

        self.tabs.addTab(tab2, "Soundscapes")

        tab3 = QWidget()
        t3_layout = QGridLayout(tab3)
        self.drift_check = QCheckBox("Auto-Drift (Calm)")
        self.drift_check.stateChanged.connect(self.toggle_drift)
        t3_layout.addWidget(self.drift_check, 0, 0)

        self.ascend_check = QCheckBox("Auto-Ascend (Wake)")
        self.ascend_check.stateChanged.connect(self.toggle_ascend)
        t3_layout.addWidget(self.ascend_check, 0, 1)

        self.bilateral_check = QCheckBox("Bilateral (EMDR)")
        self.bilateral_check.stateChanged.connect(self.toggle_bilateral)
        t3_layout.addWidget(self.bilateral_check, 1, 0)

        self.breathing_check = QCheckBox("Breathing Pacer")
        self.breathing_check.stateChanged.connect(self.toggle_breathing)
        t3_layout.addWidget(self.breathing_check, 1, 1)
        
        self.breath_combo = QComboBox()
        self.breath_combo.addItems(list(self.engine.breath_patterns.keys()))
        self.breath_combo.currentTextChanged.connect(self.change_breath_pattern)
        self.breath_combo.setStyleSheet("background-color: #1a1a1a; padding: 3px; border-radius: 4px; font-size: 11px;")
        t3_layout.addWidget(self.breath_combo, 1, 2)

        self.photic_check = QCheckBox("Photic Strobe")
        self.photic_check.stateChanged.connect(self.toggle_photic)
        self.photic_check.setStyleSheet("color: #ffaa00;")
        t3_layout.addWidget(self.photic_check, 2, 0)

        self.ganzfeld_check = QCheckBox("Ganzfeld (Hallucinate)")
        self.ganzfeld_check.stateChanged.connect(self.toggle_ganzfeld)
        self.ganzfeld_check.setStyleSheet("color: #ffaa00;")
        t3_layout.addWidget(self.ganzfeld_check, 2, 1)

        self.crossfeed_check = QCheckBox("Headphone Crossfeed")
        self.crossfeed_check.stateChanged.connect(self.toggle_crossfeed)
        t3_layout.addWidget(self.crossfeed_check, 2, 2)

        self.terror_check = QCheckBox("TERROR MODE")
        self.terror_check.stateChanged.connect(self.toggle_terror)
        self.terror_check.setStyleSheet("color: #ff3333; font-weight: bold;")
        t3_layout.addWidget(self.terror_check, 3, 0)

        self.glitch_check = QCheckBox("Glitch Engine")
        self.glitch_check.stateChanged.connect(self.toggle_glitch)
        self.glitch_check.setStyleSheet("color: #ff3333;")
        t3_layout.addWidget(self.glitch_check, 3, 1)
        
        self.subliminal_check = QCheckBox("Subliminal Voice")
        self.subliminal_check.stateChanged.connect(self.toggle_subliminal)
        t3_layout.addWidget(self.subliminal_check, 4, 0)
        
        self.subliminal_input = QLineEdit()
        self.subliminal_input.setPlaceholderText("Type subliminal affirmation...")
        self.subliminal_input.setStyleSheet("background-color: #1a1a1a; padding: 5px; border-radius: 4px;")
        self.subliminal_input.textChanged.connect(self.update_subliminal_text)
        t3_layout.addWidget(self.subliminal_input, 4, 1, 1, 2)

        self.tabs.addTab(tab3, "Modifiers")

        tab4 = QWidget()
        t4_layout = QVBoxLayout(tab4)
        t4_layout.addWidget(QLabel("Output Audio Device:"))
        self.device_combo = QComboBox()
        self.populate_audio_devices()
        self.device_combo.currentIndexChanged.connect(self.change_audio_device)
        self.device_combo.setStyleSheet("background-color: #1a1a1a; padding: 5px; border-radius: 4px;")
        t4_layout.addWidget(self.device_combo)
        t4_layout.addStretch()
        self.tabs.addTab(tab4, "Hardware")

        tab5 = QWidget()
        t5_layout = QVBoxLayout(tab5)
        self.journal_text = QTextEdit()
        self.journal_text.setPlaceholderText("After your session, type how you feel here and click 'Save Log'...")
        self.journal_text.setStyleSheet("background-color: #1a1a1a; border-radius: 4px;")
        t5_layout.addWidget(self.journal_text)
        save_log_btn = QPushButton("Save Session Log")
        save_log_btn.clicked.connect(self.save_journal)
        save_log_btn.setStyleSheet("QPushButton { background-color: #0f3460; padding: 8px; border-radius: 4px; }")
        t5_layout.addWidget(save_log_btn)
        self.tabs.addTab(tab5, "Journal")

        main_layout.addWidget(self.tabs)

        bottom_layout = QHBoxLayout()
        self.record_btn = QPushButton("● REC to WAV")
        self.record_btn.setCheckable(True)
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setStyleSheet("QPushButton { background-color: #1a1a1a; padding: 10px; font-weight: bold; border-radius: 5px; } QPushButton:checked { background-color: #e94560; }")
        bottom_layout.addWidget(self.record_btn)

        self.play_btn = QPushButton("▶  START SESSION")
        self.play_btn.setCheckable(True)
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setStyleSheet("""
            QPushButton { background-color: #e94560; padding: 15px; font-size: 16px; font-weight: bold; border-radius: 8px; }
            QPushButton:hover { background-color: #ff5570; }
            QPushButton:checked { background-color: #28a745; }
            QPushButton:checked:hover { background-color: #34c759; }
        """)
        bottom_layout.addWidget(self.play_btn)

        bottom_layout.addWidget(QLabel("Vol:"))
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(20)
        self.vol_slider.valueChanged.connect(self.change_volume)
        bottom_layout.addWidget(self.vol_slider)
        self.vol_label = QLabel("20%")
        bottom_layout.addWidget(self.vol_label)

        main_layout.addLayout(bottom_layout)

        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(30)
        
        self.drift_timer = QTimer()
        self.drift_timer.timeout.connect(self.process_drift)
        
        self.startle_timer = QTimer()
        self.startle_timer.timeout.connect(self.trigger_startle)

    def update_mouse_pos(self, evt):
        if self.engine.visual_mode == "Particle Vortex":
            # Map screen mouse pos to -1, 1 coordinate space
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(evt)
            self.mouse_pos[0] = mouse_point.x()
            self.mouse_pos[1] = mouse_point.y()

    def toggle_chroma(self, state): self.chroma_flow = bool(state)
    def toggle_trails(self, state): 
        self.trails_enabled = bool(state)
        if not self.trails_enabled: self.trail_history.clear()

    def toggle_fullscreen_viz(self):
        self.is_fullscreen_viz = not self.is_fullscreen_viz
        if self.is_fullscreen_viz:
            self.showFullScreen()
            self.tabs.hide()
            self.fft_widget.hide()
            self.record_btn.hide()
            self.vol_slider.hide()
            self.vol_label.hide()
            self.zen_ui_btn.hide()
            self.viz_fs_btn.setText("Exit Fullscreen")
        else:
            self.showNormal()
            self.tabs.show()
            self.fft_widget.show()
            self.record_btn.show()
            self.vol_slider.show()
            self.vol_label.show()
            self.zen_ui_btn.show()
            self.viz_fs_btn.setText("Fullscreen Viz")

    def populate_audio_devices(self):
        devices = sd.query_devices()
        self.device_combo.clear()
        for i, dev in enumerate(devices):
            if dev['max_output_channels'] >= 2:
                self.device_combo.addItem(f"{i}: {dev['name']}", i)

    def change_audio_device(self, index):
        device_id = self.device_combo.itemData(index)
        self.engine.output_device = device_id
        if self.engine.is_playing:
            self.engine.stop()
            self.engine.start()

    def change_visual_mode(self, mode):
        self.engine.visual_mode = mode
        if mode != "Particle Vortex": self.vortex_scatter.setData([], [])
        if mode != "Sacred Geometry": self.phyllo_scatter.setData([], [])
        if mode not in ["Lissajous", "Hypno-Tunnel", "Radial Spectrum", "Mandala Engine"]:
            self.glow_curve.setData([], [])
            self.main_curve.setData([], [])
            self.layer2_curve.setData([], [])
            self.layer3_curve.setData([], [])
        if mode != "Zen Ripples":
            for c in self.zen_ripples: c.setPen(pg.mkPen(color=(255, 255, 255, 0), width=2))

    def toggle_zen_ui(self, state):
        is_hidden = bool(state)
        self.tabs.setHidden(is_hidden)
        self.fft_widget.setHidden(is_hidden)
        self.record_btn.setHidden(is_hidden)
        self.vol_slider.setHidden(is_hidden)
        self.vol_label.setHidden(is_hidden)
        self.zen_ui_btn.setText("Show UI" if is_hidden else "Zen UI")

    def change_tinnitus_freq(self, val):
        self.engine.tinnitus_freq = float(val)
        self.tinnitus_label.setText(f"{val} Hz")

    def change_carrier_input(self):
        try:
            val = float(self.carrier_input.text())
            self.engine.carrier_freq = val
            self.tone_combo.blockSignals(True)
            self.tone_combo.setCurrentText("Custom")
            self.tone_combo.blockSignals(False)
            if self.preset_combo.currentText() != "Custom": self._manual_override()
        except ValueError: pass

    def change_breath_pattern(self, text): self.engine.current_breath_pattern = text

    def save_journal(self):
        notes = self.journal_text.toPlainText()
        duration = self.timer_label.text()
        preset = self.preset_combo.currentText()
        with open("session_log.txt", "a", encoding="utf-8") as f:
            f.write(f"\n--- Session: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            f.write(f"Preset: {preset} | Duration: {duration}\n")
            f.write(f"Notes: {notes}\n")
        self.journal_text.clear()
        QMessageBox.information(self, "Saved", "Session log saved to session_log.txt")

    def trigger_startle(self):
        if self.engine.is_terror_mode: self.engine.startle_triggered = True

    def toggle_recording(self):
        if self.record_btn.isChecked():
            filename, _ = QFileDialog.getSaveFileName(self, "Save Recording", "", "WAV Files (*.wav)")
            if filename:
                if not filename.endswith('.wav'): filename += '.wav'
                self.engine.start_recording(filename)
                self.record_btn.setText("■ STOP REC")
            else: self.record_btn.setChecked(False)
        else:
            self.engine.stop_recording()
            self.record_btn.setText("● REC to WAV")

    def toggle_photic(self, state): self.engine.photic_enabled = bool(state)
    def toggle_ganzfeld(self, state): self.engine.ganzfeld_enabled = bool(state)
    def toggle_glitch(self, state): self.engine.glitch_enabled = bool(state)
    def toggle_crossfeed(self, state): self.engine.crossfeed_enabled = bool(state)
    def toggle_subliminal(self, state): self.engine.subliminal_enabled = bool(state)

    def update_subliminal_text(self, text): self.engine.subliminal_text = text

    def toggle_terror(self, state):
        self.engine.is_terror_mode = bool(state)
        if self.engine.is_terror_mode and self.engine.is_playing:
            self.startle_timer.start(random.randint(8000, 20000))
        else: self.startle_timer.stop()
        if self.preset_combo.currentText() != "Custom": self._manual_override()

    def toggle_drift(self, state):
        self.engine.auto_drift_enabled = bool(state)
        if self.engine.auto_drift_enabled and self.engine.is_playing: self.drift_timer.start(3000)
        else: self.drift_timer.stop()
        if self.preset_combo.currentText() != "Custom": self._manual_override()

    def toggle_ascend(self, state):
        self.engine.auto_ascending_enabled = bool(state)
        if self.engine.auto_ascending_enabled and self.engine.is_playing: self.drift_timer.start(3000)
        else: self.drift_timer.stop()
        if self.preset_combo.currentText() != "Custom": self._manual_override()

    def process_drift(self):
        if self.engine.auto_drift_enabled:
            if self.engine.beat_freq > 2.0: self.brainwave_slider.setValue(int((self.engine.beat_freq - 0.1) * 10))
        elif self.engine.auto_ascending_enabled:
            if self.engine.beat_freq < 40.0: self.brainwave_slider.setValue(int((self.engine.beat_freq + 0.2) * 10))

    def _set_visual_color(self, hex_color, is_zen=False, is_terror=False):
        self.is_zen_mode = is_zen
        self.is_terror_mode = is_terror
        color = QColor(hex_color)
        r, g, b, a = color.getRgb()
        self.glow_curve.setPen(pg.mkPen(color=(r, g, b, 0 if is_zen else 50), width=12))
        self.main_curve.setPen(pg.mkPen(color=hex_color, width=0 if is_zen else 2))
        self.fft_curve.setPen(pg.mkPen(color=hex_color, width=2))
        self.phyllo_scatter.setBrush(pg.mkBrush(r, g, b, 200))
        self.vortex_scatter.setBrush(pg.mkBrush(r, g, b, 150))
        
        if is_terror:
            self.setStyleSheet("background-color: #100000; color: #eeeeee;")
            self.title.setStyleSheet("font-size: 22px; font-weight: bold; color: #ff3333;")
            self.plot_widget.setBackground('#050000')
            self.fft_widget.setBackground('#0a0000')
        else:
            self.setStyleSheet("background-color: #0a0a0a; color: #eeeeee;")
            self.title.setStyleSheet("font-size: 22px; font-weight: bold; color: #00d2fc;")
            self.plot_widget.setBackground('#050505')
            self.fft_widget.setBackground('#0a0a0a')

    def apply_preset(self, preset_name):
        state, tone, soundscape, drone, bilateral, breathing, drift, terror, photic, ganz, glitch, subliminal, ascend, color = self.presets[preset_name]
        
        for combo, value, handler in [
            (self.state_combo, state, self.change_state), (self.tone_combo, tone, self.change_tone),
            (self.soundscape_combo, soundscape, self.change_soundscape), (self.drone_combo, drone, self.change_drone)
        ]:
            combo.blockSignals(True)
            if value: combo.setCurrentText(value)
            combo.blockSignals(False)

        for check, val, engine_attr in [
            (self.bilateral_check, bilateral, "bilateral_enabled"), (self.breathing_check, breathing, "breathing_enabled"),
            (self.drift_check, drift, "auto_drift_enabled"), (self.terror_check, terror, "is_terror_mode"),
            (self.photic_check, photic, "photic_enabled"), (self.ganzfeld_check, ganz, "ganzfeld_enabled"),
            (self.glitch_check, glitch, "glitch_enabled"), (self.subliminal_check, subliminal, "subliminal_enabled"),
            (self.ascend_check, ascend, "auto_ascending_enabled")
        ]:
            check.blockSignals(True)
            check.setChecked(val)
            check.blockSignals(False)
            setattr(self.engine, engine_attr, val)

        if (drift or ascend) and self.engine.is_playing: self.drift_timer.start(3000)
        else: self.drift_timer.stop()
        if terror and self.engine.is_playing: self.startle_timer.start(random.randint(8000, 20000))
        else: self.startle_timer.stop()
        
        self._set_visual_color(color, is_zen=(preset_name == "Force Calm (Auto-Drift)" or preset_name == "Deep Sleep"), is_terror=terror)

    def _manual_override(self):
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentText("Custom")
        self.preset_combo.blockSignals(False)
        _, _, _, _, _, _, _, _, _, _, _, _, _, color = self.presets["Custom"]
        self._set_visual_color(color)

    def change_state(self, text):
        self.engine.current_state = text
        self.engine.beat_freq = self.engine.states[text]
        self.brainwave_slider.setValue(int(self.engine.beat_freq * 10))
        if self.preset_combo.currentText() != "Custom": self._manual_override()

    def change_tone(self, text):
        self.engine.current_tone = text
        self.engine.carrier_freq = self.engine.tones[text]
        self.carrier_input.setText(str(self.engine.carrier_freq))
        if self.preset_combo.currentText() != "Custom": self._manual_override()

    def change_soundscape(self, text): self.engine.soundscape = text
    def change_drone(self, text): self.engine.drone_mode = text
    def toggle_bilateral(self, state): self.engine.bilateral_enabled = bool(state)
    def toggle_breathing(self, state): self.engine.breathing_enabled = bool(state)

    def change_brainwave_hz(self, val):
        hz = val / 10.0
        self.engine.beat_freq = hz
        self.brainwave_label.setText(f"{hz:.1f} Hz")
        if self.preset_combo.currentText() != "Custom": self._manual_override()

    def change_mode(self, text): self.engine.mode = text
    def change_volume(self, val):
        self.engine.volume = val / 100.0
        self.engine.target_volume = self.engine.volume
        self.vol_label.setText(f"{val}%")

    def toggle_playback(self):
        if self.play_btn.isChecked():
            self.engine.target_volume = self.vol_slider.value() / 100.0
            self.engine.volume = self.engine.target_volume
            self.engine.start()
            self.session_start_time = time.time()
            self.play_btn.setText("⏹  STOP SESSION")
            if self.engine.auto_drift_enabled or self.engine.auto_ascending_enabled: self.drift_timer.start(3000)
            if self.engine.is_terror_mode: self.startle_timer.start(random.randint(8000, 20000))
        else:
            self.engine.stop()
            self.session_start_time = None
            self.timer_label.setText("Session: 00:00")
            self.play_btn.setText("▶  START SESSION")
            self.drift_timer.stop()
            self.startle_timer.stop()
            self.breath_text.setText("")
            self.strobe_overlay.setStyleSheet("background-color: transparent;")

    def get_color(self, base_color):
        if not self.chroma_flow:
            return base_color
        speed = 2.0 if self.is_terror_mode else 0.1
        h = (time.time() * speed) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        return QColor(int(r*255), int(g*255), int(b*255))

    def update_ui(self):
        with self.engine.lock:
            data = self.engine.buffer.copy()

        current_qcolor = self.get_color(QColor("#ff3333" if self.is_terror_mode else "#00d2fc"))
        r, g, b, a = current_qcolor.getRgb()
        
        amp = np.mean(np.abs(data)) if data.size > 0 else 0.0
        dynamic_thickness = 2 + int(amp * 30)
        dynamic_glow = 8 + int(amp * 40)
        dynamic_alpha_glow = int(30 + amp * 150)

        if data.size > 0:
            mode = self.engine.visual_mode
            
            # FFT Data needed for several visualizers
            fft_data = np.fft.fft(data[:, 0] + data[:, 1])
            magnitude = 20 * np.log10(np.abs(fft_data[:len(fft_data)//2]) + 1e-6)
            freqs = np.fft.fftfreq(len(data), 1.0/self.engine.sample_rate)[:len(fft_data)//2]
            self.fft_curve.setData(freqs, magnitude, pen=pg.mkPen(color=current_qcolor, width=2))

            if mode == "Lissajous" or mode == "Hypno-Tunnel":
                if self.is_terror_mode: data += np.random.randn(*data.shape) * 0.05 
                
                if mode == "Lissajous":
                    x_data = data[:, 0] * 0.8
                    y_data = data[:, 1] * 0.8
                else:
                    theta = np.linspace(0, 2*np.pi, len(data))
                    r_base = 0.2 + np.abs(data[:, 0]) * 0.5
                    x_base = r_base * np.cos(theta + self.engine.phase * 0.01)
                    y_base = r_base * np.sin(theta + self.engine.phase * 0.01)
                    x_data, y_data = [], []
                    for i in range(6):
                        angle = i * (np.pi / 3)
                        xr = x_base * np.cos(angle) - y_base * np.sin(angle)
                        yr = x_base * np.sin(angle) + y_base * np.cos(angle)
                        x_data.extend(xr)
                        y_data.extend(yr)

                if self.trails_enabled:
                    self.trail_history.append((x_data, y_data))
                    if len(self.trail_history) > 15: self.trail_history.pop(0)
                    # Draw trails backwards
                    for i in range(len(self.trail_history)-1, -1, -1):
                        trail_alpha = int(dynamic_alpha_glow * (i / len(self.trail_history)))
                        self.glow_curve.setData(self.trail_history[i][0], self.trail_history[i][1], pen=pg.mkPen(color=(r, g, b, trail_alpha), width=dynamic_glow))
                else:
                    self.glow_curve.setData(x_data, y_data, pen=pg.mkPen(color=(r, g, b, dynamic_alpha_glow), width=dynamic_glow))
                self.main_curve.setData(x_data, y_data, pen=pg.mkPen(color=current_qcolor, width=dynamic_thickness))
                
            elif mode == "Mandala Engine":
                # Split FFT into 3 bands: Bass, Mid, Treble
                # Note: freqs array maxes out around 22050Hz
                bass_idx = freqs < 250
                mid_idx = (freqs >= 250) & (freqs < 2000)
                treble_idx = freqs >= 2000
                
                bass_amp = np.mean(magnitude[bass_idx]) / 60.0 + 0.1
                mid_amp = np.mean(magnitude[mid_idx]) / 60.0 + 0.1
                treble_amp = np.mean(magnitude[treble_idx]) / 60.0 + 0.1
                
                t = time.time()
                
                def draw_polygon(n_sides, radius, rotation, curve, color):
                    angles = np.linspace(0, 2*np.pi, n_sides + 1) + rotation
                    x = np.cos(angles) * radius
                    y = np.sin(angles) * radius
                    curve.setData(x, y, pen=pg.mkPen(color=color, width=2))
                
                draw_polygon(3, 0.3 * bass_amp, t * 0.5, self.main_curve, current_qcolor)
                draw_polygon(4, 0.5 * mid_amp, -t * 0.3, self.layer2_curve, QColor(r, g, b, 150))
                draw_polygon(5, 0.7 * treble_amp, t * 0.1, self.layer3_curve, QColor(r, g, b, 100))
                
                self.glow_curve.setData([], [])
                
            elif mode == "Radial Spectrum":
                # Use first 256 bins of FFT, mirror them to make 512 points
                spec = magnitude[:256]
                spec = np.concatenate([spec, spec[::-1]])
                # Normalize 0-1
                spec_norm = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
                
                theta = np.linspace(0, 2*np.pi, len(spec_norm))
                r_base = 0.2 + spec_norm * 0.7
                
                x = r_base * np.cos(theta + self.engine.phase * 0.005)
                y = r_base * np.sin(theta + self.engine.phase * 0.005)
                
                self.glow_curve.setData(x, y, pen=pg.mkPen(color=(r, g, b, dynamic_alpha_glow), width=dynamic_glow))
                self.main_curve.setData(x, y, pen=pg.mkPen(color=current_qcolor, width=dynamic_thickness))
                self.layer2_curve.setData([], [])
                self.layer3_curve.setData([], [])

            elif mode == "Sacred Geometry":
                n = 100
                theta = n * np.arange(n) * np.deg2rad(137.5)
                amp_geom = np.mean(np.abs(data)) * 50.0
                r = amp_geom * np.sqrt(np.arange(n))
                x = r * np.cos(theta + self.engine.phase * 0.01)
                y = r * np.sin(theta + self.engine.phase * 0.01)
                self.phyllo_scatter.setData(x, y)
                
            elif mode == "Particle Vortex":
                dist = np.sqrt(self.vortex_pos[:,0]**2 + self.vortex_pos[:,1]**2) + 1e-5
                tan_x = -self.vortex_pos[:,1] / dist
                tan_y = self.vortex_pos[:,0] / dist
                
                push_x = self.vortex_pos[:,0] / dist
                push_y = self.vortex_pos[:,1] / dist
                audio_force = (data[::len(data)//1000, 0] * 0.1).reshape(-1, 1) if len(data) >= 1000 else np.zeros((1000, 1))
                
                self.vortex_vel[:, 0] += tan_x * 0.001 + push_x * audio_force[:, 0]
                self.vortex_vel[:, 1] += tan_y * 0.001 + push_y * audio_force[:, 0]
                
                # Mouse Repulsion
                dx = self.vortex_pos[:, 0] - self.mouse_pos[0]
                dy = self.vortex_pos[:, 1] - self.mouse_pos[1]
                m_dist = np.sqrt(dx**2 + dy**2) + 1e-5
                repel_mask = m_dist < 0.2
                self.vortex_vel[repel_mask, 0] += (dx[repel_mask] / m_dist[repel_mask]) * 0.02
                self.vortex_vel[repel_mask, 1] += (dy[repel_mask] / m_dist[repel_mask]) * 0.02
                
                self.vortex_vel *= 0.95
                self.vortex_pos += self.vortex_vel
                
                out_of_bounds = dist > 0.9
                self.vortex_pos[out_of_bounds] *= 0.8
                self.vortex_vel[out_of_bounds] *= -0.5
                
                if self.is_terror_mode:
                    self.vortex_pos += np.random.randn(1000, 2) * 0.005 
                
                self.vortex_scatter.setData(self.vortex_pos[:,0], self.vortex_pos[:,1], brush=pg.mkBrush(r, g, b, 150))

        if self.engine.visual_mode == "Zen Ripples" and self.engine.is_playing:
            t = time.time()
            for i, circle in enumerate(self.zen_ripples):
                phase = (t * 0.2 + i * 0.2) % 1.0
                size = phase * 2.0
                circle.setRect(-size/2, -size/2, size, size)
                alpha = int(150 * (1.0 - phase))
                circle.setPen(pg.mkPen(color=(r, g, b, alpha), width=2))

        if self.engine.is_playing:
            if self.engine.ganzfeld_enabled:
                t = time.time()
                r_g = 0.5 + 0.5 * np.sin(t * 0.1)
                g_g = 0.5 + 0.5 * np.sin(t * 0.1 + 2.0)
                b_g = 0.5 + 0.5 * np.sin(t * 0.1 + 4.0)
                self.strobe_overlay.setStyleSheet(f"background-color: rgba({int(r_g*255)}, {int(g_g*255)}, {int(b_g*255)}, 220);")
            elif self.engine.photic_enabled:
                cycle = (time.time() * self.engine.beat_freq) % 1.0
                if cycle < 0.5: 
                    if self.is_terror_mode: self.strobe_overlay.setStyleSheet("background-color: rgba(255, 0, 0, 150);")
                    else: self.strobe_overlay.setStyleSheet("background-color: rgba(0, 210, 252, 100);")
                else: self.strobe_overlay.setStyleSheet("background-color: transparent;")
            else: self.strobe_overlay.setStyleSheet("background-color: transparent;")
        elif self.strobe_overlay.styleSheet() != "background-color: transparent;":
            self.strobe_overlay.setStyleSheet("background-color: transparent;")

        if self.engine.breathing_enabled and self.engine.is_playing:
            i, h1, e, h2 = self.engine.breath_patterns[self.engine.current_breath_pattern]
            total = i + h1 + e + h2
            cycle_t = time.time() % total
            
            if cycle_t < i: 
                scale = 10 + (cycle_t / i) * 90 if i > 0 else 100
                self.breath_text.setText("Inhale...", )
            elif cycle_t < i + h1: 
                scale = 100
                self.breath_text.setText("Hold...", )
            elif cycle_t < i + h1 + e: 
                scale = 100 - ((cycle_t - (i + h1)) / e) * 90 if e > 0 else 10
                self.breath_text.setText("Exhale...", )
            else: 
                scale = 10
                self.breath_text.setText("Hold...", )
            self.breath_circle.setSize(scale)
            self.breath_circle.setData([0], [0])
        else:
            self.breath_circle.setData([], [])
            if self.engine.is_playing: self.breath_text.setText("")

        if self.session_start_time:
            elapsed = int(time.time() - self.session_start_time)
            mins = elapsed // 60
            secs = elapsed % 60
            self.timer_label.setText(f"Session: {mins:02d}:{secs:02d}")

    def resizeEvent(self, event):
        self.strobe_overlay.setGeometry(self.rect())
        super().resizeEvent(event)

    def closeEvent(self, event):
        if self.engine.is_recording: self.engine.stop_recording()
        self.engine.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisualizerWindow()
    window.show()
    sys.exit(app.exec())
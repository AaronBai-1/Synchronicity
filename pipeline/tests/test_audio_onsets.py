"""Audio-onset hit proposer tests (perception/audio_onsets.py).

Synthetic WAVs written with numpy + stdlib wave stand in for broadcast audio: decaying
white-noise bursts at known times are the "racket impacts". The tests pin the plan's
trust behaviours — abstention on silent/empty/too-short audio, loud errors on
wrong-format files (labeling against the wrong audio corrupts ground truth), strength
as relative salience only — plus the frame-mapping accuracy the propose-then-confirm
workflow needs (±1 video frame at 30 fps) and the pure jump helpers behind the o/O
keys in tools/label_rallies.py. ffmpeg-dependent extraction tests are skipped when
ffmpeg is not on PATH so the suite stays offline-green.
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from synchro_pipeline.perception.audio_onsets import (
    AudioExtractionError,
    OnsetEvent,
    detect_onsets,
    extract_audio,
    next_onset,
    onset_near,
    prev_onset,
)

FPS = 30.0
SR = 16_000


def write_wav(path: Path, samples: np.ndarray, *, sr: int = SR, channels: int = 1) -> Path:
    """Write float samples in [-1, 1] as a 16-bit PCM WAV (mono, or tiled to stereo)."""
    ints = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    if channels > 1:
        ints = np.repeat(ints[:, None], channels, axis=1).reshape(-1)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(ints.tobytes())
    return path


def burst_signal(
    duration_s: float,
    burst_times: list[tuple[float, float]],
    *,
    noise: float = 0.002,
    burst_len: int = 48,
    seed: int = 0,
) -> np.ndarray:
    """Low gaussian noise plus decaying broadband bursts at (time_s, amplitude)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, noise, int(duration_s * SR))
    envelope = np.exp(-np.arange(burst_len) / 8.0)
    for t, amp in burst_times:
        start = int(t * SR)
        x[start : start + burst_len] += amp * envelope * rng.normal(0.0, 1.0, burst_len)
    return x


class TestOnsetEventModel:
    def test_fields(self):
        event = OnsetEvent(frame=12, strength=7.5)
        assert event.frame == 12
        assert event.strength == 7.5

    def test_rejects_bad_values(self):
        with pytest.raises(ValidationError):
            OnsetEvent(frame=-1, strength=1.0)
        with pytest.raises(ValidationError):
            OnsetEvent(frame=0, strength=-0.1)
        with pytest.raises(ValidationError):
            OnsetEvent(frame=0, strength=1.0, probability=0.9)  # extra=forbid


class TestDetectOnsets:
    def test_bursts_found_within_one_video_frame(self, tmp_path):
        times = [0.5, 1.2, 2.4]
        wav = write_wav(tmp_path / "a.wav", burst_signal(3.0, [(t, 0.8) for t in times]))
        onsets = detect_onsets(wav, FPS)
        assert len(onsets) == len(times)
        for t, onset in zip(times, onsets, strict=True):
            assert abs(onset.frame - round(t * FPS)) <= 1
            assert onset.strength > 0.0

    def test_min_gap_suppresses_double_transient(self, tmp_path):
        # Two transients 45 ms apart — one physical impact's attack + racket ring,
        # closer than any two real hits. Default 120 ms gap must merge them.
        wav = write_wav(tmp_path / "a.wav", burst_signal(2.0, [(1.0, 0.8), (1.045, 0.6)]))
        onsets = detect_onsets(wav, FPS)
        assert len(onsets) == 1
        assert abs(onsets[0].frame - 30) <= 2

    def test_small_min_gap_keeps_both_transients(self, tmp_path):
        wav = write_wav(tmp_path / "a.wav", burst_signal(2.0, [(1.0, 0.8), (1.045, 0.6)]))
        onsets = detect_onsets(wav, FPS, min_gap_ms=20.0)
        assert len(onsets) == 2

    def test_pure_noise_proposes_nearly_nothing(self, tmp_path):
        rng = np.random.default_rng(7)
        wav = write_wav(tmp_path / "a.wav", rng.normal(0.0, 0.05, int(3.0 * SR)))
        assert len(detect_onsets(wav, FPS)) <= 2

    def test_silent_wav_returns_empty(self, tmp_path):
        wav = write_wav(tmp_path / "a.wav", np.zeros(SR))
        assert detect_onsets(wav, FPS) == []

    def test_empty_wav_returns_empty(self, tmp_path):
        wav = write_wav(tmp_path / "a.wav", np.zeros(0))
        assert detect_onsets(wav, FPS) == []

    def test_too_short_audio_abstains(self, tmp_path):
        # Under two analysis windows (default 20 ms win + 10 ms hop = 480 samples).
        rng = np.random.default_rng(0)
        wav = write_wav(tmp_path / "a.wav", rng.normal(0.0, 0.1, 100))
        assert detect_onsets(wav, FPS) == []

    def test_strength_ranks_louder_burst_higher(self, tmp_path):
        wav = write_wav(tmp_path / "a.wav", burst_signal(2.5, [(0.5, 0.9), (1.5, 0.15)]))
        onsets = detect_onsets(wav, FPS)
        assert len(onsets) == 2
        assert onsets[0].strength > onsets[1].strength

    def test_native_rate_respected(self, tmp_path):
        # Same burst at 44.1 kHz: no resampling happens, times map straight to frames.
        rng = np.random.default_rng(1)
        sr = 44_100
        x = rng.normal(0.0, 0.002, int(2.0 * sr))
        x[int(1.0 * sr) : int(1.0 * sr) + 60] += 0.8 * rng.normal(0.0, 1.0, 60)
        wav = write_wav(tmp_path / "a.wav", x, sr=sr)
        onsets = detect_onsets(wav, FPS)
        assert len(onsets) == 1
        assert abs(onsets[0].frame - 30) <= 1

    def test_stereo_rejected(self, tmp_path):
        wav = write_wav(tmp_path / "a.wav", np.zeros(SR), channels=2)
        with pytest.raises(ValueError, match="mono"):
            detect_onsets(wav, FPS)

    def test_8bit_rejected(self, tmp_path):
        path = tmp_path / "a.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(1)
            wav.setframerate(SR)
            wav.writeframes(bytes(SR))
        with pytest.raises(ValueError, match="16-bit"):
            detect_onsets(path, FPS)

    def test_garbage_file_rejected(self, tmp_path):
        path = tmp_path / "a.wav"
        path.write_bytes(b"this is not a wav file")
        with pytest.raises(ValueError, match="not a readable PCM WAV"):
            detect_onsets(path, FPS)

    def test_bad_parameters_rejected(self, tmp_path):
        wav = write_wav(tmp_path / "a.wav", np.zeros(SR))
        with pytest.raises(ValueError, match="video_fps"):
            detect_onsets(wav, 0.0)
        with pytest.raises(ValueError, match="frame_ms"):
            detect_onsets(wav, FPS, frame_ms=0.0)
        with pytest.raises(ValueError, match="non-negative"):
            detect_onsets(wav, FPS, k_mad=-1.0)


class TestJumpHelpers:
    ONSETS = [
        OnsetEvent(frame=40, strength=2.0),  # deliberately unsorted
        OnsetEvent(frame=10, strength=1.0),
        OnsetEvent(frame=90, strength=3.0),
    ]

    def test_next_onset(self):
        assert next_onset(self.ONSETS, 0).frame == 10
        assert next_onset(self.ONSETS, 10).frame == 40  # strictly after
        assert next_onset(self.ONSETS, 89).frame == 90
        assert next_onset(self.ONSETS, 90) is None
        assert next_onset([], 0) is None

    def test_prev_onset(self):
        assert prev_onset(self.ONSETS, 100).frame == 90
        assert prev_onset(self.ONSETS, 40).frame == 10  # strictly before
        assert prev_onset(self.ONSETS, 10) is None
        assert prev_onset([], 10) is None

    def test_onset_near(self):
        assert onset_near(self.ONSETS, 12).frame == 10  # default tol ±2
        assert onset_near(self.ONSETS, 13) is None
        assert onset_near(self.ONSETS, 13, tol_frames=3).frame == 10
        assert onset_near([], 10) is None

    def test_onset_near_tie_prefers_earlier(self):
        onsets = [OnsetEvent(frame=14, strength=1.0), OnsetEvent(frame=10, strength=1.0)]
        assert onset_near(onsets, 12, tol_frames=2).frame == 10


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
class TestExtractAudio:
    def _make_video(self, tmp_path: Path, audio_wav: Path | None) -> Path:
        video = tmp_path / "clip.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=128x72:rate=30:duration=2",
        ]
        if audio_wav is not None:
            cmd += ["-i", str(audio_wav), "-c:a", "aac", "-shortest"]
        cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(video)]
        subprocess.run(cmd, check=True, capture_output=True)
        return video

    def test_extracted_burst_found(self, tmp_path):
        # Burst at t=1.0 muxed into a tiny video; extraction + detection round-trips
        # to the right video frame (±2 frames allows the AAC encode/decode delay).
        source = write_wav(tmp_path / "src.wav", burst_signal(2.0, [(1.0, 0.8)]))
        video = self._make_video(tmp_path, source)
        wav = extract_audio(video, tmp_path / "extracted.wav")
        onsets = detect_onsets(wav, FPS)
        assert any(abs(o.frame - 30) <= 2 for o in onsets)

    def test_video_without_audio_stream_raises(self, tmp_path):
        video = self._make_video(tmp_path, None)
        with pytest.raises(AudioExtractionError):
            extract_audio(video, tmp_path / "extracted.wav")

    def test_missing_video_raises(self, tmp_path):
        with pytest.raises(AudioExtractionError):
            extract_audio(tmp_path / "nope.mp4", tmp_path / "extracted.wav")

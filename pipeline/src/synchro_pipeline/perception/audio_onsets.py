"""Audio-onset hit proposer for golden-set labeling (docs/plan.md Phase 0 + risk #1).

Racket impacts are sharp broadband transients, so a spectral-flux novelty curve over
the S0 audio artifact (mono 16 kHz WAV — s0_ingest.py extracts it "for the eventual S5
audio hit-cue experiments") localizes candidate hit frames far faster than a human
scrubbing video. tools/label_rallies.py uses these proposals to turn hit labeling from
scrub-and-hunt into confirm-or-reject: the labeller jumps between proposed frames
(o/O) and presses n/f — confirmation is ~3-5x faster than search, which is how the
plan's "~2-3 person-days per match" golden-set budget comes down.

Trust posture (plan: every automated output is reviewable before it becomes ground
truth): proposals are NEVER written into the golden JSON. A human confirms every hit
frame by eye; this module only decides where the labeller looks next. That is why
false positives are acceptable here in a way they are not in production S5:

    KNOWN LIMITATION — broadcast audio is polluted. Crowd claps, score beeps, line
    calls, and commentary plosives are also sharp transients and WILL be proposed as
    onsets; sustained commentary/music raises the noise floor, which the median+MAD
    adaptive threshold absorbs, but discrete non-racket transients it cannot (and
    should not — a detector tuned to reject everything unusual would also reject
    unusual hits). The labeller rejects them with a keypress.

Implementation constraints: numpy + stdlib ``wave`` only — no scipy/librosa (installed
deps are pinned; this must import and test offline). No resampling: the novelty curve
works at the WAV's native rate and only the onset *times* are mapped to video frames.
"""

from __future__ import annotations

import subprocess
import wave
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# Mirrors stages/s0_ingest.py AUDIO_SAMPLE_RATE (not imported: perception must not
# depend on the stage layer). Only extract_audio uses it — detect_onsets reads the
# WAV's own rate.
AUDIO_SAMPLE_RATE = 16_000

# Frames per rFFT batch in _spectral_flux: bounds peak memory to a few MB so a full
# match hour (~360k hops at 16 kHz / 10 ms) never materializes a giant frame matrix.
_CHUNK_FRAMES = 4096

# Absolute flux floor separating real audio transients from rFFT numerical noise on
# degenerate (near-constant) signals. Normalized samples put genuine transient flux
# orders of magnitude above this; float32 FFT jitter sits orders of magnitude below.
_NUMERIC_FLOOR = 1e-6


class AudioExtractionError(RuntimeError):
    """ffmpeg audio extraction failed (missing binary, no audio stream, bad file)."""


class OnsetEvent(BaseModel):
    """One proposed audio onset, mapped to a video frame.

    `strength` is relative onset salience — the onset's spectral flux above the
    recording's median, in MAD units (i.e. how far it stands above this recording's
    own noise floor). It is NOT a probability and is only comparable within the
    recording that produced it; use it to rank proposals, never to threshold across
    matches (same contract as hits_baseline.ProposedHit.saliency — that lesson is
    deliberately repeated here).
    """

    model_config = ConfigDict(extra="forbid")

    frame: int = Field(ge=0)
    strength: float = Field(ge=0.0)


def _read_mono_16bit(wav_path: str | Path) -> tuple[np.ndarray, int]:
    """Read a mono 16-bit PCM WAV as normalized float32 samples plus its sample rate.

    Format violations raise ValueError with an actionable message — the S0 artifact
    contract is mono/16 kHz/pcm_s16le, so anything else means the wrong file was
    passed, not a case to silently downmix (guessing a channel mix could shift
    transient timing evidence).
    """
    try:
        with wave.open(str(wav_path), "rb") as wav:
            n_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            n_frames = wav.getnframes()
            raw = wav.readframes(n_frames)
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"{wav_path}: not a readable PCM WAV: {exc}") from exc
    if n_channels != 1:
        raise ValueError(
            f"{wav_path}: expected mono audio (the S0 contract), got {n_channels} channels"
        )
    if sample_width != 2:
        raise ValueError(
            f"{wav_path}: expected 16-bit PCM (the S0 contract), got {8 * sample_width}-bit"
        )
    if sample_rate <= 0:
        raise ValueError(f"{wav_path}: implausible sample rate {sample_rate}")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return samples, sample_rate


def _spectral_flux(samples: np.ndarray, win_len: int, hop_len: int) -> np.ndarray:
    """Half-wave-rectified spectral-flux novelty per hop; flux[0] is 0 by definition.

    flux[t] = sum over rFFT bins of max(0, |X_t| - |X_{t-1}|) on Hann-windowed frames
    — energy *appearing* in the spectrum, which is what a broadband transient does.
    Processed in _CHUNK_FRAMES batches so memory stays flat over match-length audio.
    """
    n_frames = 1 + (samples.size - win_len) // hop_len
    window = np.hanning(win_len).astype(np.float32)
    flux = np.zeros(n_frames, dtype=np.float64)
    prev_mag: np.ndarray | None = None
    for start in range(0, n_frames, _CHUNK_FRAMES):
        stop = min(start + _CHUNK_FRAMES, n_frames)
        idx = np.arange(start, stop)[:, None] * hop_len + np.arange(win_len)[None, :]
        mag = np.abs(np.fft.rfft(samples[idx] * window, axis=1))
        if prev_mag is None:  # first chunk: flux[0] has no predecessor, stays 0
            flux[start + 1 : stop] = np.maximum(np.diff(mag, axis=0), 0.0).sum(axis=1)
        else:
            shifted = np.vstack([prev_mag[None, :], mag[:-1]])
            flux[start:stop] = np.maximum(mag - shifted, 0.0).sum(axis=1)
        prev_mag = mag[-1].copy()
    return flux


def detect_onsets(
    wav_path: str | Path,
    video_fps: float,
    *,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    min_gap_ms: float = 120.0,
    k_mad: float = 6.0,
) -> list[OnsetEvent]:
    """Propose candidate hit frames from a mono 16-bit WAV's spectral-flux onsets.

    Pipeline: Hann-windowed rFFT frames (`frame_ms` long, every `hop_ms`) →
    half-wave-rectified spectral flux → adaptive threshold at median + `k_mad` * MAD
    over the WHOLE recording (median/MAD, not mean/std, because the flux distribution
    of broadcast audio is heavy-tailed — the hits themselves would inflate a mean) →
    local-maximum peaks → greedy strongest-first suppression so no two onsets are
    within `min_gap_ms` (two real hits cannot be closer than a swing cycle; a single
    impact's attack+ring must not propose twice). Onset time is the peak window's
    CENTRE (Hann weighting means flux peaks when the transient is mid-window), mapped
    to a frame as round(t * video_fps).

    Abstention over guessing (plan trust architecture): empty or silent audio, or
    audio too short to fill two analysis windows, returns [] — never a fabricated
    proposal. The default `k_mad`=6 is deliberately conservative: with propose-then-
    confirm a missed quiet hit costs one manual scrub, while a flood of noise-floor
    proposals costs every rally.
    """
    if video_fps <= 0:
        raise ValueError("video_fps must be positive")
    if frame_ms <= 0 or hop_ms <= 0:
        raise ValueError("frame_ms and hop_ms must be positive")
    if min_gap_ms < 0 or k_mad < 0:
        raise ValueError("min_gap_ms and k_mad must be non-negative")

    samples, sample_rate = _read_mono_16bit(wav_path)
    win_len = max(1, round(sample_rate * frame_ms / 1000.0))
    hop_len = max(1, round(sample_rate * hop_ms / 1000.0))
    if samples.size < win_len + hop_len:  # fewer than two windows: nothing to compare
        return []

    flux = _spectral_flux(samples, win_len, hop_len)
    median = float(np.median(flux))
    mad = float(np.median(np.abs(flux - median)))
    threshold = median + k_mad * mad

    n = flux.size
    candidates = [
        i
        for i in range(1, n)
        if flux[i] > threshold
        and flux[i] > _NUMERIC_FLOOR
        and flux[i] >= flux[i - 1]
        and (i == n - 1 or flux[i] >= flux[i + 1])
    ]

    # Greedy suppression: strongest first, deterministic tie-break on time (mirrors
    # hits_baseline.propose_hits — same reasoning, different signal).
    times = (np.arange(n) * hop_len + win_len / 2.0) / sample_rate
    gap_s = min_gap_ms / 1000.0
    kept: list[int] = []
    for i in sorted(candidates, key=lambda i: (-flux[i], i)):
        if all(abs(float(times[i] - times[j])) > gap_s for j in kept):
            kept.append(i)

    # Map to video frames; if two hops land on the same frame keep the stronger.
    scale = mad if mad > 0 else 1.0  # degenerate MAD: report raw excess, still >= 0
    by_frame: dict[int, float] = {}
    for i in kept:
        frame = round(float(times[i]) * video_fps)
        strength = (float(flux[i]) - median) / scale
        by_frame[frame] = max(by_frame.get(frame, 0.0), strength)
    return [
        OnsetEvent(frame=frame, strength=strength)
        for frame, strength in sorted(by_frame.items())
    ]


# --- GUI-facing pure helpers (tools/label_rallies.py key handlers) -------------------


def next_onset(onsets: Sequence[OnsetEvent], frame: int) -> OnsetEvent | None:
    """The earliest onset strictly after `frame`, or None (end of proposals)."""
    later = [o for o in onsets if o.frame > frame]
    return min(later, key=lambda o: o.frame) if later else None


def prev_onset(onsets: Sequence[OnsetEvent], frame: int) -> OnsetEvent | None:
    """The latest onset strictly before `frame`, or None (start of proposals)."""
    earlier = [o for o in onsets if o.frame < frame]
    return max(earlier, key=lambda o: o.frame) if earlier else None


def onset_near(
    onsets: Sequence[OnsetEvent], frame: int, tol_frames: int = 2
) -> OnsetEvent | None:
    """The closest onset within ±`tol_frames` of `frame` (ties → earlier), or None.

    Backs the HUD marker: audio-to-video rounding is ±1 frame at best, so "you are on
    a proposed onset" must mean a small window, not exact equality.
    """
    near = [o for o in onsets if abs(o.frame - frame) <= tol_frames]
    return min(near, key=lambda o: (abs(o.frame - frame), o.frame)) if near else None


def extract_audio(video_path: str | Path, out_wav: str | Path) -> Path:
    """Extract a video's audio track as mono 16 kHz pcm_s16le WAV (the S0 command).

    Mirrors stages/s0_ingest.py's audio invocation so onsets computed from a raw
    --video match what the pipeline's own audio artifact would give. Raises
    AudioExtractionError when ffmpeg is missing, exits nonzero, or the video has no
    audio stream (the canonical mezzanine is encoded with ``-an`` — callers should
    treat this as "onsets unavailable", not a fatal labeling error).
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(out_wav),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise AudioExtractionError("ffmpeg not found on PATH") from exc
    if proc.returncode != 0:
        raise AudioExtractionError(
            f"audio extraction from {video_path} failed (exit {proc.returncode}): "
            f"{proc.stderr.strip()[:2000]}"
        )
    return Path(out_wav)

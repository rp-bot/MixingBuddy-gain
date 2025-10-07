import numpy as np
import soundfile as sf
import librosa


def db_to_linear(db: float) -> float:
    """Convert decibels to linear gain.

    0 dB -> 1.0, -6 dB -> ~0.501, +6 dB -> ~1.995.
    """
    return float(10 ** (db / 20.0))


def load_audio_chunk(
    path: str, start_sec: float, end_sec: float, sr: int
) -> np.ndarray:
    """Load an audio chunk [start_sec, end_sec) in seconds, resampled to sr and mono.

    Returns a 1D float32 numpy array of length roughly (end_sec-start_sec)*sr.
    """
    if end_sec <= start_sec:
        return np.zeros(0, dtype=np.float32)

    # Use librosa to handle offset/duration and resampling robustly
    duration = max(0.0, end_sec - start_sec)
    y, _ = librosa.load(
        path, sr=sr, mono=True, offset=float(start_sec), duration=float(duration)
    )
    return y.astype(np.float32, copy=False)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a mono 1D audio array from orig_sr to target_sr.
    If sampling rates are equal, returns input (no copy).
    """
    if orig_sr == target_sr:
        return audio
    y = librosa.resample(
        y=audio, orig_sr=int(orig_sr), target_sr=int(target_sr), res_type="kaiser_best"
    )
    return y.astype(np.float32, copy=False)


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Ensure mono. Accepts 1D (mono) or 2D (channels, samples) arrays.
    If 2D, average across channel axis 0.
    """
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    if audio.ndim == 2:
        # Expect shape (channels, samples) or (samples, channels)
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            # Likely (channels, samples)
            mono = np.mean(audio, axis=0)
        else:
            # Likely (samples, channels)
            mono = np.mean(audio, axis=1)
        return mono.astype(np.float32, copy=False)
    # Fallback: flatten
    return audio.reshape(-1).astype(np.float32, copy=False)

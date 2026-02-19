"""
Audio Encoder - Cochlea Simulation

Converts raw audio into spike trains that the SNN can process.
Mimics biological cochlea with:
- Mel-frequency filterbank (like basilar membrane)
- Hair cell adaptation
- Phase-locking for temporal precision
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


@dataclass
class CochleaConfig:
    """Configuration for cochlea encoder."""
    sample_rate: int = 16000      # Audio sample rate
    n_mels: int = 80              # Number of mel bands (frequency channels)
    hop_length: int = 160         # 10ms at 16kHz
    win_length: int = 400         # 25ms window
    fmin: float = 50.0            # Minimum frequency
    fmax: float = 8000.0          # Maximum frequency
    n_time_steps: int = 10        # Temporal context window
    spike_threshold: float = 0.3  # Threshold for spike generation
    adaptation_tau: float = 0.1   # Hair cell adaptation time constant


class CochleaEncoder:
    """
    Biologically-inspired audio to spike encoder.

    Converts audio waveform to spike trains:
    1. Mel spectrogram (frequency decomposition like basilar membrane)
    2. Temporal derivative (onset detection like hair cells)
    3. Threshold crossing (spike generation)
    4. Adaptation (habituation to sustained sounds)
    """

    def __init__(self, config: Optional[CochleaConfig] = None):
        self.config = config or CochleaConfig()
        self.adaptation_state = np.zeros(self.config.n_mels, dtype=np.float32)

    def encode_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert audio waveform to spike representation.

        Args:
            audio: Audio waveform, shape (n_samples,)

        Returns:
            spikes: Spike representation, shape (n_time, n_mels)
        """
        config = self.config

        if LIBROSA_AVAILABLE:
            # Mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=audio.astype(np.float32),
                sr=config.sample_rate,
                n_mels=config.n_mels,
                hop_length=config.hop_length,
                win_length=config.win_length,
                fmin=config.fmin,
                fmax=config.fmax
            )
            # Log compression (like cochlear nonlinearity)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_norm = (mel_db + 80) / 80  # Normalize to 0-1
        else:
            # Simple FFT-based alternative
            mel_norm = self._simple_mel(audio)

        mel_norm = mel_norm.T  # (time, freq)

        # Temporal derivative (onset detection)
        onset = np.diff(mel_norm, axis=0, prepend=mel_norm[:1])
        onset = np.maximum(0, onset)  # Only positive onsets

        # Combine sustained and onset signals
        combined = 0.7 * mel_norm + 0.3 * onset

        # Apply adaptation
        spikes = self._apply_adaptation(combined)

        return spikes.astype(np.float32)

    def _simple_mel(self, audio: np.ndarray) -> np.ndarray:
        """Simple mel spectrogram without librosa."""
        config = self.config

        # Frame the audio
        n_frames = len(audio) // config.hop_length
        frames = np.zeros((config.win_length, n_frames), dtype=np.float32)

        for i in range(n_frames):
            start = i * config.hop_length
            end = start + config.win_length
            if end <= len(audio):
                frames[:, i] = audio[start:end] * np.hanning(config.win_length)

        # FFT
        fft = np.abs(np.fft.rfft(frames, axis=0))

        # Simple mel filterbank (linear approximation)
        n_fft = fft.shape[0]
        mel_filters = np.zeros((config.n_mels, n_fft), dtype=np.float32)

        mel_points = np.linspace(
            self._hz_to_mel(config.fmin),
            self._hz_to_mel(config.fmax),
            config.n_mels + 2
        )
        hz_points = self._mel_to_hz(mel_points)
        bin_points = (hz_points * n_fft * 2 / config.sample_rate).astype(int)

        for i in range(config.n_mels):
            start, center, end = bin_points[i:i+3]
            for j in range(start, center):
                if j < n_fft:
                    mel_filters[i, j] = (j - start) / max(1, center - start)
            for j in range(center, end):
                if j < n_fft:
                    mel_filters[i, j] = (end - j) / max(1, end - center)

        mel = np.dot(mel_filters, fft)

        # Log and normalize
        mel = np.log1p(mel)
        mel = mel / (np.max(mel) + 1e-8)

        return mel

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700 * (10 ** (mel / 2595) - 1)

    def _apply_adaptation(self, signal: np.ndarray) -> np.ndarray:
        """Apply hair cell adaptation (habituation)."""
        config = self.config
        spikes = np.zeros_like(signal)

        for t in range(signal.shape[0]):
            # Compute effective input after adaptation
            effective = signal[t] - self.adaptation_state

            # Generate spikes where above threshold
            spikes[t] = (effective > config.spike_threshold).astype(np.float32)

            # Update adaptation state
            decay = np.exp(-1.0 / (config.adaptation_tau * 100))
            self.adaptation_state = decay * self.adaptation_state + (1 - decay) * signal[t]

        return spikes

    def encode_streaming(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Encode audio chunk for streaming/real-time processing.

        Returns spike vector for current time step.
        """
        spikes = self.encode_audio(audio_chunk)
        # Return last n_time_steps flattened for SNN input
        context = spikes[-self.config.n_time_steps:]
        return context.flatten()

    def reset_adaptation(self):
        """Reset adaptation state for new utterance."""
        self.adaptation_state = np.zeros(self.config.n_mels, dtype=np.float32)


class AudioProcessor:
    """
    High-level audio processing pipeline.

    Handles:
    - Microphone input
    - Audio file loading
    - Streaming processing
    - Noise reduction
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.cochlea = CochleaEncoder(CochleaConfig(sample_rate=sample_rate))

    def load_audio(self, path: str) -> np.ndarray:
        """Load audio file."""
        if LIBROSA_AVAILABLE:
            audio, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        else:
            # Try scipy as fallback
            try:
                from scipy.io import wavfile
                sr, audio = wavfile.read(path)
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                # Resample if needed
                if sr != self.sample_rate:
                    ratio = self.sample_rate / sr
                    new_len = int(len(audio) * ratio)
                    audio = np.interp(
                        np.linspace(0, len(audio), new_len),
                        np.arange(len(audio)),
                        audio
                    )
            except ImportError:
                raise ImportError("Install librosa or scipy to load audio files")

        return audio.astype(np.float32)

    def process_file(self, path: str) -> np.ndarray:
        """Process audio file to spikes."""
        audio = self.load_audio(path)
        return self.cochlea.encode_audio(audio)

    def process_youtube(self, url: str) -> np.ndarray:
        """
        Download and process audio from YouTube video.

        Returns spike representation of the audio track.
        """
        import subprocess
        import tempfile
        import os

        # Download audio using yt-dlp
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "audio.wav")

            cmd = [
                "yt-dlp",
                "-x",  # Extract audio
                "--audio-format", "wav",
                "--audio-quality", "0",
                "-o", output_path.replace(".wav", ".%(ext)s"),
                "--postprocessor-args", f"-ar {self.sample_rate} -ac 1",
                url
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True)
                # Find the output file (might have different extension)
                for f in os.listdir(tmpdir):
                    if f.endswith(".wav"):
                        output_path = os.path.join(tmpdir, f)
                        break

                audio = self.load_audio(output_path)
                return self.cochlea.encode_audio(audio)

            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to download audio: {e}")

    def process_chunk(self, audio: np.ndarray) -> np.ndarray:
        """Process audio chunk for streaming."""
        return self.cochlea.encode_streaming(audio)


class PhonemeEncoder:
    """
    Encodes phonemes as spike patterns for teaching.

    Used to provide supervised signal during phoneme learning.
    Each phoneme maps to a unique sparse spike pattern.
    """

    # International Phonetic Alphabet subset (English phonemes)
    PHONEMES = [
        # Vowels
        'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH',
        'IY', 'OW', 'OY', 'UH', 'UW',
        # Consonants
        'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L',
        'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V',
        'W', 'Y', 'Z', 'ZH',
        # Special
        'SIL',  # Silence
    ]

    def __init__(self, n_neurons: int = 1000, seed: int = 42):
        self.n_neurons = n_neurons
        self.rng = np.random.default_rng(seed)

        # Create sparse random patterns for each phoneme
        self.patterns = {}
        for phoneme in self.PHONEMES:
            # Each phoneme activates ~10% of neurons
            n_active = n_neurons // 10
            active_indices = self.rng.choice(n_neurons, n_active, replace=False)
            pattern = np.zeros(n_neurons, dtype=np.float32)
            pattern[active_indices] = 1.0
            self.patterns[phoneme] = pattern

    def encode(self, phoneme: str) -> np.ndarray:
        """Get spike pattern for phoneme."""
        return self.patterns.get(phoneme.upper(), self.patterns['SIL'])

    def decode(self, spikes: np.ndarray, threshold: float = 0.5) -> str:
        """Decode spike pattern to phoneme."""
        best_match = 'SIL'
        best_score = -1

        for phoneme, pattern in self.patterns.items():
            score = np.dot(spikes, pattern) / (np.sum(pattern) + 1e-8)
            if score > best_score:
                best_score = score
                best_match = phoneme

        return best_match if best_score > threshold else 'SIL'


if __name__ == "__main__":
    # Test cochlea encoder
    print("Testing Cochlea Encoder")
    print("=" * 50)

    # Generate test audio (440Hz sine wave)
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    encoder = CochleaEncoder()
    spikes = encoder.encode_audio(audio)

    print(f"Input: {len(audio)} samples ({duration}s at {sr}Hz)")
    print(f"Output: {spikes.shape} (time x frequency)")
    print(f"Spike rate: {spikes.mean():.2%}")

    # Test phoneme encoder
    print()
    print("Testing Phoneme Encoder")
    print("=" * 50)

    phoneme_enc = PhonemeEncoder(n_neurons=100)

    for phoneme in ['AA', 'B', 'S', 'SIL']:
        pattern = phoneme_enc.encode(phoneme)
        decoded = phoneme_enc.decode(pattern)
        print(f"'{phoneme}' -> {pattern.sum():.0f} active neurons -> decoded: '{decoded}'")

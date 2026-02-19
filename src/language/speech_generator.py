"""
Speech Generator - Neural Vocoder

Converts neural spike patterns into audio waveforms.
Learns to produce speech through:
- Motor control learning (articulatory gestures)
- Formant synthesis
- Prosody generation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class VocoderConfig:
    """Configuration for neural vocoder."""
    sample_rate: int = 16000
    n_phonemes: int = 40
    n_formants: int = 4           # F1, F2, F3, F4
    frame_size: int = 160         # 10ms at 16kHz
    n_harmonics: int = 30         # Harmonics for voiced sounds
    noise_bands: int = 20         # Noise bands for unvoiced
    f0_range: Tuple[float, float] = (80, 400)  # Fundamental frequency range


class FormantSynthesizer:
    """
    Formant-based speech synthesizer.

    Produces speech by controlling:
    - Fundamental frequency (F0/pitch)
    - Formant frequencies (vocal tract resonances)
    - Voicing (voiced vs unvoiced)
    - Amplitude
    """

    def __init__(self, config: Optional[VocoderConfig] = None):
        self.config = config or VocoderConfig()

        # Formant targets for phonemes (simplified)
        # Format: {phoneme: (F1, F2, F3, voiced)}
        self.phoneme_formants = {
            # Vowels (all voiced)
            'AA': (700, 1200, 2500, True),   # "father"
            'AE': (700, 1700, 2500, True),   # "cat"
            'AH': (600, 1200, 2500, True),   # "but"
            'AO': (500, 900, 2500, True),    # "bought"
            'EH': (550, 1800, 2500, True),   # "bed"
            'IH': (400, 2000, 2600, True),   # "bit"
            'IY': (300, 2300, 3000, True),   # "beat"
            'UH': (450, 1100, 2300, True),   # "book"
            'UW': (300, 900, 2300, True),    # "boot"

            # Consonants
            'B': (200, 1000, 2000, True),
            'D': (300, 1700, 2600, True),
            'G': (300, 2000, 2700, True),
            'P': (200, 1000, 2000, False),
            'T': (300, 1700, 2600, False),
            'K': (300, 2000, 2700, False),
            'S': (0, 4000, 6000, False),
            'Z': (200, 4000, 6000, True),
            'F': (0, 1500, 2500, False),
            'V': (200, 1500, 2500, True),
            'SH': (0, 2500, 3500, False),
            'M': (200, 1000, 2200, True),
            'N': (200, 1500, 2500, True),
            'L': (350, 1000, 2500, True),
            'R': (350, 1300, 1700, True),

            # Silence
            'SIL': (0, 0, 0, False),
        }

        # Current state
        self.phase = 0.0
        self.current_f0 = 120.0
        self.formant_state = np.zeros(4)

    def synthesize_phoneme(
        self,
        phoneme: str,
        duration_ms: float = 100,
        f0: Optional[float] = None,
        amplitude: float = 0.5
    ) -> np.ndarray:
        """
        Synthesize audio for a single phoneme.

        Args:
            phoneme: Phoneme symbol (e.g., 'AA', 'B')
            duration_ms: Duration in milliseconds
            f0: Fundamental frequency (pitch)
            amplitude: Volume level 0-1

        Returns:
            Audio waveform as numpy array
        """
        config = self.config
        n_samples = int(duration_ms * config.sample_rate / 1000)

        # Get formant targets
        formants = self.phoneme_formants.get(
            phoneme.upper(),
            self.phoneme_formants['SIL']
        )
        f1, f2, f3, voiced = formants

        if f0 is None:
            f0 = self.current_f0

        # Generate waveform
        t = np.arange(n_samples) / config.sample_rate

        if voiced and f1 > 0:
            # Voiced sound: harmonic series filtered by formants
            signal = self._generate_voiced(t, f0, f1, f2, f3)
        elif f2 > 1000:
            # High-frequency unvoiced (like 'S')
            signal = self._generate_fricative(t, f2, f3)
        else:
            # Silence or stop
            signal = np.zeros(n_samples)

        # Apply amplitude envelope
        envelope = self._generate_envelope(n_samples)
        signal = signal * envelope * amplitude

        return signal.astype(np.float32)

    def _generate_voiced(
        self,
        t: np.ndarray,
        f0: float,
        f1: float,
        f2: float,
        f3: float
    ) -> np.ndarray:
        """Generate voiced speech using harmonic synthesis."""
        signal = np.zeros_like(t)

        # Generate harmonics
        for h in range(1, self.config.n_harmonics + 1):
            freq = f0 * h
            if freq > self.config.sample_rate / 2:
                break

            # Amplitude based on formant filtering
            amp = 1.0 / h  # Natural roll-off
            amp *= self._formant_filter(freq, f1, 80)
            amp *= self._formant_filter(freq, f2, 100)
            amp *= self._formant_filter(freq, f3, 120)

            signal += amp * np.sin(2 * np.pi * freq * t + self.phase)

        # Update phase for continuity
        self.phase += 2 * np.pi * f0 * len(t) / self.config.sample_rate
        self.phase = self.phase % (2 * np.pi)

        # Normalize
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal /= max_val

        return signal

    def _generate_fricative(
        self,
        t: np.ndarray,
        center_freq: float,
        bandwidth: float
    ) -> np.ndarray:
        """Generate unvoiced fricative noise."""
        # White noise
        noise = np.random.randn(len(t))

        # Simple bandpass filter using sin modulation
        # (proper filter would use scipy, but keeping dependency-free)
        carrier = np.sin(2 * np.pi * center_freq * t)
        signal = noise * carrier

        # Smooth
        window = np.hanning(21)
        window /= window.sum()
        signal = np.convolve(signal, window, mode='same')

        return signal * 0.3  # Reduce amplitude of fricatives

    def _formant_filter(
        self,
        freq: float,
        formant_freq: float,
        bandwidth: float
    ) -> float:
        """Apply formant resonance filter."""
        if formant_freq <= 0:
            return 0.0
        # Simple Gaussian-like resonance
        diff = (freq - formant_freq) / bandwidth
        return np.exp(-0.5 * diff * diff)

    def _generate_envelope(self, n_samples: int) -> np.ndarray:
        """Generate amplitude envelope (attack-sustain-release)."""
        envelope = np.ones(n_samples)

        # Attack (5ms)
        attack_samples = min(int(0.005 * self.config.sample_rate), n_samples // 4)
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

        # Release (5ms)
        release_samples = min(int(0.005 * self.config.sample_rate), n_samples // 4)
        envelope[-release_samples:] = np.linspace(1, 0, release_samples)

        return envelope

    def synthesize_sequence(
        self,
        phonemes: List[str],
        durations: Optional[List[float]] = None,
        f0_contour: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Synthesize a sequence of phonemes.

        Args:
            phonemes: List of phoneme symbols
            durations: Duration for each phoneme (ms), default 100ms
            f0_contour: Pitch for each phoneme, default 120Hz

        Returns:
            Concatenated audio waveform
        """
        if durations is None:
            durations = [100.0] * len(phonemes)
        if f0_contour is None:
            f0_contour = [120.0] * len(phonemes)

        segments = []
        for phoneme, dur, f0 in zip(phonemes, durations, f0_contour):
            segment = self.synthesize_phoneme(phoneme, dur, f0)
            segments.append(segment)

        return np.concatenate(segments)


class NeuralVocoder:
    """
    Neural network-based vocoder.

    Learns to map spike patterns to audio parameters:
    - Phoneme -> formant targets
    - Context -> prosody (pitch, duration)
    - Emotion -> voice quality
    """

    def __init__(self, config: Optional[VocoderConfig] = None):
        self.config = config or VocoderConfig()
        self.synthesizer = FormantSynthesizer(config)

        # Neural mapping layers
        self.rng = np.random.default_rng(42)

        # Input: spike pattern from Broca's area
        self.input_size = 1000

        # Hidden layer
        self.hidden_size = 256
        self.w1 = self.rng.randn(self.hidden_size, self.input_size).astype(np.float32) * 0.1
        self.b1 = np.zeros(self.hidden_size, dtype=np.float32)

        # Output: phoneme probabilities
        self.output_size = self.config.n_phonemes
        self.w2 = self.rng.randn(self.output_size, self.hidden_size).astype(np.float32) * 0.1
        self.b2 = np.zeros(self.output_size, dtype=np.float32)

        # Phoneme labels
        self.phoneme_labels = list(self.synthesizer.phoneme_formants.keys())

        # Prosody network
        self.w_prosody = self.rng.randn(3, self.hidden_size).astype(np.float32) * 0.1

    def forward(self, spikes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through vocoder network.

        Args:
            spikes: Input spike pattern

        Returns:
            phoneme_probs: Probability distribution over phonemes
            prosody: [f0, duration, amplitude]
        """
        # Hidden layer
        h = np.maximum(0, np.dot(self.w1, spikes) + self.b1)  # ReLU

        # Phoneme output
        logits = np.dot(self.w2, h) + self.b2
        phoneme_probs = self._softmax(logits)

        # Prosody output
        prosody_raw = np.dot(self.w_prosody, h)
        prosody = np.array([
            80 + 320 * self._sigmoid(prosody_raw[0]),   # f0: 80-400 Hz
            50 + 200 * self._sigmoid(prosody_raw[1]),   # duration: 50-250 ms
            0.3 + 0.7 * self._sigmoid(prosody_raw[2])   # amplitude: 0.3-1.0
        ])

        return phoneme_probs, prosody

    def generate_audio(self, spikes: np.ndarray) -> np.ndarray:
        """Generate audio from spike pattern."""
        phoneme_probs, prosody = self.forward(spikes)

        # Select phoneme (could sample, but using argmax for clarity)
        phoneme_idx = np.argmax(phoneme_probs)
        phoneme = self.phoneme_labels[phoneme_idx]

        # Synthesize
        audio = self.synthesizer.synthesize_phoneme(
            phoneme,
            duration_ms=prosody[1],
            f0=prosody[0],
            amplitude=prosody[2]
        )

        return audio

    def generate_sequence(
        self,
        spike_sequence: np.ndarray,
        temperature: float = 1.0
    ) -> np.ndarray:
        """Generate audio from sequence of spike patterns."""
        audio_segments = []

        for spikes in spike_sequence:
            phoneme_probs, prosody = self.forward(spikes)

            # Apply temperature and sample
            if temperature != 1.0:
                logits = np.log(phoneme_probs + 1e-10) / temperature
                phoneme_probs = self._softmax(logits)

            phoneme_idx = np.argmax(phoneme_probs)
            phoneme = self.phoneme_labels[phoneme_idx]

            segment = self.synthesizer.synthesize_phoneme(
                phoneme,
                duration_ms=prosody[1],
                f0=prosody[0],
                amplitude=prosody[2]
            )
            audio_segments.append(segment)

        return np.concatenate(audio_segments) if audio_segments else np.array([])

    def learn(
        self,
        spikes: np.ndarray,
        target_phoneme: str,
        target_prosody: Optional[np.ndarray] = None,
        lr: float = 0.001
    ):
        """Learn to produce target phoneme from spike pattern."""
        # Forward pass
        h = np.maximum(0, np.dot(self.w1, spikes) + self.b1)
        logits = np.dot(self.w2, h) + self.b2
        phoneme_probs = self._softmax(logits)

        # Target
        target_idx = self.phoneme_labels.index(target_phoneme.upper())
        target = np.zeros(self.output_size, dtype=np.float32)
        target[target_idx] = 1.0

        # Backward pass (simple gradient descent)
        # Output layer
        d_logits = phoneme_probs - target
        d_w2 = np.outer(d_logits, h)
        d_b2 = d_logits

        # Hidden layer
        d_h = np.dot(self.w2.T, d_logits)
        d_h[h <= 0] = 0  # ReLU gradient
        d_w1 = np.outer(d_h, spikes)
        d_b1 = d_h

        # Update weights
        self.w2 -= lr * d_w2
        self.b2 -= lr * d_b2
        self.w1 -= lr * d_w1
        self.b1 -= lr * d_b1

    @staticmethod
    def _softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / (exp_x.sum() + 1e-10)

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class SpeechGenerator:
    """
    High-level speech generation interface.

    Converts text/concepts to speech:
    1. Text -> Phoneme sequence
    2. Phoneme sequence -> Neural activations
    3. Neural activations -> Audio
    """

    def __init__(self, config: Optional[VocoderConfig] = None):
        self.config = config or VocoderConfig()
        self.vocoder = NeuralVocoder(config)

        # Simple grapheme-to-phoneme rules (very simplified)
        self.g2p_rules = {
            'a': 'AE', 'e': 'EH', 'i': 'IH', 'o': 'AO', 'u': 'UH',
            'b': 'B', 'c': 'K', 'd': 'D', 'f': 'F', 'g': 'G',
            'h': 'HH', 'j': 'JH', 'k': 'K', 'l': 'L', 'm': 'M',
            'n': 'N', 'p': 'P', 'q': 'K', 'r': 'R', 's': 'S',
            't': 'T', 'v': 'V', 'w': 'W', 'x': 'K', 'y': 'IY', 'z': 'Z',
            ' ': 'SIL', '.': 'SIL', ',': 'SIL', '!': 'SIL', '?': 'SIL'
        }

    def text_to_phonemes(self, text: str) -> List[str]:
        """Convert text to phoneme sequence (simplified G2P)."""
        phonemes = []
        text = text.lower()

        i = 0
        while i < len(text):
            # Check for digraphs
            if i < len(text) - 1:
                digraph = text[i:i+2]
                if digraph == 'th':
                    phonemes.append('TH')
                    i += 2
                    continue
                elif digraph == 'sh':
                    phonemes.append('SH')
                    i += 2
                    continue
                elif digraph == 'ch':
                    phonemes.append('CH')
                    i += 2
                    continue
                elif digraph == 'ee':
                    phonemes.append('IY')
                    i += 2
                    continue
                elif digraph == 'oo':
                    phonemes.append('UW')
                    i += 2
                    continue

            # Single character
            char = text[i]
            phoneme = self.g2p_rules.get(char, 'SIL')
            if phoneme != 'SIL' or (phonemes and phonemes[-1] != 'SIL'):
                phonemes.append(phoneme)
            i += 1

        return phonemes

    def speak(self, text: str, speed: float = 1.0) -> np.ndarray:
        """
        Convert text to speech audio.

        Args:
            text: Text to speak
            speed: Speech rate multiplier

        Returns:
            Audio waveform
        """
        phonemes = self.text_to_phonemes(text)

        if not phonemes:
            return np.zeros(1600, dtype=np.float32)

        # Generate durations (simple)
        base_duration = 100 / speed
        durations = [base_duration] * len(phonemes)

        # Generate f0 contour (simple declination)
        f0_start = 150
        f0_end = 100
        f0_contour = np.linspace(f0_start, f0_end, len(phonemes))

        # Synthesize
        audio = self.vocoder.synthesizer.synthesize_sequence(
            phonemes, durations, f0_contour.tolist()
        )

        return audio

    def speak_from_spikes(
        self,
        spike_sequence: np.ndarray
    ) -> np.ndarray:
        """Generate speech from neural spike patterns."""
        return self.vocoder.generate_sequence(spike_sequence)


def save_audio(audio: np.ndarray, path: str, sample_rate: int = 16000):
    """Save audio to WAV file."""
    try:
        from scipy.io import wavfile
        # Normalize and convert to int16
        audio_normalized = audio / (np.max(np.abs(audio)) + 1e-8)
        audio_int16 = (audio_normalized * 32767).astype(np.int16)
        wavfile.write(path, sample_rate, audio_int16)
    except ImportError:
        print("scipy required to save audio files")


if __name__ == "__main__":
    print("Testing Speech Generator")
    print("=" * 50)

    # Create generator
    generator = SpeechGenerator()

    # Test text to phonemes
    test_text = "hello world"
    phonemes = generator.text_to_phonemes(test_text)
    print(f"Text: '{test_text}'")
    print(f"Phonemes: {' '.join(phonemes)}")

    # Generate speech
    print("\nGenerating speech...")
    audio = generator.speak(test_text)
    print(f"Audio length: {len(audio)} samples ({len(audio)/16000:.2f}s)")

    # Test formant synthesizer directly
    print("\nTesting formant synthesizer...")
    synth = FormantSynthesizer()

    # Synthesize a vowel
    vowel = synth.synthesize_phoneme('IY', duration_ms=200, f0=150)
    print(f"Vowel 'IY': {len(vowel)} samples")

    # Save test audio
    try:
        save_audio(audio, 'test_speech.wav')
        print("\nSaved to test_speech.wav")
    except Exception as e:
        print(f"\nCould not save audio: {e}")

    # Test neural vocoder
    print("\nTesting neural vocoder...")
    vocoder = NeuralVocoder()

    # Random spike pattern
    spikes = np.random.rand(1000).astype(np.float32)
    spikes[spikes < 0.8] = 0  # Sparse

    probs, prosody = vocoder.forward(spikes)
    predicted_phoneme = vocoder.phoneme_labels[np.argmax(probs)]
    print(f"Predicted phoneme: {predicted_phoneme}")
    print(f"Prosody - F0: {prosody[0]:.1f}Hz, Duration: {prosody[1]:.1f}ms, Amp: {prosody[2]:.2f}")

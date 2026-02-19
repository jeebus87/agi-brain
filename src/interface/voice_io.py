"""
Voice I/O - Microphone input and speaker output

Provides real-time audio interface for:
- Speech input (microphone capture)
- Speech output (audio playback)
- Voice activity detection
"""

import numpy as np
from typing import Optional, Callable, Generator
from dataclasses import dataclass
import threading
import queue
import time

# Optional audio libraries
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False


@dataclass
class AudioConfig:
    """Audio configuration."""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024      # Samples per chunk
    dtype: str = 'float32'
    vad_threshold: float = 0.02  # Voice activity threshold


class VoiceActivityDetector:
    """Simple voice activity detection based on energy."""

    def __init__(self, threshold: float = 0.02, history_size: int = 10):
        self.threshold = threshold
        self.history_size = history_size
        self.energy_history = []

    def is_speech(self, audio: np.ndarray) -> bool:
        """Detect if audio contains speech."""
        energy = np.sqrt(np.mean(audio ** 2))
        self.energy_history.append(energy)

        if len(self.energy_history) > self.history_size:
            self.energy_history.pop(0)

        # Adaptive threshold based on recent history
        if len(self.energy_history) >= 3:
            baseline = np.percentile(self.energy_history, 20)
            adaptive_threshold = max(self.threshold, baseline * 2)
        else:
            adaptive_threshold = self.threshold

        return energy > adaptive_threshold

    def reset(self):
        """Reset energy history."""
        self.energy_history = []


class Microphone:
    """
    Microphone input handler.

    Captures audio from microphone and provides:
    - Continuous streaming
    - Voice activity detection
    - Automatic gain control
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.vad = VoiceActivityDetector(self.config.vad_threshold)

        self._audio_queue = queue.Queue()
        self._is_recording = False
        self._stream = None

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream."""
        if status:
            print(f"Audio status: {status}")
        self._audio_queue.put(indata.copy())

    def start_recording(self):
        """Start capturing audio from microphone."""
        if not SOUNDDEVICE_AVAILABLE and not PYAUDIO_AVAILABLE:
            raise RuntimeError("No audio library available. Install sounddevice or pyaudio.")

        self._is_recording = True

        if SOUNDDEVICE_AVAILABLE:
            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype,
                blocksize=self.config.chunk_size,
                callback=self._audio_callback
            )
            self._stream.start()
        else:
            # PyAudio fallback
            self._start_pyaudio()

    def _start_pyaudio(self):
        """Start recording with PyAudio."""
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=self.config.chunk_size,
            stream_callback=self._pyaudio_callback
        )
        self._stream.start_stream()

    def _pyaudio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback."""
        audio = np.frombuffer(in_data, dtype=np.float32)
        self._audio_queue.put(audio.reshape(-1, 1))
        return (None, pyaudio.paContinue)

    def stop_recording(self):
        """Stop capturing audio."""
        self._is_recording = False
        if self._stream:
            if SOUNDDEVICE_AVAILABLE:
                self._stream.stop()
                self._stream.close()
            else:
                self._stream.stop_stream()
                self._stream.close()
                self._pa.terminate()
            self._stream = None

    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get next audio chunk from queue."""
        try:
            chunk = self._audio_queue.get(timeout=timeout)
            return chunk.flatten()
        except queue.Empty:
            return None

    def record_utterance(
        self,
        max_duration: float = 10.0,
        silence_timeout: float = 1.0
    ) -> np.ndarray:
        """
        Record a complete utterance.

        Automatically detects start and end of speech.
        """
        self.start_recording()
        self.vad.reset()

        audio_chunks = []
        speech_started = False
        silence_duration = 0.0
        total_duration = 0.0

        chunk_duration = self.config.chunk_size / self.config.sample_rate

        try:
            while total_duration < max_duration:
                chunk = self.get_audio_chunk(timeout=0.5)
                if chunk is None:
                    continue

                total_duration += chunk_duration
                is_speech = self.vad.is_speech(chunk)

                if is_speech:
                    speech_started = True
                    silence_duration = 0.0
                    audio_chunks.append(chunk)
                elif speech_started:
                    silence_duration += chunk_duration
                    audio_chunks.append(chunk)

                    if silence_duration >= silence_timeout:
                        # End of utterance
                        break
        finally:
            self.stop_recording()

        if audio_chunks:
            return np.concatenate(audio_chunks)
        return np.array([], dtype=np.float32)

    def stream_audio(self) -> Generator[np.ndarray, None, None]:
        """Generator that yields audio chunks."""
        self.start_recording()
        try:
            while self._is_recording:
                chunk = self.get_audio_chunk(timeout=0.5)
                if chunk is not None:
                    yield chunk
        finally:
            self.stop_recording()


class Speaker:
    """
    Speaker output handler.

    Plays audio through speakers with:
    - Non-blocking playback
    - Volume control
    - Playback queue
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self._playback_queue = queue.Queue()
        self._is_playing = False
        self._volume = 1.0

    def play(self, audio: np.ndarray, blocking: bool = True):
        """
        Play audio through speakers.

        Args:
            audio: Audio waveform
            blocking: Wait for playback to complete
        """
        if not SOUNDDEVICE_AVAILABLE:
            print("sounddevice not available, cannot play audio")
            return

        # Apply volume
        audio = audio * self._volume

        # Ensure correct dtype
        audio = audio.astype(np.float32)

        if blocking:
            sd.play(audio, self.config.sample_rate)
            sd.wait()
        else:
            self._playback_queue.put(audio)
            if not self._is_playing:
                self._start_playback_thread()

    def _start_playback_thread(self):
        """Start background playback thread."""
        self._is_playing = True
        thread = threading.Thread(target=self._playback_loop, daemon=True)
        thread.start()

    def _playback_loop(self):
        """Background playback loop."""
        while True:
            try:
                audio = self._playback_queue.get(timeout=1.0)
                sd.play(audio, self.config.sample_rate)
                sd.wait()
            except queue.Empty:
                self._is_playing = False
                break

    def set_volume(self, volume: float):
        """Set playback volume (0.0 - 1.0)."""
        self._volume = max(0.0, min(1.0, volume))

    def stop(self):
        """Stop current playback."""
        if SOUNDDEVICE_AVAILABLE:
            sd.stop()


class VoiceInterface:
    """
    Complete voice interface for the AGI brain.

    Handles:
    - Speech input (listen)
    - Speech output (speak)
    - Continuous conversation
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.microphone = Microphone(config)
        self.speaker = Speaker(config)

        # Callbacks
        self.on_speech_start: Optional[Callable] = None
        self.on_speech_end: Optional[Callable[[np.ndarray], None]] = None
        self.on_response: Optional[Callable[[str], None]] = None

        # State
        self._is_listening = False
        self._conversation_active = False

    def listen(self, max_duration: float = 10.0) -> np.ndarray:
        """Listen for speech input."""
        if self.on_speech_start:
            self.on_speech_start()

        audio = self.microphone.record_utterance(max_duration=max_duration)

        if self.on_speech_end:
            self.on_speech_end(audio)

        return audio

    def speak(self, audio: np.ndarray, blocking: bool = True):
        """Output speech audio."""
        self.speaker.play(audio, blocking=blocking)

    def start_conversation(
        self,
        process_callback: Callable[[np.ndarray], np.ndarray]
    ):
        """
        Start continuous conversation loop.

        Args:
            process_callback: Function that takes input audio and returns response audio
        """
        self._conversation_active = True

        while self._conversation_active:
            # Listen
            input_audio = self.listen()

            if len(input_audio) < self.config.sample_rate * 0.5:
                # Too short, skip
                continue

            # Process
            response_audio = process_callback(input_audio)

            # Speak
            if len(response_audio) > 0:
                self.speak(response_audio)

    def stop_conversation(self):
        """Stop conversation loop."""
        self._conversation_active = False


def test_audio_available() -> dict:
    """Test which audio libraries are available."""
    return {
        'sounddevice': SOUNDDEVICE_AVAILABLE,
        'pyaudio': PYAUDIO_AVAILABLE,
        'can_record': SOUNDDEVICE_AVAILABLE or PYAUDIO_AVAILABLE,
        'can_play': SOUNDDEVICE_AVAILABLE
    }


if __name__ == "__main__":
    print("Testing Voice I/O")
    print("=" * 50)

    status = test_audio_available()
    print("Audio library status:")
    for lib, available in status.items():
        print(f"  {lib}: {available}")

    if not status['can_record']:
        print("\nNo audio library available. Install with:")
        print("  pip install sounddevice")
        print("  or")
        print("  pip install pyaudio")
    else:
        print("\nAudio libraries available!")

        # Test tone generation
        print("\nGenerating test tone...")
        sample_rate = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration))
        tone = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        if status['can_play']:
            print("Playing test tone...")
            speaker = Speaker()
            speaker.play(tone)
            print("Done!")

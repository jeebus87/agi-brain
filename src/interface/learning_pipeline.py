"""
Learning Pipeline - YouTube and Web Learning

Enables the AGI brain to learn from:
- YouTube videos (audio transcription -> language learning)
- Web pages (text extraction -> semantic memory)
- Conversations (dialogue -> associations)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass
import subprocess
import tempfile
import os
import re
import json
from pathlib import Path


@dataclass
class LearningConfig:
    """Configuration for learning pipeline."""
    audio_sample_rate: int = 16000
    chunk_duration: float = 30.0  # Process audio in 30s chunks
    min_word_frequency: int = 2   # Minimum occurrences to learn word
    learning_rate: float = 0.001
    batch_size: int = 32


class YouTubeLearner:
    """
    Learn from YouTube videos.

    Pipeline:
    1. Download audio from video (yt-dlp)
    2. Transcribe audio (local Whisper or simple VAD + phoneme detection)
    3. Extract words and sentences
    4. Update semantic memory with co-occurrences
    5. Train phoneme recognition on audio
    """

    def __init__(self, config: Optional[LearningConfig] = None):
        self.config = config or LearningConfig()
        self.downloaded_videos: List[str] = []

        # Check for yt-dlp
        self._check_ytdlp()

    def _check_ytdlp(self):
        """Check if yt-dlp is installed."""
        try:
            result = subprocess.run(
                ['yt-dlp', '--version'],
                capture_output=True,
                text=True
            )
            self.ytdlp_available = result.returncode == 0
        except FileNotFoundError:
            self.ytdlp_available = False
            print("yt-dlp not found. Install with: pip install yt-dlp")

    def download_audio(
        self,
        url: str,
        output_dir: Optional[str] = None
    ) -> Optional[str]:
        """
        Download audio from YouTube video.

        Returns path to downloaded audio file.
        """
        if not self.ytdlp_available:
            raise RuntimeError("yt-dlp not installed")

        if output_dir is None:
            output_dir = tempfile.mkdtemp()

        output_template = os.path.join(output_dir, '%(title)s.%(ext)s')

        cmd = [
            'yt-dlp',
            '-x',  # Extract audio
            '--audio-format', 'wav',
            '--audio-quality', '0',
            '-o', output_template,
            '--postprocessor-args', f'-ar {self.config.audio_sample_rate} -ac 1',
            '--no-playlist',
            url
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                print(f"yt-dlp error: {result.stderr}")
                return None

            # Find the output file
            for f in os.listdir(output_dir):
                if f.endswith('.wav'):
                    path = os.path.join(output_dir, f)
                    self.downloaded_videos.append(path)
                    return path

            return None

        except subprocess.TimeoutExpired:
            print("Download timed out")
            return None
        except Exception as e:
            print(f"Download error: {e}")
            return None

    def get_video_info(self, url: str) -> Optional[Dict]:
        """Get video metadata without downloading."""
        if not self.ytdlp_available:
            return None

        cmd = ['yt-dlp', '-j', '--no-playlist', url]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return json.loads(result.stdout)
        except:
            pass

        return None

    def transcribe_audio(
        self,
        audio_path: str,
        use_whisper: bool = True
    ) -> List[Dict]:
        """
        Transcribe audio to text with timestamps.

        Returns list of segments: [{'start': float, 'end': float, 'text': str}]
        """
        # Try to use Whisper if available
        if use_whisper:
            try:
                return self._transcribe_with_whisper(audio_path)
            except ImportError:
                print("Whisper not available, using simple transcription")

        # Fallback: simple VAD-based segmentation (no actual transcription)
        return self._segment_audio(audio_path)

    def _transcribe_with_whisper(self, audio_path: str) -> List[Dict]:
        """Transcribe using OpenAI Whisper (runs locally)."""
        import whisper

        model = whisper.load_model("base")  # or "tiny" for faster
        result = model.transcribe(audio_path)

        segments = []
        for seg in result['segments']:
            segments.append({
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text'].strip()
            })

        return segments

    def _segment_audio(self, audio_path: str) -> List[Dict]:
        """
        Simple audio segmentation without transcription.

        Returns segments based on silence detection.
        """
        try:
            from scipy.io import wavfile
            sr, audio = wavfile.read(audio_path)

            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0

        except ImportError:
            # Can't read audio without scipy
            return [{'start': 0, 'end': 30, 'text': '[audio segment]'}]

        # Simple energy-based segmentation
        frame_size = int(0.025 * sr)  # 25ms frames
        hop_size = int(0.010 * sr)    # 10ms hop

        segments = []
        in_speech = False
        speech_start = 0

        threshold = 0.02

        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            energy = np.sqrt(np.mean(frame ** 2))

            current_time = i / sr

            if energy > threshold and not in_speech:
                in_speech = True
                speech_start = current_time
            elif energy < threshold * 0.5 and in_speech:
                in_speech = False
                if current_time - speech_start > 0.3:  # Min 300ms
                    segments.append({
                        'start': speech_start,
                        'end': current_time,
                        'text': f'[segment {len(segments)}]'
                    })

        return segments

    def learn_from_video(
        self,
        url: str,
        brain,  # AGI brain instance
        progress_callback: Optional[callable] = None
    ) -> Dict:
        """
        Complete pipeline to learn from a YouTube video.

        Args:
            url: YouTube video URL
            brain: AGI brain instance with semantic_memory and phoneme_learner
            progress_callback: Optional callback(progress, message)

        Returns:
            Statistics about what was learned
        """
        stats = {
            'url': url,
            'duration': 0,
            'segments': 0,
            'words_learned': 0,
            'associations_learned': 0
        }

        # Download
        if progress_callback:
            progress_callback(0.1, "Downloading audio...")

        audio_path = self.download_audio(url)
        if not audio_path:
            stats['error'] = 'Download failed'
            return stats

        # Get info
        info = self.get_video_info(url)
        if info:
            stats['title'] = info.get('title', 'Unknown')
            stats['duration'] = info.get('duration', 0)

        # Transcribe
        if progress_callback:
            progress_callback(0.3, "Transcribing audio...")

        segments = self.transcribe_audio(audio_path)
        stats['segments'] = len(segments)

        # Learn from transcription
        if progress_callback:
            progress_callback(0.5, "Learning from content...")

        all_words = []
        for i, segment in enumerate(segments):
            text = segment.get('text', '')
            words = self._extract_words(text)
            all_words.extend(words)

            # Learn word associations from sentence
            if hasattr(brain, 'semantic_memory') and words:
                brain.semantic_memory.learn_from_sentence(words)
                stats['associations_learned'] += len(words) - 1

            if progress_callback:
                progress = 0.5 + 0.4 * (i / len(segments))
                progress_callback(progress, f"Processing segment {i+1}/{len(segments)}")

        # Count unique words learned
        unique_words = set(all_words)
        stats['words_learned'] = len(unique_words)

        if progress_callback:
            progress_callback(1.0, "Complete!")

        return stats

    def _extract_words(self, text: str) -> List[str]:
        """Extract words from text."""
        # Simple tokenization
        text = text.lower()
        words = re.findall(r'\b[a-z]+\b', text)
        return words


class WebLearner:
    """
    Learn from web pages.

    Pipeline:
    1. Fetch web page
    2. Extract text content
    3. Process sentences
    4. Update semantic memory
    """

    def __init__(self, config: Optional[LearningConfig] = None):
        self.config = config or LearningConfig()

    def fetch_page(self, url: str) -> Optional[str]:
        """Fetch web page content."""
        try:
            import urllib.request
            from html.parser import HTMLParser

            # Simple HTML parser to extract text
            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text = []
                    self.skip_tags = {'script', 'style', 'nav', 'header', 'footer'}
                    self.current_tag = None

                def handle_starttag(self, tag, attrs):
                    self.current_tag = tag

                def handle_data(self, data):
                    if self.current_tag not in self.skip_tags:
                        text = data.strip()
                        if text:
                            self.text.append(text)

            # Fetch page
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'AGI-Brain-Learner/1.0'}
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                html = response.read().decode('utf-8', errors='ignore')

            # Extract text
            parser = TextExtractor()
            parser.feed(html)

            return ' '.join(parser.text)

        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            return None

    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        # Simple sentence splitting
        text = re.sub(r'\s+', ' ', text)
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def learn_from_url(
        self,
        url: str,
        brain,  # AGI brain instance
        max_sentences: int = 100
    ) -> Dict:
        """Learn from a web page."""
        stats = {
            'url': url,
            'sentences': 0,
            'words_learned': 0,
            'associations_learned': 0
        }

        # Fetch
        content = self.fetch_page(url)
        if not content:
            stats['error'] = 'Failed to fetch page'
            return stats

        # Extract sentences
        sentences = self.extract_sentences(content)[:max_sentences]
        stats['sentences'] = len(sentences)

        # Learn
        all_words = set()
        for sentence in sentences:
            words = re.findall(r'\b[a-z]+\b', sentence.lower())
            all_words.update(words)

            if hasattr(brain, 'semantic_memory') and words:
                brain.semantic_memory.learn_from_sentence(words)
                stats['associations_learned'] += len(words) - 1

        stats['words_learned'] = len(all_words)

        return stats

    def search_and_learn(
        self,
        query: str,
        brain,
        max_results: int = 5
    ) -> List[Dict]:
        """Search web and learn from results."""
        # Use DuckDuckGo (no API key needed)
        search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"

        content = self.fetch_page(search_url)
        if not content:
            return []

        # Extract URLs from search results (simplified)
        urls = re.findall(r'href="(https?://[^"]+)"', content)
        urls = [u for u in urls if 'duckduckgo' not in u][:max_results]

        results = []
        for url in urls:
            try:
                stats = self.learn_from_url(url, brain)
                results.append(stats)
            except Exception as e:
                results.append({'url': url, 'error': str(e)})

        return results


class LearningPipeline:
    """
    Unified learning pipeline for the AGI brain.

    Coordinates learning from multiple sources:
    - YouTube videos
    - Web pages
    - Live conversations
    - Local files
    """

    def __init__(
        self,
        brain,  # AGI brain instance
        config: Optional[LearningConfig] = None
    ):
        self.brain = brain
        self.config = config or LearningConfig()

        self.youtube = YouTubeLearner(config)
        self.web = WebLearner(config)

        # Learning history
        self.history: List[Dict] = []

    def learn_youtube(
        self,
        url: str,
        progress_callback: Optional[callable] = None
    ) -> Dict:
        """Learn from YouTube video."""
        stats = self.youtube.learn_from_video(
            url, self.brain, progress_callback
        )
        self.history.append({'type': 'youtube', **stats})
        return stats

    def learn_web(self, url: str) -> Dict:
        """Learn from web page."""
        stats = self.web.learn_from_url(url, self.brain)
        self.history.append({'type': 'web', **stats})
        return stats

    def learn_text(self, text: str) -> Dict:
        """Learn from raw text."""
        sentences = self.web.extract_sentences(text)
        stats = {
            'type': 'text',
            'sentences': len(sentences),
            'words_learned': 0,
            'associations_learned': 0
        }

        all_words = set()
        for sentence in sentences:
            words = re.findall(r'\b[a-z]+\b', sentence.lower())
            all_words.update(words)

            if hasattr(self.brain, 'semantic_memory'):
                self.brain.semantic_memory.learn_from_sentence(words)
                stats['associations_learned'] += len(words) - 1

        stats['words_learned'] = len(all_words)
        self.history.append(stats)

        return stats

    def learn_conversation(
        self,
        user_text: str,
        response_text: str
    ) -> Dict:
        """Learn from a conversation exchange."""
        # Combine as context
        combined = f"{user_text} {response_text}"

        stats = self.learn_text(combined)
        stats['type'] = 'conversation'

        return stats

    def get_statistics(self) -> Dict:
        """Get overall learning statistics."""
        total_words = 0
        total_associations = 0
        sources = {'youtube': 0, 'web': 0, 'text': 0, 'conversation': 0}

        for entry in self.history:
            total_words += entry.get('words_learned', 0)
            total_associations += entry.get('associations_learned', 0)
            source_type = entry.get('type', 'unknown')
            sources[source_type] = sources.get(source_type, 0) + 1

        return {
            'total_sessions': len(self.history),
            'total_words_learned': total_words,
            'total_associations': total_associations,
            'sources': sources
        }


if __name__ == "__main__":
    print("Testing Learning Pipeline")
    print("=" * 50)

    # Test YouTube learner
    yt = YouTubeLearner()
    print(f"yt-dlp available: {yt.ytdlp_available}")

    # Test web learner
    web = WebLearner()
    print("\nTesting web extraction...")

    test_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    content = web.fetch_page(test_url)

    if content:
        sentences = web.extract_sentences(content)
        print(f"Extracted {len(sentences)} sentences from Wikipedia")
        if sentences:
            print(f"First sentence: {sentences[0][:100]}...")
    else:
        print("Could not fetch page (may need internet connection)")

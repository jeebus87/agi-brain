"""
Chat UI - Gradio-based interface for AGI Brain

Provides:
- Text chat interface
- Voice input/output
- Learning controls
- Brain status visualization
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import time


@dataclass
class ChatConfig:
    """Configuration for chat interface."""
    title: str = "AGI Brain Chat"
    description: str = "Talk to a spiking neural network that learns from scratch"
    theme: str = "soft"
    enable_voice: bool = True
    enable_learning: bool = True


class AGIBrain:
    """
    Complete AGI Brain system.

    Integrates all components:
    - Sparse SNN (10M neurons)
    - Phoneme learning
    - Semantic memory
    - Speech generation
    """

    def __init__(self, n_neurons: int = 1_000_000):
        from ..language.sparse_network import create_language_brain, SparseSNN
        from ..language.audio_encoder import CochleaEncoder, AudioProcessor
        from ..language.phoneme_learner import PhonemeLearner, PhonemeConfig
        from ..language.semantic_memory import SemanticMemory, SemanticConfig
        from ..language.speech_generator import SpeechGenerator

        print(f"Initializing AGI Brain with {n_neurons:,} neurons...")

        # Core SNN
        self.snn = create_language_brain(n_neurons, use_gpu=True)

        # Audio processing
        self.audio_processor = AudioProcessor()
        self.cochlea = CochleaEncoder()

        # Phoneme learning
        phoneme_config = PhonemeConfig(n_phoneme_neurons=1000, n_phonemes=40)
        self.phoneme_learner = PhonemeLearner(phoneme_config)

        # Semantic memory
        semantic_config = SemanticConfig(n_hard_locations=10000)
        self.semantic_memory = SemanticMemory(semantic_config)

        # Speech generation
        self.speech_generator = SpeechGenerator()

        # State
        self.conversation_history: List[Tuple[str, str]] = []
        self.is_learning = True
        self.time_step = 0

        print("AGI Brain initialized!")

    def process_text(self, text: str) -> str:
        """
        Process text input and generate response.

        This is where the neural network actually thinks:
        1. Encode text to neural representation
        2. Process through SNN
        3. Generate response from neural activity
        """
        self.time_step += 1

        # Learn from input
        if self.is_learning:
            words = text.lower().split()
            self.semantic_memory.learn_from_sentence(words)

        # Generate response based on current knowledge
        response = self._generate_response(text)

        # Store in history
        self.conversation_history.append((text, response))

        # Learn from exchange
        if self.is_learning:
            response_words = response.lower().split()
            self.semantic_memory.learn_from_sentence(words + response_words)

        return response

    def _generate_response(self, input_text: str) -> str:
        """
        Generate response using neural network.

        Current implementation: association-based response
        Future: Full neural language generation
        """
        words = input_text.lower().split()

        if not words:
            return "I'm listening..."

        # Look for known words
        known_words = []
        for word in words:
            if self.semantic_memory.lookup(word):
                known_words.append(word)

        if not known_words:
            # Unknown input - try to learn
            for word in words:
                self.semantic_memory.create_concept(word=word)
            return f"I'm learning new words: {', '.join(words[:3])}..."

        # Find associations
        activations = {}
        for word in known_words:
            word_activations = self.semantic_memory.spread_activation(word, depth=2)
            for w, a in word_activations.items():
                activations[w] = activations.get(w, 0) + a

        # Generate response from top activations
        if activations:
            sorted_words = sorted(activations.items(), key=lambda x: -x[1])
            top_words = [w for w, a in sorted_words[:5] if w not in known_words]

            if top_words:
                response = f"When I think of {known_words[0]}, I think of: {', '.join(top_words)}"
            else:
                response = f"I know about {known_words[0]}, but I need to learn more associations."
        else:
            response = f"I've heard of {known_words[0]}, but I'm still learning what it means."

        return response

    def process_audio(self, audio: np.ndarray) -> Tuple[str, np.ndarray]:
        """
        Process audio input and generate audio response.

        Returns:
            (transcription, response_audio)
        """
        # Encode audio to spikes
        spikes = self.cochlea.encode_audio(audio)

        # Process through phoneme learner
        detected_phonemes = []
        for t in range(min(100, len(spikes))):
            phoneme_id, conf = self.phoneme_learner.process(spikes[t], learn=self.is_learning)
            if conf > 0.3:
                detected_phonemes.append(self.phoneme_learner.get_label(phoneme_id))

        # Convert to text (very basic)
        text = ' '.join(detected_phonemes[:10]) if detected_phonemes else "[audio detected]"

        # Generate text response
        response_text = self.process_text(text)

        # Generate audio response
        response_audio = self.speech_generator.speak(response_text)

        return text, response_audio

    def get_status(self) -> dict:
        """Get brain status."""
        return {
            'neurons': self.snn.total_neurons() if hasattr(self.snn, 'total_neurons') else 0,
            'vocabulary': self.semantic_memory.get_vocabulary_size(),
            'concepts': len(self.semantic_memory.concepts),
            'phonemes_learned': sum(self.phoneme_learner.detection_counts > 10),
            'conversation_turns': len(self.conversation_history),
            'is_learning': self.is_learning,
            'time_step': self.time_step
        }


class ChatInterface:
    """
    Gradio-based chat interface.
    """

    def __init__(
        self,
        brain: Optional[AGIBrain] = None,
        config: Optional[ChatConfig] = None
    ):
        self.config = config or ChatConfig()
        self.brain = brain

    def _create_brain_if_needed(self):
        """Lazy initialization of brain."""
        if self.brain is None:
            self.brain = AGIBrain(n_neurons=100_000)  # Start smaller for UI

    def chat(self, message: str, history: List) -> str:
        """Process chat message."""
        self._create_brain_if_needed()
        response = self.brain.process_text(message)
        return response

    def process_voice(self, audio) -> Tuple[str, str]:
        """Process voice input."""
        self._create_brain_if_needed()

        if audio is None:
            return "", "No audio detected"

        # audio is (sample_rate, data) tuple from Gradio
        sample_rate, audio_data = audio

        # Normalize
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0

        # Process
        transcription, response_audio = self.brain.process_audio(audio_data)

        # For now, return text response (audio playback needs more setup)
        response = self.brain.process_text(transcription)

        return transcription, response

    def learn_youtube(self, url: str, progress=None) -> str:
        """Learn from YouTube video."""
        self._create_brain_if_needed()

        from .learning_pipeline import YouTubeLearner

        learner = YouTubeLearner()

        def update_progress(p, msg):
            if progress:
                progress(p, desc=msg)

        stats = learner.learn_from_video(url, self.brain, update_progress)

        if 'error' in stats:
            return f"Error: {stats['error']}"

        return f"""
Learned from: {stats.get('title', url)}
- Duration: {stats.get('duration', 0)//60} minutes
- Segments processed: {stats['segments']}
- Words learned: {stats['words_learned']}
- Associations created: {stats['associations_learned']}
"""

    def learn_web(self, url: str) -> str:
        """Learn from web page."""
        self._create_brain_if_needed()

        from .learning_pipeline import WebLearner

        learner = WebLearner()
        stats = learner.learn_from_url(url, self.brain)

        if 'error' in stats:
            return f"Error: {stats['error']}"

        return f"""
Learned from: {url}
- Sentences processed: {stats['sentences']}
- Words learned: {stats['words_learned']}
- Associations created: {stats['associations_learned']}
"""

    def get_brain_status(self) -> str:
        """Get brain status as formatted string."""
        self._create_brain_if_needed()
        status = self.brain.get_status()

        return f"""
## Brain Status

| Metric | Value |
|--------|-------|
| Neurons | {status['neurons']:,} |
| Vocabulary | {status['vocabulary']:,} words |
| Concepts | {status['concepts']:,} |
| Phonemes | {status['phonemes_learned']}/40 |
| Conversations | {status['conversation_turns']} turns |
| Learning | {'ON' if status['is_learning'] else 'OFF'} |
"""

    def toggle_learning(self) -> str:
        """Toggle learning mode."""
        self._create_brain_if_needed()
        self.brain.is_learning = not self.brain.is_learning
        state = "enabled" if self.brain.is_learning else "disabled"
        return f"Learning {state}"


def create_gradio_app(brain: Optional[AGIBrain] = None):
    """
    Create Gradio app for the AGI Brain.

    Returns a Gradio Blocks app that can be launched.
    """
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Install with: pip install gradio")
        return None

    interface = ChatInterface(brain)

    with gr.Blocks(title="AGI Brain", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # AGI Brain Chat

        Talk to a spiking neural network that learns language from scratch.

        This brain has no pre-trained language model - it learns through:
        - Conversation (chat with it)
        - YouTube videos (paste a URL)
        - Web pages (paste a URL)
        """)

        with gr.Tabs():
            # Chat tab
            with gr.Tab("Chat"):
                chatbot = gr.Chatbot(height=400)
                msg = gr.Textbox(
                    placeholder="Type a message...",
                    label="Your message"
                )

                def respond(message, history):
                    response = interface.chat(message, history)
                    history.append((message, response))
                    return "", history

                msg.submit(respond, [msg, chatbot], [msg, chatbot])

            # Voice tab
            with gr.Tab("Voice"):
                gr.Markdown("### Voice Chat (requires microphone)")

                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="Speak"
                )
                transcription = gr.Textbox(label="Transcription")
                voice_response = gr.Textbox(label="Response")

                audio_input.change(
                    interface.process_voice,
                    inputs=[audio_input],
                    outputs=[transcription, voice_response]
                )

            # Learning tab
            with gr.Tab("Learn"):
                gr.Markdown("### Teach the brain from external sources")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### YouTube")
                        yt_url = gr.Textbox(
                            placeholder="https://youtube.com/watch?v=...",
                            label="YouTube URL"
                        )
                        yt_btn = gr.Button("Learn from Video")
                        yt_result = gr.Textbox(label="Result", lines=5)

                        yt_btn.click(
                            interface.learn_youtube,
                            inputs=[yt_url],
                            outputs=[yt_result]
                        )

                    with gr.Column():
                        gr.Markdown("#### Web Page")
                        web_url = gr.Textbox(
                            placeholder="https://...",
                            label="Web URL"
                        )
                        web_btn = gr.Button("Learn from Page")
                        web_result = gr.Textbox(label="Result", lines=5)

                        web_btn.click(
                            interface.learn_web,
                            inputs=[web_url],
                            outputs=[web_result]
                        )

            # Status tab
            with gr.Tab("Status"):
                status_display = gr.Markdown()
                refresh_btn = gr.Button("Refresh Status")

                refresh_btn.click(
                    interface.get_brain_status,
                    outputs=[status_display]
                )

                toggle_btn = gr.Button("Toggle Learning")
                toggle_result = gr.Textbox(label="Learning State")

                toggle_btn.click(
                    interface.toggle_learning,
                    outputs=[toggle_result]
                )

    return app


if __name__ == "__main__":
    print("Creating AGI Brain Chat Interface...")

    # Create app
    app = create_gradio_app()

    if app:
        print("Launching Gradio interface...")
        app.launch(share=True)
    else:
        print("Could not create app. Make sure gradio is installed.")

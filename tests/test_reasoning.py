"""Tests for reasoning POC."""

import pytest

# Skip tests if dependencies not installed
pytest.importorskip("nengo")
pytest.importorskip("nengo_spa")

import nengo
import nengo_spa as spa

from src.reasoning.reasoning_poc import (
    ReasoningPOC,
    ProblemEncoder,
    RuleEngine,
    AnalogyEngine,
    ResponseGenerator,
    create_reasoning_poc,
)


class TestReasoningPOC:
    """Tests for the main reasoning POC."""

    @pytest.fixture
    def vocab(self):
        """Create a test vocabulary."""
        v = spa.Vocabulary(64)  # Small for testing
        v.populate("A; B; C; IMPLIES; PREMISE; CONCLUSION")
        return v

    def test_poc_creation(self, vocab):
        """Test basic POC creation."""
        with spa.Network() as model:
            poc = ReasoningPOC(vocab=vocab, scale=0.01)
        assert poc is not None

    def test_poc_has_all_components(self, vocab):
        """Test that POC has all required components."""
        with spa.Network() as model:
            poc = ReasoningPOC(vocab=vocab, scale=0.01)

            assert poc.encoder is not None
            assert poc.working_memory is not None
            assert poc.rule_engine is not None
            assert poc.analogy_engine is not None
            assert poc.executive is not None
            assert poc.response is not None

    def test_create_reasoning_poc_helper(self):
        """Test the helper function for creating POC."""
        model, vocab = create_reasoning_poc(scale=0.01)
        assert model is not None
        assert vocab is not None
        assert "A" in vocab


class TestProblemEncoder:
    """Tests for the problem encoder."""

    @pytest.fixture
    def vocab(self):
        v = spa.Vocabulary(64)
        v.populate("A; B; C")
        return v

    def test_encoder_creation(self, vocab):
        with spa.Network() as model:
            encoder = ProblemEncoder(vocab=vocab, n_neurons=100)
        assert encoder is not None


class TestRuleEngine:
    """Tests for the rule engine."""

    @pytest.fixture
    def vocab(self):
        v = spa.Vocabulary(64)
        v.populate("A; B; C; IMPLIES")
        return v

    def test_rule_engine_creation(self, vocab):
        with spa.Network() as model:
            engine = RuleEngine(vocab=vocab, n_neurons=100)
        assert engine is not None


class TestAnalogyEngine:
    """Tests for the analogy engine."""

    @pytest.fixture
    def vocab(self):
        v = spa.Vocabulary(64)
        v.populate("A; B; C; D")
        return v

    def test_analogy_engine_creation(self, vocab):
        with spa.Network() as model:
            engine = AnalogyEngine(vocab=vocab, n_neurons=100)
        assert engine is not None


class TestResponseGenerator:
    """Tests for the response generator."""

    @pytest.fixture
    def vocab(self):
        v = spa.Vocabulary(64)
        v.populate("A; B; ANSWER")
        return v

    def test_response_generator_creation(self, vocab):
        with spa.Network() as model:
            generator = ResponseGenerator(vocab=vocab, n_neurons=100)
        assert generator is not None


class TestIntegration:
    """Integration tests for the reasoning system."""

    def test_full_simulation(self):
        """Test that the full system can be simulated."""
        model, vocab = create_reasoning_poc(scale=0.01)

        # Run a short simulation
        with nengo.Simulator(model, progress_bar=False) as sim:
            sim.run(0.1)

        # Check that we got output
        assert sim.data[model.output_probe].shape[0] > 0

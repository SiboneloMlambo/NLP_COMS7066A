import numpy as np
import torch

from src.data import Vocabulary, context_mapping, generate_real_examples, make_autoregressive_inputs
from src.model import TinyTransformer
from src.analysis import circuit_scores, generation_metrics, head_ablation, heuristic_baselines


def test_data_shapes_and_targets():
    vocab = Vocabulary(8)
    xb, yb = generate_real_examples(10, vocab, 6, "base", 1)
    xe, ye = generate_real_examples(10, vocab, 6, "extended", 1)
    assert xb.shape == xe.shape == (10, 14)
    assert yb.shape == (10, 1)
    assert ye.shape == (10, 3)
    assert np.all((yb >= vocab.value_offset) & (yb < 2 * vocab.num_symbols))
    assert np.all((ye[:, 1] >= 0) & (ye[:, 1] < vocab.num_symbols))


def test_targets_are_consistent_with_context():
    vocab = Vocabulary(8)
    x, y = generate_real_examples(200, vocab, 6, "extended", 3)
    for ctx, tgt in zip(x, y):
        mapping = context_mapping(ctx, vocab)
        assert mapping[int(ctx[-1])] == tgt[0]          # query value inferable
        assert int(tgt[1]) in mapping                   # next symbol shown in context
        assert mapping[int(tgt[1])] == tgt[2]           # its value inferable
    # Real targets must score perfectly on the consistency metrics.
    m = generation_metrics(y, x, y, vocab.num_symbols, "extended", "real")
    assert m["real_first_value_accuracy"] == 1.0
    assert m["real_pair_consistency"] == 1.0
    assert m["real_symbol_in_context"] == 1.0


def test_baselines_are_below_perfect():
    vocab = Vocabulary(8)
    x, y = generate_real_examples(2000, vocab, 6, "base", 4)
    b = heuristic_baselines(x, y)
    assert 0.15 < b["baseline_random_context_value"] < 0.5
    assert 0 < b["baseline_uniform_value_attention_induction_score"] < 1


def test_model_alignment_scores_and_ablation():
    vocab = Vocabulary(8)
    x, y = generate_real_examples(4, vocab, 6, "extended", 2)
    xt, yt = torch.tensor(x), torch.tensor(y)
    inputs = make_autoregressive_inputs(xt, yt)
    model = TinyTransformer(vocab.vocab_size, 20, 32, 4, 2, 64, 0.0)
    logits = model(inputs)
    assert inputs.shape[1] == x.shape[1] + 2
    assert logits.shape == (4, inputs.shape[1], vocab.vocab_size)
    assert len(model.attention_maps()) == 2
    scores = circuit_scores(model, x, torch.device("cpu"))
    assert 0 <= scores["best_previous_token_score"] <= 1
    assert 0 <= scores["best_induction_score"] <= 1
    ablation = head_ablation(model, x, y, torch.device("cpu"))
    assert "L1H1_ablation_drop" in ablation
    # Masks must be restored after ablation.
    assert all(float(b.attn.head_mask.min()) == 1.0 for b in model.blocks)

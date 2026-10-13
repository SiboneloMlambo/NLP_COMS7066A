from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Vocabulary:
    num_symbols: int

    @property
    def symbol_offset(self) -> int:
        return 0

    @property
    def value_offset(self) -> int:
        return self.num_symbols

    @property
    def bos(self) -> int:
        return 2 * self.num_symbols

    @property
    def vocab_size(self) -> int:
        return 2 * self.num_symbols + 1

    def symbol(self, i: int) -> int:
        return self.symbol_offset + i

    def value(self, i: int) -> int:
        return self.value_offset + i

    def is_symbol(self, token: int) -> bool:
        return 0 <= token < self.num_symbols


def generate_real_examples(
    n: int,
    vocab: Vocabulary,
    sequence_pairs: int,
    task: str,
    seed: int,
    next_symbol_source: str = "context",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate arbitrary symbol-value mappings and repeated-symbol queries.

    Context format: BOS, s1,v1,s2,v2,...,sq.  The query symbol sq always
    appeared earlier. Base target=[value(sq)]. Extended target additionally
    asks the model to generate a next symbol and its mapped value.

    next_symbol_source:
      "context" (default): the next symbol is drawn uniformly from the distinct
          symbols shown in the context, so its value is always inferable in
          context. The symbol itself is unpredictable by design (uniform), which
          gives a known reference distribution against which collapse of the
          generated-symbol distribution can be measured.
      "any": drawn uniformly from all symbols (the pilot design). In about 45%
          of examples the symbol never appears in context, so its value cannot
          be inferred and exact-match accuracy is capped far below 1.
    """
    if task not in {"base", "extended"}:
        raise ValueError("task must be 'base' or 'extended'")
    if next_symbol_source not in {"context", "any"}:
        raise ValueError("next_symbol_source must be 'context' or 'any'")
    rng = np.random.default_rng(seed)
    x, y = [], []
    for _ in range(n):
        permutation = rng.permutation(vocab.num_symbols)
        shown = rng.choice(vocab.num_symbols, size=sequence_pairs, replace=True)
        query = int(rng.choice(shown))
        context = [vocab.bos]
        for s in shown:
            context += [vocab.symbol(int(s)), vocab.value(int(permutation[s]))]
        context.append(vocab.symbol(query))
        target = [vocab.value(int(permutation[query]))]
        if task == "extended":
            if next_symbol_source == "context":
                next_symbol = int(rng.choice(np.unique(shown)))
            else:
                next_symbol = int(rng.integers(vocab.num_symbols))
            target += [vocab.symbol(next_symbol), vocab.value(int(permutation[next_symbol]))]
        x.append(context)
        y.append(target)
    return np.asarray(x, dtype=np.int64), np.asarray(y, dtype=np.int64)


def context_mapping(context: np.ndarray, vocab: Vocabulary) -> dict[int, int]:
    """Return the symbol -> value-token mapping visible in one context."""
    pairs = context[1:-1]
    return {int(pairs[i]): int(pairs[i + 1]) for i in range(0, len(pairs), 2)}


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


def make_autoregressive_inputs(context: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Append shifted targets so each output is predicted autoregressively."""
    if targets.size(1) == 1:
        return context
    return torch.cat([context, targets[:, :-1]], dim=1)


"""Text-message corpus and a long-range memory task. No downloads."""
import numpy as np

_STARTS = ["running", "stuck in traffic,", "on the train,", "just left,",
           "parking now,", "sorry,", "heads up,", "quick one,"]
_LATE = ["5 mins late", "10 mins late", "about 20 mins late", "a bit late",
         "there in 5", "nearly there", "almost home", "outside now"]
_ASKS = ["are you free for dinner on friday?", "fancy a coffee tomorrow?",
         "can you grab milk on the way?", "did you feed the cat?",
         "what time is the match?", "shall we book the restaurant?",
         "are we still on for saturday?", "have you seen my keys?"]
_REPLIES = ["sounds good", "yes please", "no worries", "see you soon",
            "love that idea", "on my way", "thanks so much", "will do"]
_ENDS = ["", " x", " :)", " thanks", " see you there"]


def messages(n=3000, seed=0):
    """Generate realistic short texts with recurring structure.

    The structure is learnable (phrases recur), but there is genuine
    variety, so the model has to learn patterns rather than memorise.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        kind = rng.integers(3)
        if kind == 0:
            m = f"{rng.choice(_STARTS)} {rng.choice(_LATE)}"
        elif kind == 1:
            m = str(rng.choice(_ASKS))
        else:
            m = str(rng.choice(_REPLIES))
        out.append(m + str(rng.choice(_ENDS)))
    return out


def corpus(n=3000, seed=0):
    """One long string, messages separated by newlines."""
    return "\n".join(messages(n, seed)) + "\n"


class Vocab:
    """Characters <-> integer ids. Every language model starts here.

    Real LLMs use sub-word tokens (pieces like 'ing', ' the') instead of
    single characters, but the idea is identical.
    """

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    def __len__(self):
        return len(self.chars)

    def encode(self, s):
        return np.array([self.stoi[c] for c in s], dtype=np.int64)

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def recall_task(n=2000, distance=20, n_codes=8, seed=0):
    """Can the model remember something it saw `distance` steps ago?

    Each sequence:   [code]  [distance random filler tokens]  [?]
    Target: the code from the very first position.

    Framed as: an order number at the start of a long message, which you
    need when you reach the end. Trivial for attention. Brutal for an RNN.

    Tokens 0..n_codes-1 are codes, n_codes..n_codes+9 are filler,
    n_codes+10 is the '?' query token.
    """
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, n_codes, n)
    filler = rng.integers(n_codes, n_codes + 10, (n, distance))
    query = np.full((n, 1), n_codes + 10)
    X = np.concatenate([codes[:, None], filler, query], axis=1)
    return X.astype(np.int64), codes.astype(np.int64), n_codes + 11

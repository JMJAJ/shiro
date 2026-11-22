"""Reversible word tokenizer + vocabulary trie for token->byte marginalisation.

Token ids:   0..255  single bytes          256..V-1  letter-run words (len>=2)
             V       PAD (context before the start of a stream)

The byte stream is split into maximal ASCII letter runs and single non-letter
bytes. A run that is in the vocabulary becomes one word token; an OOV run
becomes its letters as byte tokens. Decoding = concatenating token bytes.

Byte records (what the NN tells the CM about each byte):
  kind 0: first byte of a token run / non-letter token: D(b) = mass of all
          vocab strings starting with b, under P_t.
  kind 1: inside a letter run (prefix s = node) or the delimiter right after
          it: letters get the mass of vocab strings extending s by b; non-letters
          get P_t(s exact) * Q_t(b); plus the OOV mass P_t(byte token s[0])
          spread by a generic letter/delimiter distribution.
  kind 2: prefix left the vocabulary trie, so the run is OOV and was emitted as
          byte tokens: the decoder knows this, and the NN's own prediction for that
          byte token (P_{t+k} over ids 0..255) is causal. Also used for the
          delimiter after such a run.
Every record uses the distribution P_t of the run's FIRST token, because a
decoder cannot consume the run until its delimiter has been decoded.
"""
from bisect import bisect_left, bisect_right
import numpy as np

LETTER = np.zeros(256, dtype=bool)
for c in range(ord("A"), ord("Z") + 1):
    LETTER[c] = True
for c in range(ord("a"), ord("z") + 1):
    LETTER[c] = True


def split_runs(data: bytes):
    """Yield (start, end, is_run) over the data: letter runs and single non-letter bytes."""
    a = np.frombuffer(data, dtype=np.uint8)
    isl = LETTER[a]
    n = len(a)
    # run starts: letter whose previous is not a letter
    prev = np.concatenate([[False], isl[:-1]])
    starts = np.flatnonzero(isl & ~prev)
    nxt = np.concatenate([isl[1:], [False]])
    ends = np.flatnonzero(isl & ~nxt) + 1
    return a, isl, starts, ends


class Vocab:
    def __init__(self, words):
        # words: list of bytes (len>=2), all letter runs
        self.words = list(words)
        self.V = 256 + len(self.words)
        self.PAD = self.V
        self.word_id = {w: 256 + i for i, w in enumerate(self.words)}
        strings = [bytes([b]) for b in range(256)] + self.words
        order = sorted(range(self.V), key=lambda i: strings[i])
        self.sorted_ids = np.array(order, dtype=np.int64)  # sorted position -> token id
        self.S = [strings[i] for i in order]  # sorted strings
        self._build_trie()

    @staticmethod
    def build(data: bytes, size: int, maxlen: int = 24):
        a, isl, starts, ends = split_runs(data)
        counts = {}
        mv = memoryview(data)
        for s, e in zip(starts.tolist(), ends.tolist()):
            if 2 <= e - s <= maxlen:
                w = bytes(mv[s:e])
                counts[w] = counts.get(w, 0) + 1
        # rank by bytes saved-ish: count * (len - 1)
        ranked = sorted(counts.items(), key=lambda kv: -kv[1] * (len(kv[0]) - 1))
        words = [w for w, c in ranked[: max(0, size - 256)] if c >= 2]
        return Vocab(words)

    def _range(self, prefix: bytes):
        lo = bisect_left(self.S, prefix)
        hi = bisect_right(self.S, prefix + b"\xff" * 32)
        return lo, hi

    def _build_trie(self):
        """Nodes = every prefix (incl. root) of a sorted string that is a letter run
        prefix or the root. For each node: child mass ranges lo/hi[node, b] in the
        sorted order, and the sorted index of the exact string (or -1)."""
        prefixes = {b""}
        for w in self.words:
            for k in range(1, len(w) + 1):
                prefixes.add(w[:k])
        for b in range(256):
            if LETTER[b]:
                prefixes.add(bytes([b]))
        plist = sorted(prefixes)
        self.node_of = {p: i for i, p in enumerate(plist)}
        N = len(plist)
        lo = np.zeros((N, 256), dtype=np.int32)
        hi = np.zeros((N, 256), dtype=np.int32)
        exact = np.full(N, -1, dtype=np.int64)
        child = np.full((N, 256), -1, dtype=np.int32)
        sidx = {s: i for i, s in enumerate(self.S)}
        for i, p in enumerate(plist):
            plo, phi = self._range(p)
            if p in sidx:
                exact[i] = sidx[p]
            # walk children inside [plo, phi)
            j = plo + (1 if p in sidx else 0)
            d = len(p)
            while j < phi:
                b = self.S[j][d]
                clo, chi = self._range(p + bytes([b]))
                lo[i, b] = clo
                hi[i, b] = chi
                if p + bytes([b]) in self.node_of:
                    child[i, b] = self.node_of[p + bytes([b])]
                j = chi
        self.n_nodes = N
        self.byte_sorted = np.array([sidx[bytes([b])] for b in range(256)], dtype=np.int64)
        # exact mass only for real word tokens (depth >= 2); depth-1 is OOV mass
        for i, p in enumerate(plist):
            if len(p) < 2:
                exact[i] = -1
        self.lo, self.hi, self.exact, self.child = lo, hi, exact, child

    def tokenize(self, data: bytes):
        """Returns tokens (int32), and per-byte records (tok_index, node, kind) with
        kind -1 meaning 'none'. Also run_start flag per token and the Q target
        (byte after the run, or -1) per token."""
        a, isl, starts, ends = split_runs(data)
        n = len(a)
        toks = []
        q_target = []
        rec_tok = np.full(n, -1, dtype=np.int64)
        rec_node = np.full(n, -1, dtype=np.int32)
        rec_kind = np.full(n, -1, dtype=np.int8)
        rec_first = np.zeros(n, dtype=np.uint8)
        mv = memoryview(data)
        child = self.child
        root = self.node_of[b""]
        i = 0
        ri = 0
        nruns = len(starts)
        st_list = starts.tolist()
        en_list = ends.tolist()
        while i < n:
            if ri < nruns and st_list[ri] == i:
                s, e = i, en_list[ri]
                ri += 1
                w = bytes(mv[s:e])
                t = len(toks)
                wid = self.word_id.get(w)
                if wid is not None:
                    toks.append(wid)
                    q_target.append(a[e] if e < n else -1)
                else:
                    for k, c in enumerate(w):
                        toks.append(c)
                        q_target.append(a[e] if (k == 0 and e < n) else -1)
                # records for run bytes
                node = root
                for k in range(s, e):
                    if node < 0:
                        # OOV interior: byte token t + (k - s) predicts this letter
                        rec_tok[k] = t + (k - s)
                        rec_node[k] = root
                        rec_kind[k] = 2
                        continue
                    rec_tok[k] = t
                    rec_node[k] = node
                    rec_kind[k] = 0 if k == s else 1
                    rec_first[k] = a[s]
                    node = child[node, a[k]]
                # delimiter right after the run
                if e < n:
                    if node >= 0:
                        rec_tok[e] = t
                        rec_node[e] = node
                        rec_kind[e] = 1
                        rec_first[e] = a[s]
                    else:
                        # known-OOV run: the delimiter is byte token t + len(run)
                        rec_tok[e] = t + (e - s)
                        rec_node[e] = root
                        rec_kind[e] = 2
                    toks.append(a[e])
                    q_target.append(-1)
                    i = e + 1
                else:
                    i = e
            else:
                t = len(toks)
                toks.append(a[i])
                q_target.append(-1)
                if rec_kind[i] < 0:
                    rec_tok[i] = t
                    rec_node[i] = root
                    rec_kind[i] = 0
                i += 1
        return (
            np.array(toks, dtype=np.int64),
            np.array(q_target, dtype=np.int64),
            rec_tok,
            rec_node,
            rec_kind,
            rec_first,
        )

    def detokenize(self, toks):
        strings = [bytes([b]) for b in range(256)] + self.words
        return b"".join(strings[t] for t in toks)

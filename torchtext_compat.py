"""Drop-in replacements for the deprecated torchtext utilities used in this project."""
import re
import os
import gzip
import urllib.request
from collections import Counter

_MULTI30K_BASE = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw"

_MULTI30K_URLS = {
    "train": {"en": f"{_MULTI30K_BASE}/train.en.gz", "de": f"{_MULTI30K_BASE}/train.de.gz"},
    "valid": {"en": f"{_MULTI30K_BASE}/val.en.gz", "de": f"{_MULTI30K_BASE}/val.de.gz"},
}

_BASIC_ENGLISH_RE = re.compile(r"[A-Za-z]+|[0-9]+|[^\s]")


def get_tokenizer(name="basic_english"):
    if name != "basic_english":
        raise ValueError(f"Only 'basic_english' tokenizer is supported, got '{name}'")

    def _tokenize(text):
        return _BASIC_ENGLISH_RE.findall(text.lower())

    return _tokenize


class Vocab:
    def __init__(self, ordered_tokens, specials):
        self.itos = list(specials) + list(ordered_tokens)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self._default_index = None

    def set_default_index(self, idx):
        self._default_index = idx

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token):
        if token in self.stoi:
            return self.stoi[token]
        if self._default_index is not None:
            return self._default_index
        raise KeyError(token)

    def __call__(self, tokens):
        return [self[t] for t in tokens]


def build_vocab_from_iterator(iterator, specials=None):
    specials = specials or []
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)
    ordered = [tok for tok, _ in counter.most_common()]
    ordered = [tok for tok in ordered if tok not in specials]
    v = Vocab(ordered, specials)
    return v


def _download_and_cache(url, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        gz_path = local_path + ".gz"
        urllib.request.urlretrieve(url, gz_path)
        with gzip.open(gz_path, "rb") as f_in, open(local_path, "wb") as f_out:
            f_out.write(f_in.read())
        os.remove(gz_path)
    return local_path


def Multi30k(split="train", language_pair=("en", "de")):
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "multi30k")
    src_lang, tgt_lang = language_pair

    src_path = _download_and_cache(
        _MULTI30K_URLS[split][src_lang],
        os.path.join(cache_dir, f"{split}.{src_lang}"),
    )
    tgt_path = _download_and_cache(
        _MULTI30K_URLS[split][tgt_lang],
        os.path.join(cache_dir, f"{split}.{tgt_lang}"),
    )

    with open(src_path, encoding="utf-8") as sf, open(tgt_path, encoding="utf-8") as tf:
        for src_line, tgt_line in zip(sf, tf):
            yield src_line.strip(), tgt_line.strip()

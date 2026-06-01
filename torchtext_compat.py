"""Drop-in replacements for the deprecated torchtext utilities used in this project."""
import re
import os
import io
import urllib.request
import tarfile
from collections import Counter, OrderedDict

_MULTI30K_URLS = {
    "train": "http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz",
    "valid": "http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz",
}

_MULTI30K_FILENAMES = {
    "train": {"en": "train.en", "de": "train.de"},
    "valid": {"en": "val.en", "de": "val.de"},
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


def _download_and_cache(url, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.basename(url)
    local_path = os.path.join(cache_dir, fname)
    if not os.path.exists(local_path):
        urllib.request.urlretrieve(url, local_path)
    return local_path


def Multi30k(split="train", language_pair=("en", "de")):
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "multi30k")
    src_lang, tgt_lang = language_pair
    src_file = _MULTI30K_FILENAMES[split][src_lang]
    tgt_file = _MULTI30K_FILENAMES[split][tgt_lang]

    src_path = os.path.join(cache_dir, src_file)
    tgt_path = os.path.join(cache_dir, tgt_file)

    if not (os.path.exists(src_path) and os.path.exists(tgt_path)):
        tar_path = _download_and_cache(_MULTI30K_URLS[split], cache_dir)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(cache_dir)

    with open(src_path, encoding="utf-8") as sf, open(tgt_path, encoding="utf-8") as tf:
        for src_line, tgt_line in zip(sf, tf):
            yield src_line.strip(), tgt_line.strip()

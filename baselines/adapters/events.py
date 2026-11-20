"""Train-only event/word vectors and raw historical log inputs."""
from __future__ import annotations

from functools import partial
from pathlib import Path
from collections import Counter
import random

import numpy as np
import pandas as pd

from baselines.author_imports import author_symbols


class EventFeatures:
    def __init__(self, raw, reference, fit_ends, output, kind, dimension=32, seed=42):
        if not set(fit_ends).issubset(raw.data.train):
            raise ValueError("Embedding may use training windows only")
        import fasttext
        self.raw, self.reference, self.dimension = raw, reference, dimension
        self.kind, self.cache = kind, {}
        output = Path(output)
        corpus = output / "embedding_train.txt"
        documents, labels = [], []
        d = raw.dataset
        for end in fit_ends:
            if kind == "hades":
                _, _, logs = raw.window(end)
                documents.extend(row["text"] for row in logs[:512] if row["text"])
            else:
                for service, events in enumerate(raw.event_tokens(end, reference)):
                    documents.append(" ".join(t for tokens in events.values() for t in tokens) or "__empty__")
                    labels.append((service, int(d.fault_type_targets[end])))
        documents = documents or ["__empty__"]
        if kind == "diagfusion":
            # Reuse the author's embedding augmentation and supervised fastText
            # sequence, with explicit finite per-class augmentation budget.
            api = author_symbols("diagfusion", "transforms/events/fasttext_with_DA.py", ["FastTextLab"],
                                  dict(random=random, np=np, pd=pd, Counter=Counter))
            author = api.FastTextLab.__new__(api.FastTextLab)
            author.nodes = list(d.services)
            author.anomaly_types = list(range(len(d.fault_type_names)))
            author.node_labels = {s: i for i, s in enumerate(d.services)}
            author.anomaly_type_labels = {i: i for i in author.anomaly_types}
            author.config = dict(train_path=str(corpus), vector_dim=dimension, minCount=1,
                                 epoch=5, sample_count=64, edit_count=1,
                                 train_da_path=str(output / "embedding_augmented.txt"))
            author.train_data = [f"{doc}\t__label__{service}{fault}" for doc, (service, fault) in zip(documents, labels)]
            corpus.write_text("\n".join(author.train_data)+"\n")
            author.method = partial(fasttext.train_supervised, thread=1, seed=seed, bucket=1000, verbose=0)
            # Author replacement requires at least two events to have a neighbor.
            # __empty__ documents remain valid training examples.
            author.w2v_DA()
            self.embedding = author.method(author.config["train_da_path"], dim=dimension,
                                            minCount=1, minn=0, maxn=0, epoch=5)
        else:
            corpus.write_text("\n".join(documents)+"\n")
            self.embedding = fasttext.train_unsupervised(str(corpus), model="skipgram", dim=dimension,
                                                         minCount=1, epoch=5, minn=0, maxn=0,
                                                         thread=1, bucket=1000, verbose=0)
        self.embedding.save_model(str(output / "event_embedding.bin"))

    def vector(self, text):
        words = text.split() or ["__empty__"]
        return np.mean([self.embedding.get_word_vector(w) for w in words], axis=0).astype(np.float32)

    def __call__(self, end):
        if end in self.cache:
            return self.cache[end]
        if self.kind == "hades":
            _, _, logs = self.raw.window(end)
            lo, hi = self.raw.bounds(end)
            features = np.zeros((5, 16, self.dimension), dtype=np.float32)
            counts = [0]*5
            for log in logs:
                bucket = int((log["available"]-lo)//60)
                if counts[bucket] < 16:
                    features[bucket, counts[bucket]] = self.vector(log["text"])
                    counts[bucket] += 1
        else:
            events = self.raw.event_tokens(end, self.reference)
            features = np.asarray([[self.vector(" ".join(e[m]) or "__empty__")
                                    for m in ("metric", "log", "trace")] for e in events], dtype=np.float32)
        self.cache[end] = features
        return features

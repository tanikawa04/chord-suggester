# coding: utf-8

import numpy as np


_TRAIN_DATASET_PATH = './data/train.txt'
_VAL_DATASET_PATH = './data/val.txt'

_BOS = '<s>'
_EOS = '</s>'


def load_dataset():
    with open(_TRAIN_DATASET_PATH) as f:
        train_songs = [[_BOS] + line.strip().split(' ') + [_EOS]
                       for line in f.readlines()]

    with open(_VAL_DATASET_PATH) as f:
        val_songs = [[_BOS] + line.strip().split(' ') + [_EOS]
                     for line in f.readlines()]

    c2i, i2c = build_vocab(sum(train_songs, []) + sum(val_songs, []))

    train = [[c2i[chord] for chord in chords] for chords in train_songs]
    val = [[c2i[chord] for chord in chords] for chords in val_songs]

    return train, val, c2i, i2c


def build_vocab(chords):
    symbols = sorted(set(chords))
    chord2idx = dict((c, i) for i, c in enumerate(symbols))
    idx2chord = dict((i, c) for i, c in enumerate(symbols))
    return chord2idx, idx2chord

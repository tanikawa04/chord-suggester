# coding: utf-8

import argparse

import numpy as np
import chainer
import chainer.functions as F
from chainer import Variable

from net import ChordNet
from util import load_dataset


_BOS = '<s>'
_EOS = '</s>'
_MAX_LEN = 16


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # This script is only CPU support.
    # parser.add_argument('--gpu', '-g', type=int, default=-1,
    #                     help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model', '-m', type=str,
                        default='result/chordnet.model',
                        help='Model file path')
    parser.add_argument('--unit', '-u', type=int, default=256,
                        help='Number of LSTM units in each layer')

    args = parser.parse_args()
    model_path = args.model
    n_unit = args.unit

    # FIXME 学習前段階でコード辞書をファイル化すべき
    _, _, c2i, i2c = load_dataset()
    eos_id = c2i[_EOS]

    model = ChordNet(len(c2i), n_unit)

    chainer.serializers.load_npz(model_path, model)

    while True:
        s = input('> ')
        # if len(s.strip()) == 0:
        #     continue

        s = _BOS + ' ' + s
        chord_ids = [c2i[chord] for chord in s.strip().split(' ')]

        with chainer.using_config('train', False):
            model.reset_state()

            for chord_id in chord_ids[:-1]:
                xs = Variable(np.array([chord_id], dtype=np.int32))
                model(xs)

            for i in range(_MAX_LEN):
                xs = Variable(np.array([chord_ids[-1]], dtype=np.int32))
                cid = sample(F.softmax(model(xs))[0].data)
                # cid = np.argmax(F.softmax(model(xs))[0].data)

                if cid == eos_id:
                    break

                chord_ids.append(cid)

        print(' '.join([i2c[cid] for cid in chord_ids[1:]]))
        print()

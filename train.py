# coding: utf-8

import argparse
from os import path

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable
from chainer import optimizers
from chainer import serializers
from chainer import cuda

from net import ChordNet
from util import load_dataset


_PAD_ID = -1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--bproplen', '-l', type=int, default=35,
                        help='Number of chords in each mini-batch '
                             '(= length of truncated BPTT)')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=256,
                        help='Number of LSTM units in each layer')

    args = parser.parse_args()
    batchsize = args.batchsize
    bplen = args.bproplen
    n_epoch = args.epoch
    grad_clip = args.gradclip
    out_path = args.out
    n_unit = args.unit

    train, val, c2i, _ = load_dataset()
    n_train = len(train)
    n_val = len(val)

    model = ChordNet(len(c2i), n_unit)

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    xp = np if args.gpu < 0 else cuda.cupy

    optimizer = optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))

    def make_batch(data, indexes):
        batch = [Variable(xp.array(data[i], dtype=xp.int32)) for i in indexes]
        max_len = max([d.shape[0] for d in batch])
        return F.pad_sequence(batch, length=max_len, padding=_PAD_ID)

    # epoch loop
    for i in range(1, n_epoch + 1):
        print('epoch {}'.format(i))

        sum_loss = 0
        step = 0

        perm = np.random.permutation(n_train)

        # batch loop
        for j in range(0, n_train, batchsize):
            loss = 0
            model.reset_state()

            batch = make_batch(train, perm[j:j + batchsize])

            # chord loop
            for k in range(batch.shape[1] - 1):
                y = model(batch[:, k])
                loss += F.softmax_cross_entropy(y, batch[:, k + 1])
                step += 1

                if (k + 1) % bplen == 0 or k == batch.shape[1] - 1:
                    model.cleargrads()
                    loss.backward()
                    loss.unchain_backward()
                    optimizer.update()

                    sum_loss += loss.data
                    loss = 0

        # validation
        val_sum_loss = 0
        val_step = 0

        for j in range(0, n_val, batchsize):
            model.reset_state()

            batch = make_batch(val, list(range(j, min(j + batchsize, n_val))))

            for k in range(batch.shape[1] - 1):
                y = model(batch[:, k])
                loss = F.softmax_cross_entropy(y, batch[:, k + 1])
                val_sum_loss += loss.data
                val_step += 1

        print('training loss: {:.4f}'.format(float(sum_loss / step)))
        print('validation loss: {:.4f}'.format(float(val_sum_loss / val_step)))

        with open(path.join(out_path, 'loss.csv'), 'a') as f:
            f.write('{},{:.4f},{:.4f}\n'.format(
                i, float(sum_loss / step), float(val_sum_loss / val_step)))

        serializers.save_npz(
            path.join(out_path, 'chordnet.model'), model)
        serializers.save_npz(
            path.join(out_path, 'chordnet.state'), optimizer)

#!/usr/bin/env python3
import argparse
import numpy as np


def topk_reference(x: np.ndarray, k: int):
    order = np.lexsort((np.arange(x.shape[1])[None, :].repeat(x.shape[0], axis=0), -x))
    idx = order[:, :k].astype(np.int32)
    vals = np.take_along_axis(x, idx, axis=1).astype(np.float32)
    return vals, idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--n', type=int, default=1024)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    x = rng.normal(size=(args.batch, args.n)).astype(np.float32)
    vals, idx = topk_reference(x, args.k)
    print({'shape': x.shape, 'k': args.k, 'values0': vals[0].tolist(), 'indices0': idx[0].tolist()})


if __name__ == '__main__':
    main()

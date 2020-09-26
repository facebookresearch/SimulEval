# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from simuleval.metrics.latency import (
    AverageLagging,
    AverageProportion,
    DifferentiableAverageLagging
)

delays = [1, 1, 2, 3, 5, 7, 7, 9]
src_len = 9


def test_al():
    metrics_from_equation = 0
    tgt_len = len(delays)
    gamma = tgt_len / src_len
    tau = 0
    for t_miuns_1, d in enumerate(delays):
        if d <= src_len:
            metrics_from_equation += d - t_miuns_1 / gamma
            tau = t_miuns_1 + 1

            if d == src_len:
                break
    metrics_from_equation /= tau

    al = AverageLagging(delays, src_len)

    assert al == metrics_from_equation, f"{al, metrics_from_equation}"


def test_ap():
    metrics_from_equation = 0
    tgt_len = len(delays)
    for d in delays:
        metrics_from_equation += d

    metrics_from_equation /= (src_len * tgt_len)

    ap = AverageProportion(delays, src_len)

    assert ap == metrics_from_equation, f"{ap, metrics_from_equation}"


def test_dal():
    metrics_from_equation = 0
    tgt_len = len(delays)
    gamma = tgt_len / src_len
    g_prime_last = 0
    for i_miuns_1, g in enumerate(delays):
        if i_miuns_1 + 1 == 1:
            g_prime = g
        else:
            g_prime = max([g, g_prime_last + 1 / gamma])

        metrics_from_equation += g_prime - i_miuns_1 / gamma
        g_prime_last = g_prime

    metrics_from_equation /= tgt_len

    ap = DifferentiableAverageLagging(delays, src_len)

    assert ap == metrics_from_equation, f"{ap, metrics_from_equation}"

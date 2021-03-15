# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numbers


def latency_metric(func):
    def prepare_latency_metric(
        delays,
        src_lens,
        target_padding_mask=None,
    ):
        """
        delays: bsz, tgt_len
        src_lens: bsz
        target_padding_mask: bsz, tgt_len
        """
        if isinstance(delays, list):
            delays = torch.FloatTensor(delays)

        if len(delays.size()) == 1:
            delays = delays.view(1, -1)

        if isinstance(src_lens, list):
            src_lens = torch.FloatTensor(src_lens)

        if isinstance(src_lens, numbers.Number):
            src_lens = torch.FloatTensor([src_lens])

        src_lens = src_lens.type_as(delays)

        if target_padding_mask is not None:
            tgt_lens = target_padding_mask.sum(dim=1)
            delays = delays.masked_fill(target_padding_mask, 0)
        else:
            tgt_lens = torch.ones_like(src_lens) * delays.size(1)

        return delays, src_lens, tgt_lens, target_padding_mask

    def latency_wrapper(
        delays, src_lens, ref_len=None, target_padding_mask=None
    ):
        delays, src_lens, tgt_lens, target_padding_mask = prepare_latency_metric(
            delays, src_lens, target_padding_mask)
        return func(delays, src_lens, tgt_lens, ref_len, target_padding_mask)

    return latency_wrapper


@latency_metric
def AverageProportion(
    delays, src_lens, tgt_lens, ref_len=None, target_padding_mask=None):
    """
    Function to calculate Average Proportion from
    Can neural machine translation do simultaneous translation?
    (https://arxiv.org/abs/1606.02012)
    Delays are monotonic steps, range from 1 to src_len.
    Give src x tgt y, AP is calculated as:
    AP = 1 / (|x||y]) sum_i^|Y| deleys_i
    """
    if target_padding_mask is not None:
        AP = torch.sum(delays.masked_fill(target_padding_mask, 0), dim=1)
    else:
        AP = torch.sum(delays, dim=1)

    AP = AP / (src_lens * tgt_lens)
    return AP


@latency_metric
def AverageLagging(delays, src_lens, tgt_lens, ref_len=None, target_padding_mask=None):
    """
    Function to calculate Average Lagging from
    STACL: Simultaneous Translation with Implicit Anticipation
    and Controllable Latency using Prefix-to-Prefix Framework
    (https://arxiv.org/abs/1810.08398)
    Delays are monotonic steps, range from 1 to src_len.
    Give src x tgt y, AP is calculated as:
    AL = 1 / tau sum_i^tau delays_i - (i - 1) / gamma
    Where
    gamma = |y| / |x|
    tau = argmin_i(delays_i = |x|)
    """
    _, tgt_len = delays.size()
    if ref_len is None:
        ref_len = tgt_len

    # tau = argmin_i(delays_i = |x|)
    # Only consider one delays that has already larger than src_lens
    lagging_padding_mask = delays >= src_lens.unsqueeze(1)
    # Padding one token at beginning to consider at least one delays that
    # larget than src_lens
    lagging_padding_mask = torch.nn.functional.pad(
        lagging_padding_mask, (1, 0))[:, :-1]

    if target_padding_mask is not None:
        lagging_padding_mask = lagging_padding_mask.masked_fill(
            target_padding_mask, True)

    # oracle delays are the delay for the oracle system which goes diagonally
    oracle_delays = (
        torch.arange(ref_len)
        .unsqueeze(0)
        .type_as(delays)
        .expand([delays.size(0), ref_len])
    ) * src_lens / ref_len

    if delays.size(1) < ref_len:
        oracle_delays = oracle_delays[:, :delays.size(1)]

    if delays.size(1) > ref_len:
        oracle_delays = torch.cat(
            [
                oracle_delays,
                oracle_delays[:,-1]
                * oracle_delays.new_ones(
                    [delays.size(0), delays.size(1) - ref_len]
                )
            ],
            dim=1
        )

    lagging = delays - oracle_delays
    lagging = lagging.masked_fill(lagging_padding_mask, 0)

    tau = (1 - lagging_padding_mask.type_as(lagging)).sum(dim=1)
    AL = lagging.sum(dim=1) / tau

    return AL


@latency_metric
def DifferentiableAverageLagging(
        delays, src_lens, tgt_lens, ref_len=None, target_padding_mask=None
    ):

    _, tgt_len = delays.size()

    gamma = tgt_lens / src_lens
    new_delays = torch.zeros_like(delays)

    for i in range(tgt_len):
        if i == 0:
            new_delays[:, i] = delays[:, i]
        else:
            new_delays[:, i] = torch.cat(
                [
                    new_delays[:, i - 1].unsqueeze(1) + 1 / gamma,
                    delays[:, i].unsqueeze(1)
                ],
                dim=1
            ).max(dim=1).values

    DAL = (
        new_delays -
        torch.arange(tgt_len).unsqueeze(0).type_as(
            delays).expand_as(delays) / gamma
    )
    if target_padding_mask is not None:
        DAL = DAL.masked_fill(target_padding_mask, 0)

    DAL = DAL.sum(dim=1) / tgt_len
    return DAL

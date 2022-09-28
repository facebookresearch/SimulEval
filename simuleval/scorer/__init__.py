# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .scorer import SentenceLevelTextScorer, SentenceLevelSpeechScorer
from .scorer import compute_score


def build_scorer(dataloader, args):
    if args.target_type == "text":
        return SentenceLevelTextScorer(dataloader, args)
    elif args.target_type == "speech":
        return SentenceLevelSpeechScorer(dataloader, args)
    else:
        raise NotImplementedError

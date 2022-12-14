# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from .cli import evaluate, EVALUATION_SYSTEM_LIST
from simuleval import options


def entrypoint(klass):
    EVALUATION_SYSTEM_LIST.append(klass)
    return klass


def eval_agent(klass, **kwargs):
    parser = options.general_parser()
    options.add_evaluator_args(parser)
    options.add_dataloader_args(parser)

    # To make sure all args are valid
    klass.add_args(parser)
    string = ""
    for key, value in kwargs.items():
        if type(value) is not bool:
            string += f" --{key.replace('_', '-')} {value}"
        else:
            string += f" --{key.replace('_', '-')}"

    args = parser.parse_args(string.split())

    evaluate(klass, args)

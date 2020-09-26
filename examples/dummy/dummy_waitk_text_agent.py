# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from simuleval.agents import TextAgent
from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS


class DummyWaitkTextAgent(TextAgent):

    data_type = "text"

    def __init__(self, args):
        super().__init__(args)
        self.waitk = args.waitk
        # Initialize your agent here, for example load model, vocab, etc

    @staticmethod
    def add_args(parser):
        # Add additional command line arguments here
        parser.add_argument("--waitk", type=int, default=3)

    def policy(self, states):
        # Make decision here
        if len(states.source) - len(states.target) < self.waitk and not states.finish_read():
            return READ_ACTION
        else:
            return WRITE_ACTION

    def predict(self, states):
        # predict token here
        if states.finish_read():
            if states.target.length() == states.source.length():
                return DEFAULT_EOS

        return f"word_{len(states.target)}"

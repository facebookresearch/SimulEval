# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from cmath import log
from typing import Dict, List, Union
from simuleval.states import TextStates, SpeechStates
from simuleval.online.client import Client
from simuleval import DEFAULT_EOS


class Agent(object):
    source_type = None
    target_type = None

    def __init__(self, args) -> None:
        assert self.source_type
        assert self.target_type
        self.args = args
        self.client = None
        self.source_segment_size = 1
        self.reset()

    def reset(self) -> None:
        self.index = None
        self.is_finish_eval = False
        self.is_finish_read = False
        self.states = {"source": [], "target": [], "actions": []}

    def eval(self, index: int) -> None:
        self.index = index
        while not self.is_finish_eval:
            self.policy()
        self.write(DEFAULT_EOS)

    def finish_eval(self) -> None:
        self.is_finish_eval = True

    def finish_read(self) -> None:
        self.is_finish_read = True

    def set_client(self, client: Client) -> None:
        self.client = client

    def set_index(self, index: int) -> None:
        self.index = index

    def read(self):
        if self.is_finish_read:
            return
        info = self.client.get_source(
            self.index, {"segment_size": self.source_segment_size}
        )
        self.states["source"].append(info)
        return self.process_read(info)

    def write(self, predictions: Union[List, str]) -> None:
        predictions = self.process_write(predictions)
        if isinstance(predictions, str):
            predictions = [predictions]
        for pred in predictions:
            self.client.send_hypo(self.index, pred)

    @staticmethod
    def add_args(parser) -> None:
        # Add additional command line arguments here
        pass

    def policy(self) -> None:
        # Make decision here
        assert NotImplementedError

    def process_read(self, info: Dict) -> Dict:
        # Process the read here
        return info

    def process_write(self, predictions: Union[List, str]) -> List:
        # Process the write here
        return predictions

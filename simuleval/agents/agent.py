# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from simuleval.states import TextStates, SpeechStates


class Agent(object):
    data_type = None

    def __init__(self, args):
        assert self.data_type is not None

    def states_type(self, args):
        if self.data_type == "text":
            return TextStates
        elif self.data_type == "speech":
            return SpeechStates
        else:
            raise NotImplementedError

    def segment_to_units(self, segment, states):
        return [segment]

    def units_to_segment(self, unit_queue, states):
        return unit_queue.pop()

    def update_states_read(self, states):
        pass

    def update_states_write(self, states):
        pass

    def build_states(self, args, client, sentence_id):
        # Initialize states here, for example add customized entry to states
        # This funcion will be caused at begining of every new sentence
        states = self.states_type(args)(args, client, sentence_id, self)
        self.initialize_states(states)
        return states

    def initialize_states(self, states):
        pass

    @staticmethod
    def add_args(parser):
        # Add additional command line arguments here
        pass

    def policy(self, states):
        # Make decision here
        assert NotImplementedError

    def predict(self, states):
        # predict token here
        assert NotImplementedError

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import logging

sys.path.append("..")
from simuleval.data.segments import Segment, segment_from_json_string
from simuleval.evaluator import SentenceLevelEvaluator
from examples.speech_to_text.demo_remote import DemoRemote
import requests

logger = logging.getLogger("simuleval.remote_evaluator")


class RemoteEvaluator:
    def __init__(self, evaluator: SentenceLevelEvaluator) -> None:
        self.evaluator = evaluator
        self.address = evaluator.args.remote_address
        self.port = evaluator.args.remote_port
        self.source_segment_size = evaluator.args.source_segment_size
        self.base_url = f"http://{self.address}:{self.port}"
        self.is_demo = evaluator.args.demo

    def send_source(self, segment: Segment):
        url = f"{self.base_url}/input"
        requests.put(url, data=segment.json())

    def receive_prediction(self) -> Segment:
        url = f"{self.base_url}/output"
        r = requests.get(url)
        return segment_from_json_string(r.text)

    def system_reset(self):
        requests.post(f"{self.base_url}/reset")

    def results(self):
        return self.evaluator.results()

    def remote_eval(self):
        if self.is_demo:
            demo = DemoRemote(self.source_segment_size / 1000)  # ms -> s
            demo.record_audio()
            return

        for instance in self.evaluator.iterator:
            self.system_reset()
            while not instance.finish_prediction:
                self.send_source(instance.send_source(self.source_segment_size))
                # instance.py line 275, returns a segment object with all the floats in the 500 ms range

                output_segment = self.receive_prediction()
                # gets the prediction in text! like "This"...
                # refreshes each time. "This" for the 1st, "is" for the second

                instance.receive_prediction(output_segment)
                # instance.py line 190
                # processes data, gets in a prediction list with ["This", "is"] on 2nd iteration
            self.evaluator.write_log(instance)

        self.evaluator.dump_results()

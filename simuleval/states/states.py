# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from collections import deque
from simuleval import DEFAULT_EOS


def check_status_method(action):
    def _checked_methods(func):
        def wrapper(*args):
            if not args[0].status[action]:
                return
            else:
                return func(*args)
        return wrapper
    return _checked_methods


class Entry(object):

    def __init__(self, value, new_value_type):
        self.value = value
        self.new_value_type = new_value_type

    def update(self, new_value):
        if self.new_value_type:
            assert isinstance(
                new_value, self.new_value_type), f"{new_value}, {self.new_value_type}"
        self.value += self.preprocess(new_value)

    def preprocess(self, value):
        return value


class ListEntry(Entry):
    def __init__(self, new_value_type=None, value=None):
        if value is None:
            value = list()
        super().__init__(value, new_value_type)

    def preprocess(self, value):
        return [value]

    def __len__(self):
        return self.length()

    def length(self):
        return len(self.value)

    def info(self):
        return {
            "type": str(self.new_value_type),
            "length": self.__len__(),
            "value": " ".join(str(x) for x in self.value)
        }

    def append(self, value):
        self.update(value)

    def __repr__(self):
        return json.dumps(self.info(), indent=4)

    def __str__(self):
        return json.dumps(self.info(), indent=4)

    def __iter__(self):
        for value in self.value:
            yield value

    def __getitem__(self, idx):
        return self.value[idx]


class SignalEntry(ListEntry):
    def info(self):
        return {
            "type": str(self.new_value_type),
            "length": self.__len__(),
            "value": len(self.value)
        }


class ScalarEntry(object):
    def __init__(self, value=0):
        super.__init__(value, None)


class QueueEntry(ListEntry):
    def __init__(self, new_value_type=None, value=None):
        if value is None:
            value = deque()
        super().__init__(new_value_type, value)

    def push(self, value):
        self.value.append(value)

    def pop(self):
        if len(self):
            return self.value.popleft()
        else:
            return None

    def empty(self):
        return len(self) == 0

    def clear(self):
        return self.value.clear()


class SampleEntry(ListEntry):
    def preprocess(self, value):
        return value


class BiSideEntries(object):
    def __init__(self, source_entry, target_entry):
        self.source = source_entry
        self.target = target_entry


class BaseStates(object):
    def __init__(self, args, client, instance_id, agent):
        self.client = client
        self.instance_id = instance_id
        self.status = {"read": True, "write": True}
        self.agent = agent
        self.init_entries()

    def init_entries(self):
        self.unit_queue = BiSideEntries(QueueEntry(), QueueEntry())
        self.units = BiSideEntries(ListEntry(), ListEntry(str))
        self.segments = BiSideEntries(ListEntry(), ListEntry(str))

    def finish_read(self):
        return not self.status["read"]

    def finish_hypo(self):
        return not self.status["write"]

    @property
    def source(self):
        return self.units.source

    @property
    def target(self):
        return self.units.target

    def update_target_segment(self):
        segment = self.units_to_segment(self.unit_queue.target)
        if segment is None:
            return

        if type(segment) is str:
            segment = [segment]

        for seg in segment:
            self.segments.target.append(seg)
            self.client.send_hypo(self.instance_id, seg)
            if seg == DEFAULT_EOS:
                self.status["write"] = False
                break

    def get_info_from_server(self, num_segment):
        return self.client.get_source(self.instance_id)

    def update_source_segment(self, num_segment=1):
        # Read a segment from server
        info = self.get_info_from_server(num_segment)
        segment = info["segment"]

        self.segments.source.append(segment)

        if (
            info.get("finish", False) is True
            or segment in [DEFAULT_EOS]
        ):
            self.status["read"] = False
            # Receive an EOS from server
            if segment in [DEFAULT_EOS]:
                return

        # Preprocess a segment into units
        units = self.segment_to_units(segment)

        # Update the source unit entry
        for unit in units:
            self.unit_queue.source.push(unit)

    def segment_to_units(self, segment):
        # Split segment into units
        return self.agent.segment_to_units(segment, self)

    def units_to_segment(self, unit_queue):
        # Merge unit into segments
        segment = self.agent.units_to_segment(unit_queue, self)
        return segment

    def summarize(self):
        return {
            "finish_read": self.finish_read(),
            "finish_hypo": self.finish_hypo(),
            "segments": {
                "source": self.segments.source.info(),
                "target": self.segments.target.info(),
            },
            "units": {
                "source": self.units.source.info(),
                "target": self.units.target.info(),
            },
            "unit_queue": {
                "source": self.unit_queue.source.info(),
                "target": self.unit_queue.target.info(),
            },
        }

    def __str__(self):
        return json.dumps(self.summarize(), indent=4)

    def __repr__(self):
        return json.dumps(self.summarize(), indent=4)

    @check_status_method("read")
    def update_source(self, num_segment=1):
        # If unit queue is empty, try to update a segment from server
        if self.unit_queue.source.empty():
            self.update_source_segment(num_segment)

        if not self.unit_queue.source.empty():
            self.units.source.append(
                self.unit_queue.source.pop()
            )

        self.agent.update_states_read(self)

    @check_status_method("write")
    def update_target(self, unit):
        # update unit list and queue
        self.unit_queue.target.push(unit)
        self.units.target.append(unit)

        # update the segement
        self.update_target_segment()
        self.agent.update_states_write(self)

    def pop_all_units_source(self):
        while not self.unit_queue.source.empty():
            self.units.source.append(
                self.unit_queue.source.pop()
            )

from typing import List
from simuleval.data.segments import Segment
from .agent import GenericAgent


class AgentPipeline(GenericAgent):
    """A pipeline of agents

    Attributes:
        pipeline (list): a list of agent classes.

    """

    pipeline: list = []

    def __init__(self, module_list: List[GenericAgent]) -> None:
        self.module_list = module_list

    @property
    def source_type(self) -> str:
        return self.pipeline[0].source_type

    @property
    def target_type(self) -> str:
        return self.pipeline[-1].target_type

    def reset(self) -> None:
        for module in self.module_list:
            module.reset()

    def push(self, segment: Segment) -> None:
        for module in self.module_list[:-1]:
            if not segment.is_empty:
                module.push(segment)
                segment = module.pop()
            else:
                return
        self.module_list[-1].push(segment)

    def pop(self) -> None:
        return self.module_list[-1].pop()

    def pushpop(self, segment: Segment) -> Segment:
        for module in self.module_list:
            segment = module.pushpop(segment)
            if segment.is_empty:
                break
        return segment

    @classmethod
    def add_args(cls, parser) -> None:
        for module_class in cls.pipeline:
            module_class.add_args(parser)

    @classmethod
    def from_args(cls, args):
        assert len(cls.pipeline) > 0
        if len(cls.pipeline) > 1:
            for i in range(1, len(cls.pipeline)):
                if cls.pipeline[i].source_type != cls.pipeline[i - 1].target_type:
                    raise RuntimeError(
                        f"{cls.pipeline[i]}.source_type({cls.pipeline[i].source_type}) != {cls.pipeline[i-1]}.target_type({cls.pipeline[i - 1].target_type})"
                    )

        return cls([module_class.from_args(args) for module_class in cls.pipeline])

    def __repr__(self) -> str:
        pipline_str = "\n\t".join(
            module_class.__name__ for module_class in self.pipeline
        )
        return f"{self.__class__.__name__}(\n\t{pipline_str}\n)"

    def __str__(self) -> str:
        return self.__repr__()

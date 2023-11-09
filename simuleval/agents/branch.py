from typing import Dict, List, Optional
from argparse import Namespace
from simuleval.agents.pipeline import AgentPipeline
from simuleval.data.segments import Segment
from simuleval.agents.agent import GenericAgent, AgentStates


class BranchedAgentPipelineStates(AgentStates):
    def __init__(self, states_dict: Dict[str, AgentStates]):
        self.states_dict = states_dict
        super().__init__()

    def reset(self) -> None:
        super().reset()
        for states in self.states_dict.values():
            for s in states:
                s.reset()

    def update_source(self, segment: Segment):
        self.source_finished = segment.finished
        for states in self.states_dict.values():
            states[0] = segment.finished

    def update_target(self, segment: Segment):
        self.target_finished = segment.finished
        for states in self.states_dict.values():
            states[1] = segment.finished


class BranchedAgentPipeline(AgentPipeline):
    """
    Select different agent branch to use

    Args:
        pipeline_dict (dict): dictionary of agents can be select from different branch
    """

    branches = {}
    name = "branch"

    def __init__(
        self,
        pipeline_dict: Dict[str, GenericAgent],
    ):
        self.pipeline_dict = pipeline_dict
        for pipeline in self.pipeline_dict.values():
            assert isinstance(pipeline, AgentPipeline)
        # the default branch model is the first one
        self.default_branch_name = list(self.pipeline_dict.keys())[0]
        self.states = self.build_states()
        # Don't check the type for Now

    @property
    def source_type(self) -> Optional[str]:
        source_type = list(
            set(pipeline.source_type for pipeline in self.pipeline_dict.values())
        )
        assert len(source_type) == 1, "source type should be the same for all branches"
        return source_type[0]

    @property
    def target_type(self) -> Optional[str]:
        target_type = list(
            set(pipeline.target_type for pipeline in self.pipeline_dict.values())
        )
        assert len(target_type) == 1, "target type should be the same for all branches"
        return target_type[0]

    def push(
        self,
        segment: Segment,
        states: BranchedAgentPipelineStates | None = None,
        upstream_states: List[AgentStates | None] | None = None,
    ) -> None:
        is_stateless = True
        if states is None:
            states = self.states
            is_stateless = False

        states.update_config(segment.config)
        branch_name = self.get_branch_from_states(states)
        branch_states = states.states_dict[branch_name]

        return super().push(
            segment,
            branch_states if states == is_stateless else None,
            upstream_states,
            module_list=self.pipeline_dict[branch_name].module_list,
        )

    def pop(self, states: BranchedAgentPipelineStates | None = None):
        is_stateless = True
        if states is None:
            states = self.states
            is_stateless = False

        branch_name = self.get_branch_from_states(states)
        branch_states = states.states_dict[branch_name]

        return super().pop(
            branch_states if states == is_stateless else None,
            module_list=self.pipeline_dict[branch_name].module_list,
        )

    def build_states(self) -> BranchedAgentPipelineStates:
        return BranchedAgentPipelineStates(
            {
                key: pipeline.build_states()
                for key, pipeline in self.pipeline_dict.items()
            }
        )

    def reset(self) -> None:
        for agent in self.pipeline_dict.values():
            agent.reset()

    def get_branch_from_states(self, states):
        if states is None:
            # stateful agent
            states = self.states

        branch_name = states.config.get(self.name, self.default_branch_name)
        assert branch_name in self.pipeline_dict
        return branch_name

    @classmethod
    def from_args(cls, arg: Namespace):
        pipeline_dict = {}
        for branch_name, pipeline_class_or_list in cls.branches.items():
            if isinstance(pipeline_class_or_list, list):
                pipeline_dict[branch_name] = AgentPipeline.from_pipeline_args(
                    pipeline_class_or_list, arg
                )
            elif isinstance(pipeline_class_or_list, AgentPipeline):
                pipeline_dict[branch_name] = pipeline_class_or_list.from_args(arg)
            else:
                raise NotImplementedError

        return cls(pipeline_dict)

    @classmethod
    def add_args(cls, parser) -> None:
        for pipeline_class_or_list in cls.branches.values():
            if isinstance(pipeline_class_or_list, list):
                for agent in pipeline_class_or_list:
                    agent.add_args(parser)
            else:
                pipeline_class_or_list.add_args(parser)

    def __repr__(self) -> str:
        # TODO, indent here is not correct
        string_list = []
        for branch_name, pipeline in self.pipeline_dict.items():
            string_list.append(f"{branch_name}:\n\t\t{pipeline}")
        string = ",\n".join(
            [
                f"\t{branch_name}:{pipeline}"
                for branch_name, pipeline in self.pipeline_dict.items()
            ]
        )
        return f"{self.__class__.__name__}(\n{string}\n)"

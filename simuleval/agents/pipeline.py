# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from typing import List, Optional, Dict, Set, Type, Union
from simuleval.data.segments import Segment
from .agent import GenericAgent, AgentStates


class AgentPipeline(GenericAgent):
    """A pipeline of agents

    Attributes:
        pipeline (list): a list of agent classes.

    """

    pipeline: List = []

    def __init__(self, module_list: List[GenericAgent]) -> None:
        self.module_list = module_list
        self.check_pipeline_types()

    def check_pipeline_types(self):
        if len(self.pipeline) > 1:
            for i in range(1, len(self.pipeline)):
                if (
                    self.module_list[i].source_type
                    != self.module_list[i - 1].target_type
                ):
                    raise RuntimeError(
                        f"{self.module_list[i]}.source_type({self.module_list[i].source_type}) != {self.pipeline[i-1]}.target_type({self.pipeline[i - 1].target_type}"  # noqa F401
                    )

    @property
    def source_type(self) -> Optional[str]:
        return self.module_list[0].source_type

    @property
    def target_type(self) -> Optional[str]:
        return self.module_list[-1].target_type

    def reset(self) -> None:
        for module in self.module_list:
            module.reset()

    def build_states(self) -> List[AgentStates]:
        return [module.build_states() for module in self.module_list]

    def push(
        self,
        segment: Segment,
        states: Optional[List[Optional[AgentStates]]] = None,
        upstream_states: Optional[List[Optional[AgentStates]]] = None,
    ) -> None:
        if states is None:
            # stateful agent
            states = [None for _ in self.module_list]
            states_list = [module.states for module in self.module_list]
        else:
            # stateless agent
            assert len(states) == len(self.module_list)
            states_list = states

        if upstream_states is None:
            upstream_states = []

        for index, module in enumerate(self.module_list[:-1]):
            config = segment.config
            segment = module.pushpop(
                segment,
                states[index],
                upstream_states=upstream_states + states_list[:index],
            )
            segment.config = config
        self.module_list[-1].push(
            segment,
            states[-1],
            upstream_states=upstream_states + states_list[: len(self.module_list[:-1])],
        )

    def pop(self, states: Optional[List[Optional[AgentStates]]] = None) -> Segment:
        if states is None:
            last_states = None
        else:
            assert len(states) == len(self.module_list)
            last_states = states[-1]

        return self.module_list[-1].pop(last_states)

    @classmethod
    def add_args(cls, parser) -> None:
        for module_class in cls.pipeline:
            module_class.add_args(parser)

    @classmethod
    def from_args(cls, args):
        assert len(cls.pipeline) > 0
        return cls([module_class.from_args(args) for module_class in cls.pipeline])

    def __repr__(self) -> str:
        pipline_str = "\n\t".join(
            "\t".join(str(module).splitlines(True)) for module in self.module_list
        )
        return f"{self.__class__.__name__}(\n\t{pipline_str}\n)"

    def __str__(self) -> str:
        return self.__repr__()


GenericAgentClass = Type[GenericAgent]


class TreeAgentPipeline(AgentPipeline):
    """
    A pipeline which passes intermediate outputs in a directed acyclic graph
    Note: the target_type will be a "_"-concatenated list of types. If the
        argument --output-index is specified, `pop` will return the result
        corresponding to this index.

    Arguments:
        module_dict (dict): a dict mapping instantiated agent to downstream class(es)
            or downstream instance(s). If the downtream objects are classes,
            they will be replaced with the corresponding agent instance from the
            module_dict's keys using `get_instance_from_class`
        args: optionally, args.output_index is used to set the output index

    Attributes:
        pipeline (dict): a dict mapping agent class to the downstream class(es).
            pipeline is only used to initialize the class using `from_args`

    Examples:
        ```
        pipeline = {
            ClassOne: [ClassTwo],
            ClassTwo: [ClassThree],
            ClassThree: []
        }
        ```
        In this case, ClassOne is the root input node (no parent nodes)
            and ClassThree is the leaf output node (no child nodes)
        ```
        pipeline = {
            ClassOne: [ClassTwo, ClassThree],
            ClassTwo: [],
            ClassThree: [],
        }
        ```
        In this case, ClassOne is the input node,
            and ClassTwo, ClassThree are output nodes.

        Note: currently does not support multiple inputs to a node or
            multiple root nodes

        ```
        class InstantiatedTreeAgentPipeline(TreeAgentPipeline):
            pipeline = {ClassOne, ClassTwo, ClassThree}
            def __init__(self, args) -> None:
                one = ClassOne()
                two = ClassTwo()
                three = ClassThree()

                module_dict = {
                    one: [two, three],
                    two: [],
                    three: [],
                }

                super().__init__(module_dict, args)

            @classmethod
            def from_args(cls, args):
                return cls(args)
        ```
        In this example, the downstream children in module_dict are
        already instantiated.
    """

    pipeline: Union[
        Dict[GenericAgentClass, List[GenericAgentClass]], List[GenericAgentClass]
    ] = {}

    def __init__(
        self,
        module_dict: Dict[GenericAgent, List[Union[GenericAgent, GenericAgentClass]]],
        args,
    ) -> None:
        self.check_pipeline_types(module_dict)

        self.output_index = args.output_index
        if self.output_index is not None:
            assert len(self.target_modules) > args.output_index

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        super().add_args(parser)
        parser.add_argument(
            "--output-index",
            type=int,
            default=None,
            help="If specified, `pop` will only output the result at this index.",
        )

    @classmethod
    def get_instance_from_class(cls, klass, module_dict):
        ins_list = [ins for ins in module_dict.keys() if type(ins) == klass]
        assert len(ins_list) == 1, f"Instances of {klass}: {ins_list}"
        return ins_list[0]

    def check_pipeline_types(self, module_dict):
        for parent, children in module_dict.items():
            for child in children:
                if isinstance(child, type):
                    child = self.get_instance_from_class(child, module_dict)
                if child.source_type != parent.target_type:
                    raise RuntimeError(
                        f"{child}.source_type({child.source_type}) != {parent}.target_type({parent.target_type}"  # noqa F401
                    )
        self.set_pipeline_tree(module_dict)
        self.check_cycle(set(), self.source_module)

    def set_pipeline_tree(self, module_dict):
        root_instance = list(module_dict.keys())
        leaf_instances = []
        output_dict = {}
        for parent, children in module_dict.items():
            output_dict[parent] = []
            if len(children) == 0:
                leaf_instances.append(parent)
                continue
            for child in children:
                if isinstance(child, type):
                    child = self.get_instance_from_class(child, module_dict)
                if child in root_instance:
                    root_instance.remove(child)
                output_dict[parent].append(child)

        assert len(root_instance) == 1
        assert len(leaf_instances) > 0
        self.source_module = root_instance[0]
        self.target_modules = leaf_instances
        self.module_dict = output_dict

    def check_cycle(self, visited, ins):
        if ins in visited:
            raise ValueError(f"cycle in graph: {ins}")
        for child in self.module_dict[ins]:
            visited.add(ins)
            self.check_cycle(visited, child)

    @property
    def source_type(self) -> Optional[str]:
        return self.source_module.source_type

    @property
    def target_type(self) -> Optional[List[str]]:
        if self.output_index is not None:
            return self.target_modules[self.output_index].target_type
        return "_".join([target.target_type for target in self.target_modules])

    @property
    def module_list(self) -> List[GenericAgent]:
        return self.module_dict.keys()

    def build_states(self) -> Dict[GenericAgent, AgentStates]:
        return {module: module.build_states() for module in self.module_dict.keys()}

    def push_impl(
        self,
        module: GenericAgent,
        segment: Segment,
        states: Optional[Dict[GenericAgent, AgentStates]],
        upstream_states: Dict[int, AgentStates],
    ):
        # DFS over the tree
        children = self.module_dict[module]
        if len(children) == 0:  # leaf node
            module.push(segment, states[module], upstream_states)
            upstream_states[len(upstream_states)] = states[module]
            return []

        # start = time.time()
        config = segment.config
        segment = module.pushpop(segment, states[module], upstream_states)
        segment.config = config
        # logger.warning(f"{type(module).__name__}, {round(time.time() - start, 3)}")
        assert len(upstream_states) not in upstream_states
        upstream_states[len(upstream_states)] = (
            states[module] if states[module] is not None else module.states
        )

        for child in children:
            self.push_impl(child, segment, states, upstream_states)

    def pushpop(
        self,
        segment: Segment,
        states: Optional[Dict[GenericAgent, AgentStates]] = None,
        upstream_states: Optional[List[AgentStates]] = None,
    ) -> Segment:
        self.push(segment, states, upstream_states)
        return self.pop(states)

    def push(
        self,
        segment: Segment,
        states: Optional[Dict[GenericAgent, AgentStates]] = None,
        upstream_states: Optional[List[AgentStates]] = None,
    ) -> None:
        if states is None:
            states = {module: None for module in self.module_dict}
        else:
            assert len(states) == len(self.module_dict)

        if upstream_states is None:
            upstream_states = {}

        self.push_impl(self.source_module, segment, states, upstream_states)

    def pop(
        self, states: Optional[Dict[GenericAgent, AgentStates]] = None
    ) -> List[Segment]:
        outputs = []
        for module in self.target_modules:
            if states is None:
                last_states = None
            else:
                assert len(states) == len(self.module_dict)
                last_states = states[module]

            outputs.append(module.pop(last_states))
        if self.output_index is not None:
            return outputs[self.output_index]
        return outputs

    @classmethod
    def from_args(cls, args):
        assert len(cls.pipeline) > 0
        return cls(
            {
                module_class.from_args(args): children
                for module_class, children in cls.pipeline.items()
            },
            args,
        )

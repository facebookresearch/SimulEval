# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import os
import re
import sys
import logging
import importlib
from simuleval.agents import GenericAgent

logger = logging.getLogger("simuleval.util.agent_builder")


def new_class_names(user_file):
    new_class_names = []

    pattern = re.compile("^class\\s+([^(]+)(\\(.*\\))*:")
    with open(user_file) as f:
        for line in f:
            result = pattern.search(line)
            if result is not None:
                new_class_names.append(result.group(1))

    return new_class_names

def import_file(file_path):
    spec = importlib.util.spec_from_file_location("agents", file_path)
    agent_modules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_modules)


def find_agent_cls(args):
    agent_file = args.agent
    if args.agent is None:
        agent_file = os.environ.get("SIMULEVAL_AGENT", None)
        if agent_file is None:
            logger.error(
                "You have to specify an agent file either by --agent for set environmental variable SIMULEVAL_AGENT"
            )
            sys.exit(1)

    agent_file = os.path.abspath(agent_file)

    new_class_names_in_file = new_class_names(agent_file)

    agent_name = None


    spec = importlib.util.spec_from_file_location("agents", agent_file)
    agent_modules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_modules)

    agent_cls = []

    for cls_name in new_class_names_in_file:
        kls = getattr(agent_modules, cls_name)

        if isinstance(kls, type) and issubclass(kls, GenericGenericAgent):
            agent_cls.append(kls)

    if len(agent_cls) == 0:
        logger.error(f"No 'GenericAgent' class found in {agent_file}\n")
        sys.exit(1)

    if len(agent_cls) > 1:
        if agent_name is None:
            logger.error(
                f"Multiple 'GenericAgent' classes found in {agent_file}. Please select one by {agent_file}:GenericAgentClassName.\n"
            )
            sys.exit(1)
        agent_cls = getattr(agent_modules, agent_name, None)
        if agent_cls is None:
            logger.error(f"{agent_name} not found in {agent_file}.\n")
            sys.exit(1)
    else:
        agent_cls = agent_cls[0]
        if agent_name is not None and agent_name != agent_cls.__name__:
            logger.error(
                f"Failed to find {agent_name} in {agent_file}. Do you mean {agent_cls.__name__}?\n"
            )
            sys.exit(1)

        agent_name = agent_cls.__name__

    return agent_name, agent_cls


def infer_data_types_from_agent(args: Namespace, agent: GenericAgent):
    for side in ["source", "target"]:
        if getattr(args, side + "_type") is None:
            setattr(args, side + "_type", getattr(agent, side + "_type"))
        else:
            if getattr(args, side + "_type") != getattr(agent, side + "_type"):
                logger.error(
                    f"Data type mismatch '--{side}-type {getattr(args, side + '_type')}' "
                    f"vs {agent.__name__}.{side}_type = {getattr(agent, side + '_type')}"
                )
                sys.exit(1)

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import sys
import logging
import importlib
from simuleval.agents import Agent

logger = logging.getLogger('simuleval.util.agent_builder')


def new_class_names(user_file):
    new_class_names = []

    pattern = re.compile("^class\\s+(\\w+)[\\(\\w+\\)]*:")
    with open(user_file) as f:
        for line in f:
            result = pattern.search(line)
            if result is not None:
                new_class_names.append(result.group(1))

    return new_class_names


def find_agent_cls(args):
    agent_file = args.agent
    if args.agent is None:
        agent_file = os.environ.get("SIMULEVAL_AGENT", None)
        if agent_file is None:
            logger.error(
                "You have to specify an agent file either by --agent for set environmental variable SIMULEVAL_AGENT")
            sys.exit(1)

    agent_file = os.path.abspath(agent_file)

    new_class_names_in_file = new_class_names(agent_file)

    agent_name = None
    if ":" in agent_file:
        agent_name = agent_file.split(":")[1:]
        agent_file = os.path.abspath(agent_file.split(":")[0])
        if len(agent_name) > 1:
            logger.error(
                f"Only one agent name at one time, {len(agent_name)} are provided. {' '.join(agent_name)}"
            )
        sys.exit(1)
        agent_name = agent_name[0]

        if agent_name not in new_class_names_in_file:
            logger.error(
                f"No definition found for class {agent_name} in {agent_file}"
            )
            sys.exit(1)

        new_class_names_in_file = [agent_name]

    spec = importlib.util.spec_from_file_location("agents", agent_file)
    agent_modules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_modules)

    agent_cls = []

    for cls_name in new_class_names_in_file:
        kls = getattr(agent_modules, cls_name)

        if isinstance(kls, type) and issubclass(kls, Agent):
            agent_cls.append(kls)

    if len(agent_cls) == 0:
        logger.error(f"No 'Agent' class found in {agent_file}\n")
        sys.exit(1)

    if len(agent_cls) > 1:
        if agent_name is None:
            logger.error(
                f"Multiple 'Agent' classes found in {agent_file}. Please select one by {agent_file}:AgentClassName.\n")
            sys.exit(1)
        agent_cls = getattr(agent_modules, agent_name, None)
        if agent_cls is None:
            logger.error(f"{agent_name} not found in {agent_file}.\n")
            sys.exit(1)
    else:
        agent_cls = agent_cls[0]
        if agent_name is not None and agent_name != agent_cls.__name__:
            logger.error(
                f"Failed to find {agent_name} in {agent_file}. Do you mean {agent_cls.__name__}?\n")
            sys.exit(1)

        agent_name = agent_cls.__name__

    return agent_name, agent_cls


def check_data_type(args, agent_cls):
    if args.data_type is None:
        args.data_type = agent_cls.data_type
    else:
        if args.data_type != agent_cls.data_type:
            logger.error(
                f"Data type mismatch '--data-type {args.data_type}', '{agent_cls.__name__}.data_type: {agent_cls.data_type}'")
            sys.exit(1)

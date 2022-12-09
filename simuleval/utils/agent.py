# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib


def import_file(file_path):
    spec = importlib.util.spec_from_file_location("agents", file_path)
    agent_modules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_modules)

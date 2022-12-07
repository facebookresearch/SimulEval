# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

DEFAULT_EOS = "</s>"

SUPPORTED_MEDIUM = ["text", "speech"]
SUPPORTED_SOURCE_MEDIUM = ["text", "speech"]
SUPPORTED_TARGET_MEDIUM = ["text", "speech"]

EVALUATION_SYSTEM_LIST = []


def entrypoint(klass):
    EVALUATION_SYSTEM_LIST.append(klass)
    return klass

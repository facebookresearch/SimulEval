# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

EVALUATION_SYSTEM_LIST = []


def entrypoint(klass):
    EVALUATION_SYSTEM_LIST.append(klass)
    return klass

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

DEFAULT_EOS = '</s>'
DEFAULT_SERVER_PATH = os.path.join(os.getenv("HOME"), ".simuleval")
DEFAULT_HOSTNAME = 'localhost'
DEFAULT_PORT = 12321

READ_ACTION = "read_action"
WRITE_ACTION = "write_action"

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
# import tensorflow # need load tensorflow first

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


if os.path.basename(os.path.abspath(os.getcwd())) == "tools":
    os.chdir("..")

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, os.path.join(root_dir))
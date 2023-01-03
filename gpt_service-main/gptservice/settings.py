"""
-- Created by: Ashok Kumar Pant
-- Created on: 9/22/21
"""
import os
from traceback import print_exc

from gptservice.utils.config import Config


def load_env_variables(env_filename=".env"):
    try:
        if os.path.exists(env_filename):
            with open(env_filename, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    kv = line.split("=")
                    if len(kv) == 1:
                        k = kv[0]
                        v = ""
                    else:
                        k = kv[0]
                        v = kv[1]
                    os.environ.setdefault(k.strip(), v.strip())
    except Exception as e:
        print_exc(e)
        pass


load_env_variables()

GPTJ_MODEL_NAME = os.environ.get("GPT_MODEL_NAME", "step_383500_slim.tar.zstd")
MODEL_BASE_DIR = os.environ.get("MODEL_BASE_DIR",
                                Config.get_instance().get("default", "model.base.dir", fallback="./data"))
TPU_NAME = os.environ.get("TPU_NAME", Config.get_instance().get("default", "tpu.name", fallback=None))

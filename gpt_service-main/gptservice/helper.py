"""
-- Created by: Ashok Kumar Pant
-- Created on: 9/28/21
"""
import os
import time
from traceback import print_exc

from gptservice import settings


def run_subprocess(cmd):
    import subprocess
    process = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)
    return process


def uncompress_zstd_to_folder(input_file, output_path):
    try:
        # apt install zstd
        cmd = "mkdir -p {} && tar -I zstd -xf {} --directory {} --strip-components=1".format(output_path,
                                                                                             input_file,
                                                                                             output_path)
        run_subprocess(cmd)
        return True
    except Exception as e:
        print_exc(e)
        return False


def download_model(model_url, output_path):
    try:
        # apt install wget
        cmd = "wget -O {} -c {}".format(output_path, model_url)
        run_subprocess(cmd)
        return True
    except Exception as e:
        print_exc(e)
        return False


def download_or_get_gptj_6b_model():
    model_name = settings.GPTJ_MODEL_NAME
    resource_base_dir = settings.MODEL_BASE_DIR
    model_folder_name = model_name.split(".")[0]
    model_path = os.path.join(resource_base_dir, model_folder_name)
    model_compressed_path = os.path.join(resource_base_dir, model_name)
    if os.path.exists(model_path):
        return model_path
    elif not os.path.exists(model_compressed_path):
        print("Downloading the model: {}".format(model_name))
        model_url = "{}/{}".format("https://the-eye.eu/public/AI/GPT-J-6B", model_name)
        success = download_model(model_url, model_compressed_path)
        if not success:
            return None
    print("Unzipping the model: {}".format(model_compressed_path))
    success = uncompress_zstd_to_folder(model_compressed_path, model_path)
    if success:
        return model_path
    else:
        return None


def timer(start_time=None):
    if not start_time:
        return time.time()
    return time.time() - start_time


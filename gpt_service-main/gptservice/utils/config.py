"""
-- Created by: Ashok Kumar Pant
-- Email: ashokpant@treeleaf.ai
-- Created on: 2/7/20
"""
import argparse
import configparser
import logging
import os
from configparser import ConfigParser


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ConfigError(Exception):
    def __init__(self, *args, **kwargs):
        super(ConfigError, self).__init__(*args, kwargs)


class EnvInterpolation(configparser.BasicInterpolation):
    def before_get(self, parser, section, option, value, defaults):
        if "os.environ" in value:
            return eval(value)
        else:
            return value


class Config(metaclass=Singleton):
    _conf = None

    @classmethod
    def build(cls, filename=None):
        if filename is None:
            parser = argparse.ArgumentParser(__name__)
            parser.add_argument('--config', type=str,
                                help='Config file path', default=None)
            args, unknown = parser.parse_known_args()
            if args.config is None:
                filename = os.environ.get("SERVICE_CONFIG_FILE", "config.ini")
            else:
                filename = args.config

        logging.debug("Config file: {}".format(os.path.abspath(filename)))
        if not os.path.exists(filename):
            raise ConfigError("Config file {} is not found.".format(filename))
        try:
            Config._conf = ConfigParser(interpolation=EnvInterpolation())
            Config._conf.read(filenames=filename)
            return Config._conf
        except Exception as e:
            raise ConfigError(e)

    @staticmethod
    def get_instance():
        if Config._conf is None:
            Config.build()
        return Config._conf

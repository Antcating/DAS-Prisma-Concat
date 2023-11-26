import configparser
from os.path import isdir

config_dict = configparser.ConfigParser()
config_dict.read("config.ini", encoding="UTF-8")

# CONSTANTS
CHUNK_SIZE = int(config_dict["CONSTANTS"]["CHUNK_SIZE"])
UNIT_SIZE = int(config_dict["CONSTANTS"]["UNIT_SIZE"])
# PACKET CHARACTERISTICS
PRR = int(config_dict["CONSTANTS"]["PRR"])
DX = float(config_dict["CONSTANTS"]["DX"])

# PATHs to files and save
INPUT_PATH = config_dict["PATH"]["INPUT_PATH"]

if not isdir(INPUT_PATH):
    raise Exception("PATH is not accessible!")

OUTPUT_PATH = config_dict["PATH"]["OUTPUT_PATH"]

if not isdir(OUTPUT_PATH):
    raise Exception("PATH is not accessible!")
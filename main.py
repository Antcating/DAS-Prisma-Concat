#!/usr/bin/env python

from Concatenator import PrismaConcatenator
from log.main_logger import logger as log
from config import INPUT_PATH, OUTPUT_PATH

concatenator = PrismaConcatenator(INPUT_PATH, OUTPUT_PATH)
try:
    concatenator.run()
except Exception as e:
    log.error(e)
    # Print traceback to console
    import traceback
    traceback.print_exc()
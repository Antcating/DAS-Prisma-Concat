#!/usr/bin/env python

from Concatenator import PrismaConcatenator

from config import INPUT_PATH, OUTPUT_PATH

concatenator = PrismaConcatenator(INPUT_PATH, OUTPUT_PATH)
concatenator.run()
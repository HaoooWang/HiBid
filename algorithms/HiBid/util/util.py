#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys


def set_logger():
    logger = logging.getLogger("tensorflow")

    if len(logger.handlers) == 1:
        logger.handlers = []
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s - [%(filename)s:%(lineno)d] - %(name)s - %(levelname)s - %(message)s")
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        fh = logging.FileHandler('tensorflow.log')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger

import logging

logger = logging.getLogger("LOG")
logger.setLevel(logging.DEBUG)
logger.propagate = False
streamhandler = logging.StreamHandler()
streamhandler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s\t%(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)

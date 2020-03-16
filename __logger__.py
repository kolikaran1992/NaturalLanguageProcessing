import logging

LOGGER_NAME = 'NLP'

logger = logging.getLogger(LOGGER_NAME)

formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

import os
import logging
from json import JSONDecodeError
from logging.handlers import RotatingFileHandler
import json
from typing import Tuple, Union


def extract_score_and_mutation(response: str) -> Tuple[float, Union[str, JSONDecodeError]]:
    """
    Extracts the score from the response.
    :param response: agent response
    :return: score
    """
    try:
        response_dict = json.loads(response.replace('\n', ' ').strip().lower())
        return float(response_dict['response']), response_dict['modification']
    except JSONDecodeError as e:
        idx = response.rfind("Result:")
        return float(response[idx + len("Result:"):].split(' ')[0].strip()), e


def get_logger_file_handler(log_file: str):
    """
    Returns a file handler for the logger.
    :param log_file: the log file
    :return: the file handler
    """
    current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_file_path = os.path.join(current_path, 'logs', log_file)
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = RotatingFileHandler(log_file_path, maxBytes=5 * 1024 * 1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)

    return file_handler

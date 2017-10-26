from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging

from rasa_core.agent import Agent
from rasa_core.channels.file import FileInputChannel
from rasa_core.interpreter import RegexInterpreter, RasaNLUInterpreter
from rasa_core.policies.memoization import MemoizationPolicy

from run import nlu_model_path
from restaurant_example import RestaurantPolicy

logger = logging.getLogger(__name__)
DIR_PATH = os.path.abspath((os.path.dirname(__file__)))

def run_babi_online(max_messages=10):
    training_data = os.path.join(DIR_PATH, 'data/babi_task5_dev_rasa_even_smaller.md')
    logger.info("Starting to train policy")
    agent = Agent(os.path.join(DIR_PATH, 'restaurant_domain.yml'),
                  policies=[MemoizationPolicy(), RestaurantPolicy()],
                  interpreter=RegexInterpreter())

    input_c = FileInputChannel(training_data,
                               message_line_pattern='^\s*\*\s(.*)$',
                               max_messages=max_messages)
    agent.train_online(training_data,
                       input_channel=input_c,
                       epochs=10)

    agent.interpreter = RasaNLUInterpreter(nlu_model_path)
    return agent


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    run_babi_online()

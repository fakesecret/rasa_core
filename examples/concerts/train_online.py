from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import logging

from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.interpreter import RegexInterpreter
from rasa_core.policies.memoization import MemoizationPolicy

logger = logging.getLogger(__name__)

DIR_PATH = os.path.abspath((os.path.dirname(__file__)))
sys.path.append(os.path.dirname(DIR_PATH))
# project modules
from concerts.policy import ConcertPolicy

def run_concertbot_online(input_channel, interpreter):
    training_data_file = os.path.join(DIR_PATH,'data/stories.md')

    agent = Agent(os.path.join(DIR_PATH,'concert_domain.yml'),
                  policies=[MemoizationPolicy(), ConcertPolicy()],
                  interpreter=interpreter)

    agent.train_online(training_data_file,
                       input_channel=input_channel,
                       max_history=2,
                       batch_size=50,
                       epochs=200,
                       max_training_samples=300)

    return agent


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    run_concertbot_online(ConsoleInputChannel(), RegexInterpreter())

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import sys
import os

from rasa_core.agent import Agent
from rasa_core.interpreter import RegexInterpreter
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy

# project modules
from fake_user import FakeUserInputChannel

logger = logging.getLogger(__name__)
DIR_PATH = os.path.abspath((os.path.dirname(__file__)))

def run_fake_user(max_training_samples=10, serve_forever=True):
    training_data = os.path.join(DIR_PATH,'data/babi_task5_fu_rasa_fewer_actions.md')

    logger.info("Starting to train policy")

    agent = Agent(os.path.join(DIR_PATH, "restaurant_domain.yml"),
                  policies=[MemoizationPolicy(), KerasPolicy()],
                  interpreter=RegexInterpreter())

    # Instead of generating the response messages ourselves, the fake user will
    # generate input messages based on the dialogue state
    input_channel = FakeUserInputChannel(agent.tracker_store)

    agent.train_online(training_data,
                       input_channel=input_channel,
                       epochs=1,
                       max_training_samples=max_training_samples)
    return agent


if __name__ == '__main__':
    logging.basicConfig(level="INFO")

    if len(sys.argv) < 2 or sys.argv[1] == 'scratch':
        max_training_samples = 10
    elif sys.argv[1] == 'pretrained':
        max_training_samples = -1
    else:
        raise Exception("Choose from pretrained or training from scratch")

    run_fake_user(max_training_samples)

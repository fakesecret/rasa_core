from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging

from rasa_core.agent import Agent
from rasa_core.policies.memoization import MemoizationPolicy

DIR_PATH = os.path.abspath((os.path.dirname(__file__)))
# sys.path.append(DIR_PATH)
# project modules
from policy import ConcertPolicy

if __name__ == '__main__':
    logging.basicConfig(level="INFO")

    training_data_file = os.path.join(DIR_PATH,'data/stories.md')
    model_path = os.path.join(DIR_PATH,'models/policy/init')

    agent = Agent(os.path.join(DIR_PATH,'concert_domain.yml'),
                  policies=[MemoizationPolicy(), ConcertPolicy()])

    agent.train(
            training_data_file,
            augmentation_factor=50,
            max_history=2,
            epochs=500,
            batch_size=10,
            validation_split=0.2
    )

    agent.persist(model_path)

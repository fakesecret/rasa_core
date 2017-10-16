from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging

from examples.restaurant_example import RestaurantPolicy
from rasa_core.agent import Agent
from rasa_core.policies.memoization import MemoizationPolicy

DIR_PATH = os.path.abspath((os.path.dirname(__file__)))

def train_babi_dm():
    training_data_file = os.path.join(DIR_PATH, 'data/babi_task5_trn_rasa_with_slots.md')
    model_path = os.path.join(DIR_PATH, 'models/policy/current')

    agent = Agent(os.path.join(DIR_PATH,  'restaurant_domain.yml'),
                  policies=[MemoizationPolicy(), RestaurantPolicy()])

    agent.train(
            training_data_file,
            max_history=3,
            epochs=100,
            batch_size=50,
            augmentation_factor=50,
            validation_split=0.2
    )

    agent.persist(model_path)


if __name__ == '__main__':
    logging.basicConfig(level="DEBUG")
    train_babi_dm()

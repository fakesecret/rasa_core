from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging

import six

from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.interpreter import RasaNLUInterpreter

DIR_PATH = os.path.abspath((os.path.dirname(__file__)))

if six.PY2:
    nlu_model_path = os.path.join(DIR_PATH, 'models/nlu/current_py2')
else:
    nlu_model_path = os.path.join(DIR_PATH, 'models/nlu/current_py3')


def run_babi(serve_forever=True):
    agent = Agent.load(os.path.join(DIR_PATH, 'models/policy/current'),
                       interpreter=RasaNLUInterpreter(nlu_model_path))

    if serve_forever:
        agent.handle_channel(ConsoleInputChannel())
    return agent


if __name__ == '__main__':
    logging.basicConfig(level="DEBUG")
    run_babi()

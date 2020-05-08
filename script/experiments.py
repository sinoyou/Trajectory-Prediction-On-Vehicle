import sys
import os

from script.tools import Recorder
from model.runner import Trainer

class ArgsMaker:
    """
    Class for making args
    """

    def __init__(self):
        pass

    def __len__(self):
        pass

    def __iter__(self):
        pass

    def add_arg_rule(self, args_name, alternatives):
        pass

    def get_args_recursively(self, level):
        return None


class ArgsBlocker:
    """
    Maintain rules of blocking invalid arguments combination.
    """

    def __init__(self):
        self.black_list = list()

    def add_block_rule(self, rule):
        pass

    def is_blocked(self, attr):
        self.black_list = None
        return False


class TaskRunner:
    """
    A runner to train/validate model. Just like running a train script.
    """

    def __init__(self, prefix, task_attr):
        """
        :param prefix: An identifier for adding to the front of args.board_name.
                    Better used to identify different experiment batches.
        :param task_attr: train arguments for passing into model runner.
        """
        pass

    def run(self):
        pass


if __name__ == '__main__':
    pass

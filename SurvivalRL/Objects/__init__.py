from os.path import dirname
from sys import path

path.insert(0, dirname(__file__))

from BaseObjects import Rectangle, Circle

from .predator import Predator
from .herbivore import Herbivore
from .plant import Plant
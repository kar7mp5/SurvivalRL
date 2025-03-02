from os.path import dirname
from sys import path

path.insert(0, dirname(__file__))

from .config import Config
from .game_object import GameObject
from Objects import *
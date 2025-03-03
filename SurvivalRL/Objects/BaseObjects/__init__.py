from os.path import dirname
from sys import path

path.insert(0, dirname(__file__))

from .position import Position
from .base_object import BaseObject
from .circle import Circle
from .rectangle import Rectangle
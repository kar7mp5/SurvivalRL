from os.path import dirname
from sys import path

path.insert(0, dirname(__file__))

from .obj import Obj
from .circle import Circle
from .rectangle import Rectangle
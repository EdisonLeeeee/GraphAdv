from graphgallery import config

set_epsilon = config.set_epsilon
set_floatx = config.set_floatx
set_intx = config.set_intx
epsilon = config.epsilon
floatx = config.floatx
intx = config.intx

from graphadv.utils.type_check import *
from graphadv.utils.data_utils import flip_adj, flip_x

from graphadv.base import BaseModel
from graphadv import attack
from graphadv import defense
from graphadv import utils



__version__ = '0.1.0'

__all__ = ['graphadv', 'attack', 'defense', 'utils', '__version__']

import itertools
import json
import pathlib

import numpy as np

from mantis.acquisition import BaseSettings
from mantis.acquisition import microscope_operations, autoexposure
from mantis.acquisition.acq_engine import MantisAcquisition


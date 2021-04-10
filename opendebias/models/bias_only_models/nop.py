import json
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from overrides import overrides
import torch.nn.functional as F

from opendebias.models.base_model import BaseModel

logger = logging.getLogger(__name__)


@BaseModel.register("nop")
class NonOpBiasOnly(BaseModel):
    def forward(self):
        return {}
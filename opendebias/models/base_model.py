from typing import Any, Dict, List, Optional, Tuple

import torch
from allennlp.common import Registrable
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Metric
from overrides import overrides


class BaseModel(Model, Registrable):
    def __init__(self,
                vocab: Vocabulary,
                regularizer: RegularizerApplicator = None,
                serialization_dir: Optional[str] = None,
                forward_keys: Optional[List[str]]) -> None:
        suepr().__init__(vocab, regularizer, serialization_dir)
        self._forward_keys = [] if forward_keys is None else forward_keys


    def get_forward_keys() -> List[str]:
        return self._forward_keys


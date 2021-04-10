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


@BaseModel.register("lookup-biasonly")
class LookUp(BaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        bias_only_model_prediction_file: List[Dict[str, str]],
        label_namespace: str = "labels",
        **kwargs,
    ) -> None:
        super().__init__(vocab, forward_keys=["metadata.dataset_name", "index"], **kwargs)
        assert len(bias_only_model_prediction_file) > 0
        self._label_namespace = label_namespace
        self._bias_only_model_logits = self._read_prediction_file(bias_only_model_prediction_file)

    def _read_prediction_file(self, args: List[Dict[str, str]]):
        bias_logits: Dict[str, np.ndarray] = {}
        for arg in args:
            if len(arg) == 0:
                continue
            if not os.path.exists(arg['file_name']):
                logger.warning('{} is not exists'.format(arg['file_name']))
                bias_logits[arg['dataset_name']] = None
            else:
                logits = self._read(arg['file_name'], arg['dataset_name'])
                bias_logits[arg['dataset_name']] = torch.from_numpy(logits).float()
        return bias_logits

    def _read(self, fn, dataset_name):
        with open(fn, 'r') as f:
            bias = json.load(f)
        bias_logits_list = [None] * len(bias['logits'])
        dim = len(bias['logits'][0])
        index2token = bias['index2token']
        for line_id, v in zip(bias['index'], bias['log_probs']):
            bias_logits = [None for _ in range(dim)]
            for i in range(dim):
                label = index2token[str(i)]
                index = self.vocab.get_token_index(label, self._label_namespace)
                bias_logits[index] = v[i]
            bias_logits_list[int(line_id)] = np.array(bias_logits)
        bias_logits_list = np.array(bias_logits_list)
        return bias_logits_list

    def forward(self, 
                dataset_name: str, 
                index: torch.IntTensor):
        # index: [bs]
        if dataset_name not in self._bias_only_model_logits:
            return {}
    
        if self._bias_only_model_logits[dataset_name] is None:
            raise Exception('Do Not Exist {}'.format(dataset_name))

        if self._bias_only_model_logits[dataset_name].device != index.device:
            self._bias_only_model_logits[dataset_name] = self._bias_only_model_logits[dataset_name].to(index.device)
            
        bias_logits = self._bias_only_model_logits[dataset_name][index]
        bias_probs = F.softmax(bias_logits)
        output_dict = {"probs": bias_probs, "logits": bias_logits}
        return output_dict
    


@BaseModel.register("nop")
class NonOpBiasOnly(BaseModel):
    def forward(self):
        return {}
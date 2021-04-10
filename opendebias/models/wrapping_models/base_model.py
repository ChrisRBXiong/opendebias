from typing import Any, Dict, List, Optional, Tuple

import torch
from allennlp.common import Registrable
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.models.model import Model
from allennlp.modules import (FeedForward, Seq2SeqEncoder, Seq2VecEncoder,
                              TextFieldEmbedder)
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Metric
from opendebias.models.base_model import BaseModel
from opendebias.modules.losses import EBDLoss
from overrides import overrides


class EBDModelBase(Model):
    def __init__(
        self, 
        vocab: Vocabulary,
        bias_only_model: BaseModel,
        main_model: BaseModel,
        ebd_loss: EBDLoss,
        label_namespace: str = "labels",
        namespace: str = "tokens",
        initializer: InitializerApplicator = InitializerApplicator(),
        ensemble_metrics: Optional[Dict[str, Metric]] = None
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._main_model = main_model
        self._bias_only_model = bias_only_model
        self._loss = ebd_loss
        self._ensemble_metrics = [] if ensemble_metrics is None else ensemble_metrics
        self._main_model_input_keys = main_model.get_forward_keys()
        self._bias_only_model_input_keys = bias_only_model.get_forward_keys()
        self._label_namespace = label_namespace
        self._namespace = namespace
        initializer(self)


    def _pack_input(self, keys: List[str], input: Dict[str, Any], main_mdoel_output: Dict[str, Any]) -> Dict[str, Any]:
        input_dict = {}
        for key in keys:
            if key.startswith('metadata.'):
                input_dict[key[9:]] = input['metadata'][0][key[9:]]
            elif key.startswith('main_model.'):
                input_dict[key[11:]] = main_mdoel_output[key[11:]]
            else:
                input_dict[key] = input[key]
        return input_dict

    def interaction(self,
                    **kw_input) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        raise NotImplementedError()
        

    @overrides
    def forward(
        self, 
        **kw_input,
    ) -> Dict[str, torch.Tensor]:

        main_model_output, bias_only_model_output = self.interaction(**kw_input)
        ensemble_output = self.loss(main_model_output, bias_only_model_output, kw_input.get("label", None))

        # main_model_output = self._main_model(**self._pack_input(self._main_model_input_keys, kw_input))
        # bias_only_model_output = self._bias_only_model(**self._pack_input(self._bias_only_model_input_keys, kw_input))
        
        output_dict: Dict[str, Any] = {}
        # save output
        to_save_fields = ["probs", "logits", "loss"]
        for field in main_model_output:
            if field in main_model_output:
                output_dict[f"main_{field}"] = main_model_output[field]
        for field in bias_only_model_output:
            if field in bias_only_model_output:
                output_dict[f"bias_only_{field}"] = bias_only_model_output[field]
        for field in ensemble_output:
            if field in ensemble_output and field != 'loss':
                output_dict[f"ensemble_{field}"] = ensemble_output[field]
        
        if "loss" in ensemble_output:
            output_dict["loss"] = ensemble_output["loss"]
        if "label" in kwargs:
            for metric in self._ensemble_metrics.values():
                metric(ensemble_output["logits"], kwargs["label"])
        
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {key, metric.get_metric(reset) for key, metrc in self._ensemble_metrics.items()}
        return metrics

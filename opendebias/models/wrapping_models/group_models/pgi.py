from enum import unique
from re import S
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.autograd import grad, grad_mode
from allennlp.common import Registrable
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.models import model
from allennlp.models.model import Model
from allennlp.modules import (FeedForward, Seq2SeqEncoder, Seq2VecEncoder,
                              TextFieldEmbedder)
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Metric, Average
from opendebias.modules.losses import EBDLoss
from overrides import overrides
from copy import deepcopy
import numpy as np
from torch import nn
import inspect
from opendebias.models.utils import dynamic_partition
from opendebias.models.wrapping_models.group_models.base import GroupBase
import torch.nn.functional as F

EPS = 1e-12

@Model.register('pgi')
class PGI(GroupBase):
    def __init__(
        self, 
        vocab: Vocabulary,
        model: Model,
        weight_adapt: float = None,
        label_namespace: str = "labels",
        namespace: str = "tokens",
        initializer: InitializerApplicator = InitializerApplicator(),
        metrics: Optional[Dict[str, Metric]] = None,
        reverse: bool = False,
        **kwargs,
    ) -> None:
        self._reverse = reverse
        super().__init__(vocab, model, weight_adapt, label_namespace, namespace, initializer, metrics, **kwargs)

    
    @overrides
    def calculate_penalty(self, all_loss, logits, labels, envs, unique_env_count, hidden=None) -> torch.Tensor:
        probs = F.softmax(logits)
        class_part_probs = dynamic_partition(probs, labels, self._label_count)
        class_part_envs = dynamic_partition(envs, labels, self._label_count)
        penalty = torch.tensor(0.0).to(logits.device)

        for class_probs, class_envs in zip(class_part_probs, class_part_envs):
            env_part_class_probs = dynamic_partition(class_probs, class_envs, unique_env_count)
            assert len(env_part_class_probs) == 2
            if len(env_part_class_probs[0]) == 0 or len(env_part_class_probs[1]) == 0: continue
            env_part_class_probs0, env_part_class_probs1 = env_part_class_probs[0], env_part_class_probs[1]
            env_part_class_probs0 = torch.mean(env_part_class_probs0, dim=0).clamp(EPS, 1-EPS)
            env_part_class_probs1 = torch.mean(env_part_class_probs1, dim=0).clamp(EPS, 1-EPS)
            if self._reverse:
                env_part_class_probs0, env_part_class_probs1 = env_part_class_probs1, env_part_class_probs0
            penalty += torch.mean(env_part_class_probs1 * torch.log(env_part_class_probs1 / env_part_class_probs0))
        return penalty
    
    @overrides
    def calculate_loss(self, all_loss, logits, labels, envs, unique_env_count, hidden=None) -> torch.Tensor:
        env_part_losses = dynamic_partition(all_loss, envs, unique_env_count)
        final_loss = 0.0
        for env_loss in env_part_losses:
            if len(env_loss) == 0: continue
            final_loss += env_loss.mean()
        return final_loss / unique_env_count
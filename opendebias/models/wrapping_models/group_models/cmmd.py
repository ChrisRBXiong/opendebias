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


def kernel(x, y):
    xnorm = torch.sum(x**2, 1, keepdims=True)
    ynorm = torch.sum(y**2, 1, keepdims=True)
    distance = xnorm + torch.transpose(ynorm, 0, 1) - 2 * torch.matmul(x, torch.transpose(y, 0, 1))
    kernel_llhood = torch.zeros_like(distance)
    bandwidths = [1, 5, 10]
    for s in bandwidths:
        kernel_llhood += (1./len(bandwidths))*torch.exp(-distance/(2*s*s))
    return kernel_llhood

def mmd(x, y):
    K_xx = torch.mean(kernel(x, x))
    K_yy = torch.mean(kernel(y, y))
    K_xy = torch.mean(kernel(x, y))
    return K_xx + K_yy - 2*K_xy


@Model.register('cmmd')
class ConditionalMMD(GroupBase):
    def __init__(
        self, 
        vocab: Vocabulary,
        model: Model,
        weight_adapt: float,
        label_namespace: str = "labels",
        namespace: str = "tokens",
        initializer: InitializerApplicator = InitializerApplicator(),
        metrics: Optional[Dict[str, Metric]] = None,
        **kwargs,
    ) -> None:
        super().__init__(vocab, model, weight_adapt, label_namespace, namespace, initializer, metrics, **kwargs)

    
    @overrides
    def calculate_penalty(self, all_loss, logits, labels, envs, unique_env_count, hidden=None) -> torch.Tensor:
        class_part_envs = dynamic_partition(envs, labels, self._label_count)
        class_part_hiddens = dynamic_partition(hidden, labels, self._label_count)
        penalty = torch.tensor(0.0).to(logits.device)
        for class_hiddens, class_envs in zip(class_part_hiddens, class_part_envs):
            if len(class_envs) == 0: continue 
            env_part_class_hiddens = dynamic_partition(class_hiddens, class_envs, unique_env_count)
            assert len(env_part_class_hiddens) == 2
            if len(env_part_class_hiddens[0]) == 0 or len(env_part_class_hiddens[1]) == 0: continue
            dist = mmd(env_part_class_hiddens[0], env_part_class_hiddens[1])
            penalty += dist
        penalty /= self._label_count
        return penalty
    
    @overrides
    def calculate_loss(self, all_loss, logits, labels, envs, unique_env_count, hidden=None) -> torch.Tensor:
        env_part_losses = dynamic_partition(all_loss, envs, unique_env_count)
        final_loss = 0.0
        for env_loss in env_part_losses:
            if len(env_loss) == 0: continue
            final_loss += env_loss.mean()
        return final_loss / unique_env_count
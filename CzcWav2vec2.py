
_HIDDEN_STATES_START_POSITION = 2
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForSequenceClassification
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
def _compute_mask_indices_random(
    shape: Tuple[int, int],
    mask_prob: float,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 1,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement `SpecAugment: A Simple Data Augmentation Method for
    ASR <https://arxiv.org/abs/1904.08779>`__. Note that this method is not optimized to run on TPU and should be run
    on CPU as part of the preprocessing during training.

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_length: size of the mask
        min_masks: minimum number of masked spans
    """
    batch_size, sequence_length = shape
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked indices <= sequence_length
        num_masked_span = sequence_length if num_masked_span > sequence_length else num_masked_span
        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )
    # print(f"input_lengths = {input_lengths}")
    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=np.bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)
    
    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)
        # print(f"num_masked_span = {num_masked_span}")
        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length), num_masked_span, replace=False
        )
        # print(f"spec_aug_mask_idx = {spec_aug_mask_idx}")

        # pick first sampled index that will serve as a dummy index to pad vector
        dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask
def _compute_chunck_indices(
    shape: Tuple[int, int],
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement `SpecAugment: A Simple Data Augmentation Method for
    ASR <https://arxiv.org/abs/1904.08779>`__. Note that this method is not optimized to run on TPU and should be run
    on CPU as part of the preprocessing during training.

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_length: size of the mask
        min_masks: minimum number of masked spans
    """
    batch_size, sequence_length = shape
    # only one mask
    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`"
        )
        
    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )
    # print(f"input_lengths = {input_lengths}")
    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=np.bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = num_masked_span = 1
    
    for input_length in input_lengths:
        # print(f"num_masked_span = {num_masked_span}")
        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )
        # print(f"spec_aug_mask_idx = {spec_aug_mask_idx}")

        # pick first sampled index that will serve as a dummy index to pad vector
        # dummy_mask_idx = spec_aug_mask_idx[0]

        # spec_aug_mask_idx = np.concatenate(
            # [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        # )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask
def get_czc_scheduler(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, lr_end=1e-7, power=2.0, last_epoch: int = -1
):
    lr_init = optimizer.defaults["lr"]
    assert lr_init > lr_end, f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})"
    def lr_lambda(current_step):
        num_last_steps = 0.6 * num_training_steps
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_last_steps:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_last_steps
            pct_remaining = 1 - (current_step - num_last_steps) / decay_steps
            decay = lr_range * pct_remaining ** power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init 不是return 1
        else:
            return 1  # as LambdaLR multiplies by lr_init 不是return 1
        
    return LambdaLR(optimizer, lr_lambda, last_epoch)
def get_w2v2_scheduler(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch: int = -1
):
    def lr_lambda(current_step):
        num_last_steps = 0.5 * num_training_steps
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_last_steps:
            return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_last_steps))
        )        
        return 1

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.functional.gelu
    def forward(self, features, output_hidden_states=False, **kwargs):
        x = features
        x = self.dropout(x)
        hidden_states = self.dense(x)
        x = self.activation(hidden_states)
        x = self.dropout(x)
        x = self.out_proj(x)
        # return x,torch.cat((features, hidden_states), dim=-1) if output_hidden_states else x
        if output_hidden_states:
            return x,hidden_states
        else:
            return x

class Wav2Vec2ForSequenceClassification_czc(Wav2Vec2ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.config.pool_mode = "mean"
        self.config.lsm = False
        self.config.use_am_softmax_loss = False
        if not self.config.do_proj:
            self.projector = None
        self.config.use_affine_classifier = True
        if self.config.pool_mode == "statistics_asp":
            self.asp_layer = AttentiveStatisticsPooling(self.config.hidden_size)
        self.config.pool_hidden_size = self.config.hidden_size * 2 if self.config.pool_mode.startswith("statistics") else self.config.hidden_size
        if self.config.use_affine_classifier:
            self.classifier = Wav2Vec2ClassificationHead(self.config, input_dim=self.config.pool_hidden_size, hidden_dim=self.config.hidden_size, output_dim=self.config.num_labels)
        else:
            self.classifier = nn.Linear(self.config.pool_hidden_size, self.config.num_labels)
        self.init_weights()
    def compute_mean_std(
        self,
        hidden_states,
        attention_mask,
    ):
        mean = (hidden_states*((attention_mask/attention_mask.sum(-1,keepdim=True)).unsqueeze(-1))).sum(1)
        var_tmp = (hidden_states - mean.unsqueeze(dim=1)).pow(2)
        std = torch.sqrt((var_tmp*(attention_mask.unsqueeze(-1))).sum(dim=1) / attention_mask.sum(dim=1).view(-1, 1))
        return mean,std
    def LabelSmoothingLoss(
        self,
        logits,
        labels,
        smoothing: float = 0.1,
        padding_idx: int = -100,
        num_class: int = 32,
        normalize_length: bool = False,
        reduction='sum',
    ):
        criterion = nn.KLDivLoss(reduction="none")
        confidence = 1.0 - smoothing
        assert logits.size(1) == num_class
        logits = logits.view(-1, num_class)
        labels = labels.view(-1)
        true_dist = torch.zeros_like(logits)
        # 先用0.1/(32-1)填满，再将对应的标签位置为0.9
        true_dist.fill_(smoothing / (num_class - 1))
        # padding_index=-100用0代替，否则计算loss会报错，选用0是因为它在labels中并不会出现
        ignore = labels == padding_idx  # (B,)
        # 默认loss为sum/batch_size，若normalize_length = True则sum/tokens
        total = len(labels) - ignore.sum().item()
        labels = labels.masked_fill(ignore, 0)  # avoid -1 index
        true_dist.scatter_(1, labels.unsqueeze(1), confidence)
        kl = criterion(torch.log_softmax(logits, dim=-1), true_dist).masked_fill(ignore.unsqueeze(1), 0)  # .sum()
        if reduction == "mean":
            return kl.sum() / total
        return kl.sum()
    def merged_strategy(
        self,
        hidden_states_lid,
        attention_mask,
        mode="mean",
    ):
        if mode == "mean":
            pooled_output = self.compute_mean_std(hidden_states_lid,attention_mask)[0]
        elif mode == "max":
            pooled_output = hidden_states_lid.max(dim=1)[0]
        elif mode == "min":
            pooled_output = hidden_states_lid.min(dim=1)[0]
        elif mode == "statistics":
            mean, std = self.compute_mean_std(hidden_states_lid, attention_mask)
            pooled_output = torch.cat((mean, std), dim=1)
        elif mode == "statistics_asp":
            if attention_mask is not None:
                lengths = attention_mask.sum(-1)
            pooled_output = self.asp_layer(hidden_states_lid.transpose(1, 2), lengths).transpose(1, 2).squeeze(1)
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'min', 'max', 'statistics', 'statistics_asp']")

        return pooled_output
        
    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        pitch_embeddings=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        extract_features = outputs[1]
        if attention_mask is not None:
            attention_mask_ = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)
        else:
            attention_mask_ = torch.ones((extract_features.shape[0], extract_features.shape[1]), device=extract_features.device).bool()

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]
        if self.projector is not None:
            # hidden_states = self.projector_ln(hidden_states)
            hidden_states = self.projector(hidden_states)
            

        hidden_states_lid = hidden_states
        # hidden_states_lid = self.lid_proj(hidden_states)
        pooled_output = self.merged_strategy(hidden_states_lid=hidden_states_lid, attention_mask=attention_mask_, mode=self.config.pool_mode)

        
        if not self.config.use_am_softmax_loss:
            logits = self.classifier(pooled_output)

        loss = None
        if labels is not None or self.training:
            if self.config.lsm:
                loss = self.LabelSmoothingLoss(logits=logits, labels=labels, num_class=self.config.num_labels, smoothing=0.05)
            elif self.config.use_am_softmax_loss:
                loss, logits = self.am_softmax_layer(pooled_output.unsqueeze(2), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss(reduction='sum')
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        elif self.config.use_am_softmax_loss:
            logits = self.am_softmax_layer(pooled_output.unsqueeze(2), None)
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def main():
    model = Wav2Vec2ForSequenceClassification_czc.from_pretrained("XLSR-53")

    # 在被import时__name__不等于__main__，则不会进入main(), 当直接执行本脚本时，__name__=__main__
if __name__ == "__main__":
    main()

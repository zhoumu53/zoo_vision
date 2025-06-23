from typing import Optional, Union
import torch
from torch import nn
from transformers.models.vit.configuration_vit import ViTConfig
from transformers.models.vit.modeling_vit import (
    ViTModel,
    ViTPreTrainedModel,
)
from transformers.modeling_outputs import (
    ImageClassifierOutput,
)
from torch.nn import CrossEntropyLoss


class ViTForTrackClassification(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vit = ViTModel(config, add_pooling_layer=False)
        self.config.problem_type = "single_label_classification"
        self.loss_fct: torch.nn.Module = CrossEntropyLoss(label_smoothing=0.2)

        # Classifier head
        self.classifier = (
            nn.Linear(config.hidden_size, config.num_labels)
            if config.num_labels > 0
            else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        assert len(pixel_values.shape) == 5
        batch_length, sequence_length, channels, height, width = pixel_values.shape
        assert channels == 3

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        pixel_values = pixel_values.reshape(
            [-1, channels, height, width]
        )  # Collapse sequence

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        hidden_state = outputs[0]
        _, hidden_count, feature_count = hidden_state.shape
        hidden_state = hidden_state.reshape(
            [batch_length, sequence_length, hidden_count, feature_count]
        )  # Restore sequence dim

        sequence_outputs = hidden_state[:, :, 0, :]
        output = torch.sum(sequence_outputs, dim=1)

        logits = self.classifier(output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)

            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

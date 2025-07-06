import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput

class SegFormerHead(nn.Module):
    def __init__(self, in_channels, hidden_size, num_classes):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_size)
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Conv2d(hidden_size, num_classes, kernel_size=1)

    def forward(self, x, height, width):
        B, N, C = x.shape
        h = w = int(N ** 0.5)
        x = x.view(B, h, w, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        x = self.linear(x)                          # [B, hidden_size, H, W]
        x = self.dropout(x)
        x = self.classifier(x)                      # [B, num_classes, H, W]
        x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
        return x

# Your main segmentation model using Dinov2 + custom head
class Dinov2ForSemanticSegmentation(Dinov2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.dinov2 = Dinov2Model(config)
        self.classifier = SegFormerHead(
            in_channels=config.hidden_size,
            hidden_size=256,
            num_classes=config.num_labels
        )

    def forward(self, pixel_values, output_hidden_states=False, output_attentions=False, labels=None):
        outputs = self.dinov2(pixel_values,
                              output_hidden_states=output_hidden_states,
                              output_attentions=output_attentions)

        # Drop CLS token
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]  # [B, N-1, C]

        # Classifier head
        logits = self.classifier(
            patch_embeddings,
            height=pixel_values.shape[2],
            width=pixel_values.shape[3]
        )

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fct(logits, labels)

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

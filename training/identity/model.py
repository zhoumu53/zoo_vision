import torchvision as tv
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchinfo


class ZooIdGruModel(torch.nn.Module):
    def __init__(self, backbone_features):
        super().__init__()
        self.feature_count = 1024
        self.individual_count = 5
        self.features = backbone_features
        self.gru_layer_count = 3

        self.gru_feature_count = 128

        self.gru = torch.nn.GRU(
            input_size=self.feature_count,
            hidden_size=self.gru_feature_count,
            num_layers=self.gru_layer_count,
            batch_first=True,
            dropout=0.2,
        )
        self.classifier = nn.Linear(self.gru_feature_count, self.individual_count)

    def forward(
        self, x: torch.Tensor, gru_state0: torch.Tensor | None = None
    ) -> dict["str", torch.Tensor]:
        batch_count, time_count, channels, height, width = x.shape

        COMPUTE_TIME_FEATURES_SEPARATELY = True
        if COMPUTE_TIME_FEATURES_SEPARATELY:
            # Compute each time feature with a separate call so we get the same results
            # when passing images in bulk or sequentially
            features_list = []
            for t in range(time_count):
                xt = x[:, t, :, :, :]
                out = self.features(xt)
                out = F.relu(out, inplace=True)
                out = F.adaptive_avg_pool2d(out, (1, 1))
                out = torch.flatten(out, 1)
                features_list.append(out)
            features = torch.stack(features_list, dim=1)
        else:
            # Compute all features at the same time
            xt = x.reshape(-1, channels, height, width)  # Remove time
            out = self.features(xt)
            out = F.relu(out, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            features = out.reshape([batch_count, time_count, -1])  # Add time

        out, gru_state = self.gru(features, gru_state0)

        out = out.reshape([time_count * batch_count, -1])  # Remove time
        out = self.classifier(out)

        out = out.reshape([batch_count, time_count, -1])  # Add time

        return {"logits": out, "gru_state": gru_state}


def get_model(
    model_name: str, num_classes: int, weights: str | None = None
) -> nn.Module:
    if model_name in ["densenet121"]:
        # This is the instant model, no time accumulation
        num_pretrained_classes = 1000
        model = tv.models.get_model(
            model_name, weights=weights, num_classes=num_pretrained_classes
        )
        if num_pretrained_classes != num_classes:
            print(
                "Replacing final classifier layer because pretrained class count doesn't match"
            )
            num_features = model.classifier.in_features
            model.classifier = nn.Linear(num_features, num_classes)

    elif model_name in ["zoo_id_gru"]:
        # GRU temporal model with a densenet model as the backbone

        # weights is the pretrained densenet121 model
        backbone = tv.models.densenet121(
            num_classes=5,
        )

        if weights is not None and "gru" not in weights:
            checkpoint = torch.load(weights, weights_only=False)
            weights = checkpoint["model"]
            backbone.load_state_dict(checkpoint["model"])

        model = ZooIdGruModel(backbone_features=backbone.features)
        if weights is not None and "gru" in weights:
            checkpoint = torch.load(weights, weights_only=False)
            weights = checkpoint["model"]
            model.load_state_dict(checkpoint["model"])

    return model

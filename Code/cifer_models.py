import torch
import torch.nn as nn


class GNLeNet(nn.Module):
    """
    Lightweight LeNet-style CNN for MNIST with Group Normalization.

    Architecture:
        Input: 3 x 32 x 32

        Conv1:
            3-> 16 channels
            GroupNorm
            ReLU
            MaxPool
            Output: 16 x 16 x 16

        Conv2:
            16 -> 32 channels
            GroupNorm
            ReLU
            MaxPool
            Output: 32 x 8 x 8

        Fully Connected:
            32*7*7 -> num_classes

    Why GroupNorm?
    ----------------
    GroupNorm does not depend on batch-level running statistics,
    unlike BatchNorm.

    This makes it convenient for decentralized/federated
    experiments where:
        - clients can have heterogeneous/non-IID data
        - local batch sizes may differ
        - models are averaged across clients

    IMPORTANT:
    GroupNorm does NOT make the model "invariant to non-IID data".
    Non-IID heterogeneity is handled by the proposed topology
    learning framework, especially Dynamic Statistical Alignment.
    """

    def __init__(
        self,
        num_classes=10
    ):
        super().__init__()

        # ====================================================
        # Convolution Block 1
        # ====================================================
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2
        )

        self.gn1 = nn.GroupNorm(
            num_groups=4,
            num_channels=16
        )

        # ====================================================
        # Convolution Block 2
        # ====================================================
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2
        )

        self.gn2 = nn.GroupNorm(
            num_groups=4,
            num_channels=32
        )

        # ====================================================
        # Shared activation / pooling
        # ====================================================
        self.relu = nn.ReLU(
            inplace=False
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ====================================================
        # Classification head
        #
        # MNIST:
        #
        # 28x28
        #   ↓ pool
        # 14x14
        #   ↓ pool
        # 7x7
        #
        # 32 feature maps -> 32*7*7 features
        # ====================================================
        self.fc = nn.Linear(
            32 * 8 * 8,
            num_classes
        )

    def forward(self, x):

        # Block 1
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Block 2
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten all dimensions except batch.
        x = torch.flatten(
            x,
            start_dim=1
        )

        # Classification
        x = self.fc(x)

        # Return raw logits.
        # CrossEntropyLoss applies the required softmax internally.
        return x


def count_trainable_parameters(model):
    """
    Utility function for experiment documentation/logging.
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":
    # Simple architecture sanity check.
    model = GNLeNet()

    dummy_input = torch.randn(
        4,
        3,
        32,
        32
    )

    output = model(
        dummy_input
    )

    print(model)

    print(
        "\nOutput shape:",
        tuple(output.shape)
    )

    print(
        "Trainable parameters:",
        count_trainable_parameters(model)
    )

    assert output.shape == (4, 10)

    print(
        "\nModel sanity check passed."
    )
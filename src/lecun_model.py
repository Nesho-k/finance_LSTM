import torch
import torch.nn as nn


class RN(nn.Module):
    """
    CNN pour la reconnaissance de chiffres manuscrits (architecture LeCun 1989).
    Entrée: image 28x28 (1 canal), normalisée entre -1 et 1
    Sortie: 10 logits (un par classe 0-9)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=0
        )
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(
            in_channels=4, out_channels=12, kernel_size=5, stride=1, padding=0
        )
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(12 * 4 * 4, 10)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.float()
        x = self.tanh(self.conv1(x))
        x = self.avgpool1(x)
        x = self.tanh(self.conv2(x))
        x = self.avgpool2(x)
        x = self.flatten(x)
        x = x.view(x.shape[0], 12 * 4 * 4)
        x = self.dense(x)
        return x

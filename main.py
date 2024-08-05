import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
import operator
import functools


class STERound(torch.autograd.Function):
    @staticmethod
    def forward(_, x: Tensor) -> Tensor:
        return x.round()

    @staticmethod
    def backward(_, grad_output: Tensor) -> Tensor:
        return grad_output


def ste_round(x: Tensor) -> Tensor:
    return STERound.apply(x)


def q(x: Tensor, b: Tensor, e: Tensor) -> Tensor:
    return 2**e * ste_round(
        torch.min(torch.max((2**-e) * x, -(2 ** (b - 1))), 2 ** (b - 1) - 1)
    )


def prod(seq):
    return functools.reduce(operator.mul, seq, 1)


class QConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(QConv2d, self).__init__(*args, **kwargs)
        out_channels = self.weight.size()[0]
        self.e = nn.Parameter(torch.full((out_channels, 1, 1, 1), -8).to(torch.float32))
        self._b = nn.Parameter(
            torch.full((out_channels, 1, 1, 1), 2).to(torch.float32)
        )  # 8-bit quantization at the beginning

    @property
    def b(self) -> Tensor:
        return self._b.relu()  # constraint b to be >= 0

    def bits_used(self) -> Tensor:
        return self.b.sum() * prod(self.weight.shape[1:])

    def forward(self, x: Tensor) -> Tensor:
        quantized_weight = q(self.weight, self.b, self.e)
        return F.conv2d(
            x,
            quantized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class SelfCompressingNN(nn.Module):
    def __init__(self):
        super(SelfCompressingNN, self).__init__()
        self.conv1 = QConv2d(3, 64, 3, padding=1)
        self.conv2 = QConv2d(64, 64, 3, padding=1)
        self.conv3 = QConv2d(64, 128, 3, padding=1)
        self.conv4 = QConv2d(128, 128, 3, padding=1)
        self.conv_linear = QConv2d(128 * 8 * 8, 10, 1)  # acts as a linear layer
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.bn2(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1).view(-1, 128 * 8 * 8, 1, 1)
        x = F.relu(self.conv_linear(x))  # acts as a linear layer
        return x.view(-1, 10)


def loss_to_minimize_bits(model: SelfCompressingNN) -> Tensor:
    total_weights = sum(
        [
            layer.weight.numel()
            for layer in model.modules()
            if isinstance(layer, QConv2d)
        ]
    )
    return (
        sum(
            [
                layer.bits_used() if hasattr(layer, "bits_used") else 0
                for layer in model.modules()
            ]
        )
        / total_weights
    )


def get_validation(
    model: SelfCompressingNN,
    xentropy_crit: nn.CrossEntropyLoss,
    valloader: torch.utils.data.DataLoader,
    device: str,
) -> Tensor:
    model.eval()
    with torch.no_grad():
        xent = 0
        total_right = 0
        total = 0
        for inputs, targets in valloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            xent += xentropy_crit(outputs, targets)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            total_right += predicted.eq(targets).sum().item()
        return xent / len(valloader), total_right / total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--y", type=float, default=0.015)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.set_device(0)
    device = "cuda"

    model = SelfCompressingNN().to(device)
    # model = torch.compile(model)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    xentropy_crit = nn.CrossEntropyLoss()
    cifar = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    cifar_val = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    trainloader = torch.utils.data.DataLoader(
        cifar, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    valloader = torch.utils.data.DataLoader(
        cifar_val, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    def extract_bits_used(model: SelfCompressingNN) -> float:
        total = 0
        for layer in model.modules():
            if isinstance(layer, QConv2d):
                wshape = layer.weight.shape
                total += layer.b.sum().item() * prod(wshape[1:])
        return total

    for epoch in range(args.epochs):
        pbar = tqdm(trainloader, desc="Training", position=0)
        model.train()
        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            optim.zero_grad()
            outputs = model(inputs)
            xent = xentropy_crit(outputs, targets)
            bits_loss = loss_to_minimize_bits(model)
            (xent + args.y * bits_loss).backward()
            bits_used_by_convs = extract_bits_used(model)
            kbs = bits_used_by_convs / 8 / 1024
            optim.step()
            pbar.set_postfix(
                {"xentropy": f"{xent.item():.4f}", "KBs": f"{kbs:.4f}", "epoch": epoch}
            )

        val_xent, val_acc = get_validation(model, xentropy_crit, valloader, device)
        print(
            f"Validation cross-entropy: {val_xent.item()}, Validation accuracy: {val_acc}"
        )

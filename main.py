import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
import operator
import functools
from typing import Dict, Sequence


# ********* UTILS *********
def prod(seq: Sequence[int]) -> int:
    return functools.reduce(operator.mul, seq, 1)


# ********* QUANTIZATION *********
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


# ********* MODEL *********
class QConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(QConv2d, self).__init__(*args, **kwargs)
        out_channels = self.weight.size()[0]
        self.e = nn.Parameter(torch.full((out_channels, 1, 1, 1), -8).to(torch.float32))
        self._b = nn.Parameter(
            torch.full((out_channels, 1, 1, 1), 8).to(torch.float32)
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


class Reshape(nn.Module):
    def __init__(self, shape: Sequence[int]):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(*self.shape)


class SelfCompressingNN(nn.Module):
    def __init__(self):
        super(SelfCompressingNN, self).__init__()
        self.seq = nn.Sequential(
            QConv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            QConv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            QConv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            QConv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            QConv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            QConv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            QConv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            QConv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            QConv2d(
                64, 10, 4, padding=0
            ),  # simulates a fully connected layer with conv2d. Expects inpujt of size (b, 64, 4, 4)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x).reshape(x.size(0), -1)

    def quant_loss(self) -> Tensor:
        convs = [layer for layer in self.seq.modules() if isinstance(layer, QConv2d)]

        def conv_cost(prev: QConv2d, out: QConv2d) -> Tensor:
            outcost = (
                prod(out.weight.shape[2:])
                * (
                    torch.where(prev.b > 0, 1, 0).reshape(-1, 1) * out.b.reshape(1, -1)
                ).sum()
            )
            prevcost = (
                prod(prev.weight.shape[2:])
                * (
                    torch.where(out.b > 0, 1, 0).reshape(-1, 1) * prev.b.reshape(1, -1)
                ).sum()
            )
            return outcost + prevcost

        total_weights = sum([layer.weight.numel() for layer in convs])
        return (
            sum([conv_cost(prev, out) for prev, out in zip(convs, convs[1:])])
            / total_weights
        )

    def get_params(self) -> Dict[str, list[nn.Parameter]]:
        d = {"quant": [], "other": []}
        for name, param in self.named_parameters():
            if name.endswith("b") or name.endswith("e"):
                d["quant"].append(param)
            else:
                d["other"].append(param)
        return d

    def get_stats(self) -> Dict[str, float]:
        d = {
            "total": 0,
            "used": 0,
            "pruned": 0,
            "avg_bits_all_channels": 0,
            "avg_bits_used_channels": 0,
            "orig_kb_size": 0,
            "current_kb_size": 0,
        }
        for layer in self.modules():
            if isinstance(layer, QConv2d):
                wshape = layer.weight.shape
                total = prod(wshape)
                pruned = (layer.b == 0).sum().item() * prod(wshape[1:])
                used = total - pruned
                d["total"] += total
                d["used"] += used
                d["pruned"] += pruned
                d["avg_bits_all_channels"] += layer.bits_used().item() / total
                d["avg_bits_used_channels"] += layer.bits_used().item() / used
                d["orig_kb_size"] += total * 32 / 8 / 1024
                d["current_kb_size"] += layer.bits_used().item() / 8 / 1024

        d["avg_bits_all_channels"] /= len(
            [layer for layer in self.modules() if isinstance(layer, QConv2d)]
        )
        d["avg_bits_used_channels"] /= len(
            [layer for layer in self.modules() if isinstance(layer, QConv2d)]
        )

        return d


def validate(
    model: SelfCompressingNN,
    valloader: torch.utils.data.DataLoader,
    device: str,
) -> Tensor:
    model.eval()
    xentropy_crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        xent = 0
        total_right = 0
        total = 0
        for inputs, targets in valloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            xent += xentropy_crit(outputs, targets)
            predicted = outputs.argmax(1)
            total += targets.size(0)
            total_right += predicted.eq(targets).sum().item()
        return xent / len(valloader), total_right / total


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cifar_stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    model = SelfCompressingNN().to(device)

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))

    model = torch.compile(model)
    params = model.get_params()
    optim = torch.optim.Adam(
        [
            {"params": params["quant"], "lr": args.lr_quant, "eps": args.eps_quant},
            {
                "params": params["other"],
                "lr": args.lr_other,
                "eps": args.eps_other,
                "weight_decay": args.weight_decay_other,
            },
        ]
    )

    import torchvision.transforms as transforms

    trans = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*cifar_stats),
        ]
    )
    xentropy_crit = nn.CrossEntropyLoss()
    cifar = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=trans
    )
    cifar_val = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(*cifar_stats),
            ]
        ),
    )
    trainloader = torch.utils.data.DataLoader(
        cifar, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    valloader = torch.utils.data.DataLoader(
        cifar_val, batch_size=args.batch_size * 2, shuffle=False, num_workers=2
    )
    for epoch in range(args.epochs):
        pbar = tqdm(trainloader, desc="Training", position=0)
        model.train()
        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            optim.zero_grad()
            outputs = model(inputs)
            xent = xentropy_crit(outputs, targets)
            (xent + args.y * model.quant_loss()).backward()
            optim.step()
            pbar.set_postfix({"xentropy": f"{xent.item():.4f}", "epoch": epoch})
        torch.save(model._orig_mod.state_dict(), "model.pth")

        val_xent, val_acc = validate(model, valloader, device)
        stats = model.get_stats()
        print(
            f"Validation cross-entropy: {val_xent.item()}, Validation accuracy: {val_acc}, Original size:\
                  {stats['orig_kb_size']:.2f}KB, Current size: {stats['current_kb_size']:.2f}KB, Average bits used per channel:\
                      {stats['avg_bits_all_channels']:.2f}, Average bits used per channel (excluding pruned): {stats['avg_bits_used_channels']:.2f}"
        )


def run_validation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cifar_stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    model = SelfCompressingNN().to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model = torch.compile(model)
    import torchvision.transforms as transforms

    cifar_val = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(*cifar_stats),
            ]
        ),
    )
    valloader = torch.utils.data.DataLoader(
        cifar_val, batch_size=args.batch_size * 2, shuffle=False, num_workers=2
    )
    val_xent, val_acc = validate(model, valloader, device)
    stats = model.get_stats()

    print(
        f"Validation cross-entropy: {val_xent.item()}, Validation accuracy: {val_acc}, Original size:\
                {stats['orig_kb_size']:.2f}KB, Current size: {stats['current_kb_size']:.2f}KB, Average bits used per channel:\
                    {stats['avg_bits_all_channels']:.2f}, Average bits used per channel (excluding pruned): {stats['avg_bits_used_channels']:.2f}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr_quant", type=float, default=0.5)
    parser.add_argument("--eps_quant", type=float, default=1e-3)
    parser.add_argument("--lr_other", type=float, default=1e-3)
    parser.add_argument("--eps_other", type=float, default=1e-5)
    parser.add_argument("--weight_decay_other", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=850)
    parser.add_argument("--y", type=float, default=0.015)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train")

    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.set_device(0)
    train(args)

import torch
from romav2 import RoMaV2
from romav2.device import device


def test_smoke():
    model = RoMaV2()
    model.apply_setting("base")
    model.match(
        torch.randn(1, 3, 640, 640).to(device), torch.randn(1, 3, 640, 640).to(device)
    )


if __name__ == "__main__":
    test_smoke()

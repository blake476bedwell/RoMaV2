import torch
from romav2 import RoMaV2
from romav2.device import device


def test_smoke():
    model = RoMaV2(RoMaV2.Cfg(compile=False))
    model.apply_setting("turbo")
    model.match(
        torch.randn(1, 3, 320, 320).to(device), torch.randn(1, 3, 320, 320).to(device)
    )


if __name__ == "__main__":
    test_smoke()

import torch
from torchvision.models import vit_b_16

from main_incremental import set_tvmodel_head_var
from networks.network import LLL_Net


def test_create_vit_basic_heads():
    backbone = vit_b_16(pretrained=False)
    image_batch = torch.randn(1, 3, 224, 224)
    set_tvmodel_head_var(backbone)

    lll_net = LLL_Net(
        backbone, remove_existing_head=True, ic_config="imagenet100_vit_base"
    )
    lll_net.add_head(10)
    y = lll_net(image_batch)
    assert len(y) == 12


def test_create_vit_ln_heads():
    backbone = vit_b_16(pretrained=False)
    image_batch = torch.randn(1, 3, 224, 224)
    set_tvmodel_head_var(backbone)

    lll_net = LLL_Net(
        backbone, remove_existing_head=True, ic_config="imagenet100_vit_ln"
    )
    lll_net.add_head(10)
    y = lll_net(image_batch)
    assert len(y) == 12

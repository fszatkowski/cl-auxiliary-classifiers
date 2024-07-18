import torch
from torchvision.models import resnet18, resnet50, vit_b_16

from main_incremental import set_tvmodel_head_var
from networks import resnet32
from networks.network import LLL_Net


def test_create_rn32_sdn():
    backbone = resnet32()
    image_batch = torch.randn(1, 3, 32, 32)

    lll_net = LLL_Net(
        backbone, remove_existing_head=True, ic_config="cifar100_resnet32_sdn"
    )
    lll_net.add_head(10)
    y = lll_net(image_batch)
    assert len(y) == 7


def test_create_rn32_prop():
    backbone = resnet32()
    image_batch = torch.randn(1, 3, 32, 32)

    lll_net = LLL_Net(
        backbone, remove_existing_head=True, ic_config="cifar100_resnet32_prop"
    )
    lll_net.add_head(10)
    y = lll_net(image_batch)
    assert len(y) == 7


def test_create_rn32_dense():
    backbone = resnet32()
    image_batch = torch.randn(1, 3, 32, 32)

    lll_net = LLL_Net(
        backbone, remove_existing_head=True, ic_config="cifar100_resnet32_dense"
    )
    lll_net.add_head(10)
    y = lll_net(image_batch)
    assert len(y) == 15


def test_create_rn18_sdn():
    backbone = resnet18(pretrained=False)
    image_batch = torch.randn(1, 3, 224, 224)
    set_tvmodel_head_var(backbone)

    lll_net = LLL_Net(
        backbone, remove_existing_head=True, ic_config="imagenet100_resnet18_sdn"
    )
    lll_net.add_head(10)
    y = lll_net(image_batch)
    assert len(y) == 7


def test_create_rn18_prop():
    backbone = resnet18(pretrained=False)
    image_batch = torch.randn(1, 3, 224, 224)
    set_tvmodel_head_var(backbone)

    lll_net = LLL_Net(
        backbone, remove_existing_head=True, ic_config="imagenet100_resnet18_sdn"
    )
    lll_net.add_head(10)
    y = lll_net(image_batch)
    assert len(y) == 7


def test_create_rn18_dense():
    backbone = resnet18(pretrained=False)
    image_batch = torch.randn(1, 3, 224, 224)
    set_tvmodel_head_var(backbone)

    lll_net = LLL_Net(
        backbone, remove_existing_head=True, ic_config="imagenet100_resnet18_dense"
    )
    lll_net.add_head(10)
    y = lll_net(image_batch)
    assert len(y) == 8


def test_create_rn50_dense():
    backbone = resnet50(pretrained=False)
    image_batch = torch.randn(1, 3, 224, 224)
    set_tvmodel_head_var(backbone)

    lll_net = LLL_Net(
        backbone, remove_existing_head=True, ic_config="imagenet100_resnet50_dense"
    )
    lll_net.add_head(10)
    y = lll_net(image_batch)
    assert len(y) == 16


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


def test_create_vit_base_dense_heads():
    backbone = vit_b_16(pretrained=False)
    image_batch = torch.randn(1, 3, 224, 224)
    set_tvmodel_head_var(backbone)

    lll_net = LLL_Net(
        backbone, remove_existing_head=True, ic_config="imagenet100_vit_base_dense"
    )
    lll_net.add_head(10)
    y = lll_net(image_batch)
    assert len(y) == 24


def test_create_vit_ln_dense_heads():
    backbone = vit_b_16(pretrained=False)
    image_batch = torch.randn(1, 3, 224, 224)
    set_tvmodel_head_var(backbone)

    lll_net = LLL_Net(
        backbone, remove_existing_head=True, ic_config="imagenet100_vit_ln_dense"
    )
    lll_net.add_head(10)
    y = lll_net(image_batch)
    assert len(y) == 24

"""Model shapes, gradients, parameter counts, and the layer choices in D11 and overview 7."""

import pytest
import torch
from torch import nn

from src.models import build_model
from src.models.resnet import BasicBlock, ResNet18
from src.models.simple_cnn import SimpleCNN
from src.utils import count_parameters, set_seed

NAMES = ["simple_cnn", "resnet18"]


@pytest.fixture(params=NAMES)
def model(request):
    set_seed(0)
    return build_model(request.param)


def test_forward_shape_and_no_nan(model):
    out = model(torch.randn(4, 3, 32, 32))
    assert out.shape == (4, 10)
    assert torch.isfinite(out).all()


def test_backward_reaches_every_parameter(model):
    loss = nn.functional.cross_entropy(model(torch.randn(4, 3, 32, 32)), torch.arange(4))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("name", NAMES)
def test_num_classes_sets_output_size(name):
    assert build_model(name, num_classes=7)(torch.randn(2, 3, 32, 32)).shape == (2, 7)


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="vgg"):
        build_model("vgg")


def test_parameter_counts_in_overview_ranges():
    assert 500_000 <= count_parameters(build_model("simple_cnn")) <= 1_000_000
    assert 11_100_000 <= count_parameters(build_model("resnet18")) <= 11_300_000


def test_count_parameters_skips_frozen():
    layer = nn.Linear(2, 3)
    assert count_parameters(layer) == 9
    layer.bias.requires_grad_(False)
    assert count_parameters(layer) == 6


def test_build_model_returns_expected_class():
    assert isinstance(build_model("simple_cnn"), SimpleCNN)
    assert isinstance(build_model("resnet18"), ResNet18)


def test_same_seed_same_init(model):
    name = "simple_cnn" if isinstance(model, SimpleCNN) else "resnet18"
    set_seed(0)
    other = build_model(name)
    for a, b in zip(model.state_dict().values(), other.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def _layer_tokens(sequential: nn.Sequential) -> list[str | int]:
    tokens: list[str | int] = []
    for layer in sequential:
        tokens.append(layer.out_channels if isinstance(layer, nn.Conv2d) else type(layer).__name__)
    return tokens


def test_simple_cnn_layer_order():
    cnn = SimpleCNN()
    bn_relu = ["BatchNorm2d", "ReLU"]
    pool_drop = ["MaxPool2d", "Dropout"]
    assert _layer_tokens(cnn.features) == [
        *[32, *bn_relu, 32, *bn_relu, *pool_drop],
        *[64, *bn_relu, 64, *bn_relu, *pool_drop],
        *[128, *bn_relu, *pool_drop],
    ]
    assert _layer_tokens(cnn.classifier) == ["Flatten", "Linear", "ReLU", "Dropout", "Linear"]
    assert cnn.classifier[1].out_features == 256


def test_simple_cnn_matches_d11():
    cnn = SimpleCNN()
    convs = [m for m in cnn.modules() if isinstance(m, nn.Conv2d)]
    assert len(convs) == 5
    assert all(c.kernel_size == (3, 3) and c.padding == (1, 1) for c in convs)
    assert [m.p for m in cnn.modules() if isinstance(m, nn.Dropout)] == [0.25, 0.25, 0.25, 0.5]


def test_resnet_stem_has_no_downsampling():
    net = ResNet18()
    assert net.conv1.kernel_size == (3, 3)
    assert net.conv1.stride == (1, 1)
    assert net.conv1.padding == (1, 1)
    assert net.conv1.out_channels == 64
    assert not any(isinstance(m, nn.MaxPool2d) for m in net.modules())


def test_resnet_stages():
    net = ResNet18()
    stages = [net.layer1, net.layer2, net.layer3, net.layer4]
    assert [len(s) for s in stages] == [2, 2, 2, 2]
    assert all(isinstance(b, BasicBlock) for s in stages for b in s)
    assert [s[-1].conv2.out_channels for s in stages] == [64, 128, 256, 512]
    assert [s[0].conv1.stride for s in stages] == [(1, 1), (2, 2), (2, 2), (2, 2)]
    assert isinstance(net.layer1[0].shortcut, nn.Identity)
    for stage in stages[1:]:
        projection = stage[0].shortcut[0]
        assert projection.kernel_size == (1, 1)
        assert projection.stride == (2, 2)
    assert net.fc.in_features == 512


def test_resnet_init():
    set_seed(0)
    net = ResNet18()
    norms = [m for m in net.modules() if isinstance(m, nn.BatchNorm2d)]
    assert all((m.weight == 1).all() and (m.bias == 0).all() for m in norms)
    # Kaiming fan_out, relu: std = sqrt(2 / fan_out); fan_out = 3 * 3 * 512 for this conv.
    weight = net.layer4[1].conv2.weight
    assert weight.std().item() == pytest.approx((2 / (9 * 512)) ** 0.5, rel=0.02)

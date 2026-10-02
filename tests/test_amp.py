"""Run with python -m unittest discover -s tests on supported PyTorch versions."""

import importlib
import unittest

import torch
import torch.nn.functional as F

from yolov6.utils import amp


LOSS_MODULES = ('loss', 'loss_distill', 'loss_distill_ns', 'loss_fuseab')


class AMPCompatibilityTests(unittest.TestCase):
    def test_disabled_autocast_preserves_cpu_float32(self):
        with amp.autocast(enabled=False):
            result = torch.ones(2, 2).matmul(torch.ones(2, 2))
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.equal(result, torch.full((2, 2), 2.0)))

    def test_disabled_scaler_updates_parameters(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scaler = amp.GradScaler(enabled=False, init_scale=128.0)
        scaler.scale(parameter.sum()).backward()
        scaler.step(optimizer)
        scaler.update()
        self.assertTrue(torch.allclose(parameter, torch.tensor([0.9])))

    def check_varifocal_losses(self, device):
        for module_name in LOSS_MODULES:
            with self.subTest(loss=module_name, device=device):
                loss_type = importlib.import_module(
                    'yolov6.models.losses.' + module_name
                ).VarifocalLoss
                prediction = torch.tensor([[0.25, 0.75]], device=device, requires_grad=True)
                score = torch.tensor([[0.0, 0.8]], device=device)
                label = torch.tensor([[0.0, 1.0]], device=device)
                weight = 0.75 * prediction.pow(2.0) * (1 - label) + score * label
                expected = (F.binary_cross_entropy(prediction, score, reduction='none') * weight).sum()
                with amp.autocast(enabled=device == 'cuda'):
                    result = loss_type()(prediction, score, label)
                self.assertEqual(result.dtype, torch.float32)
                self.assertTrue(torch.allclose(result, expected))
                result.backward()
                self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_varifocal_losses_without_cuda(self):
        self.check_varifocal_losses('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA device required')
    def test_cuda_autocast_uses_float16(self):
        inputs = torch.ones(2, 2, device='cuda')
        with amp.autocast(enabled=True):
            result = inputs.matmul(inputs)
        self.assertEqual(result.dtype, torch.float16)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA device required')
    def test_cuda_scaler_updates_parameters(self):
        model = torch.nn.Linear(2, 1).cuda()
        with torch.no_grad():
            model.weight.fill_(0.5)
            model.bias.zero_()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = amp.GradScaler(enabled=True, init_scale=128.0)
        initial_weight = model.weight.detach().clone()
        with amp.autocast(enabled=True):
            prediction = model(torch.ones(2, 2, device='cuda'))
            loss = F.mse_loss(prediction, torch.zeros_like(prediction))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        self.assertTrue(torch.isfinite(model.weight).all())
        self.assertFalse(torch.equal(initial_weight, model.weight))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA device required')
    def test_varifocal_losses_inside_cuda_autocast(self):
        self.check_varifocal_losses('cuda')


if __name__ == '__main__':
    unittest.main()

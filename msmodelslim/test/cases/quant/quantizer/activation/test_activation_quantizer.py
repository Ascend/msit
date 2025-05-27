import unittest
from unittest.mock import patch

import torch

from msmodelslim.quant.quantizer.activation.base import (
    ActQuantConfig,
    ActQuantBaseConfig,
    ActQuantMethodConfig,
    ActQuantScopeConfig
)
from msmodelslim.quant.quantizer.activation.quantizer import ActivationQuantizer
from msmodelslim.quant.quantizer.base.const import (
    QuantMethod,
    QuantScope
)


class TestActivationQuantizer(unittest.TestCase):
    def setUp(self):
        self.config = ActQuantConfig(
            base=ActQuantBaseConfig(bits=8),
            method=ActQuantMethodConfig(type=QuantMethod.MINMAX),
            scope=ActQuantScopeConfig(type=QuantScope.PER_TENSOR)
        )
        self.quantizer = ActivationQuantizer(config=self.config)

    @patch('msmodelslim.quant.quantizer.activation.quantizer.linear_quantization_params')
    def test_get_scale_and_zero_point(self, mock_linear_quantization_params):
        # 模拟统计数据
        self.quantizer.strategy.min_val = torch.tensor(0.0)
        self.quantizer.strategy.max_val = torch.tensor(255.0)
        mock_linear_quantization_params.return_value = (torch.tensor(0.0), torch.tensor(0.0))
        scale, zero_point = self.quantizer.get_scale_offset()
        mock_linear_quantization_params.assert_called_once_with(8, torch.tensor(0.0), torch.tensor(255.0), True, True,
                                                                False)

    @patch('msmodelslim.quant.quantizer.activation.quantizer.fake_quantize')
    @patch('msmodelslim.quant.quantizer.activation.quantizer.linear_quantization_params')
    def test_forward(self, mock_linear_quantization_params, mock_fake_quantize):
        # 设置mock的返回值
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        expected_output = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mock_fake_quantize.return_value = (expected_output, expected_output)
        mock_linear_quantization_params.return_value = (torch.tensor(0.0), torch.tensor(0.0))

        # 调用量化器
        output = self.quantizer(x)

        # 验证linear_quantization_params被正确调用
        mock_linear_quantization_params.assert_called_once_with(8, torch.tensor(1.0), torch.tensor(4.0), True, True,
                                                                False)

        # 验证fake_quantize被正确调用
        mock_fake_quantize.assert_called_once_with(x, torch.tensor(0.0), torch.tensor(0.0), 8, True)

        # 验证输出
        self.assertEqual(output.shape, x.shape)
        torch.testing.assert_close(output, expected_output)

    @patch('msmodelslim.quant.quantizer.activation.quantizer.fake_quantize')
    @patch('msmodelslim.quant.quantizer.activation.quantizer.linear_quantization_params')
    def test_forward_with_different_scale(self, mock_linear_quantization_params, mock_fake_quantize):
        # 设置不同的scale和zero_point
        self.quantizer.strategy.min_val = torch.tensor(-10.0)
        self.quantizer.strategy.max_val = torch.tensor(10.0)

        # 设置mock的返回值
        x = torch.tensor([[-5.0, 0.0], [5.0, 10.0]])
        expected_output = torch.tensor([[-5.0, 0.0], [5.0, 10.0]])
        mock_fake_quantize.return_value = (expected_output, expected_output)
        mock_linear_quantization_params.return_value = (torch.tensor(0.0), torch.tensor(0.0))

        # 调用量化器
        output = self.quantizer(x)

        # 验证linear_quantization_params被正确调用
        mock_linear_quantization_params.assert_called_once_with(8, torch.tensor(-10.0), torch.tensor(10.0), True, True,
                                                                False)

        # 验证fake_quantize被正确调用
        mock_fake_quantize.assert_called_once_with(x, torch.tensor(0.0), torch.tensor(0.0), 8, True)

        # 验证输出
        self.assertEqual(output.shape, x.shape)
        torch.testing.assert_close(output, expected_output)

    @patch('msmodelslim.quant.quantizer.activation.quantizer.fake_quantize')
    @patch('msmodelslim.quant.quantizer.activation.quantizer.linear_quantization_params')
    def test_asymmetric_quantization(self, mock_linear_quantization_params, mock_fake_quantize):
        """测试非对称量化（min和max不相等且不关于0对称）"""

        # 设置mock的返回值
        x = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
        expected_output = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
        mock_fake_quantize.return_value = (expected_output, expected_output)
        mock_linear_quantization_params.return_value = (torch.tensor(0.0), torch.tensor(0.0))

        # 调用量化器
        output = self.quantizer(x)

        # 验证linear_quantization_params被正确调用
        mock_linear_quantization_params.assert_called_once_with(8, torch.tensor(2.0), torch.tensor(8.0), True, True,
                                                                False)

        # 验证fake_quantize被正确调用
        mock_fake_quantize.assert_called_once_with(x, torch.tensor(0.0), torch.tensor(0.0), 8, True)

        # 验证输出
        self.assertEqual(output.shape, x.shape)
        torch.testing.assert_close(output, expected_output)

    @patch('msmodelslim.quant.quantizer.activation.quantizer.fake_quantize')
    @patch('msmodelslim.quant.quantizer.activation.quantizer.linear_quantization_params')
    def test_symmetric_quantization(self, mock_linear_quantization_params, mock_fake_quantize):
        """测试对称量化（min和max关于0对称）"""
        # 设置对称的统计数据
        self.quantizer.strategy.min_val = torch.tensor(-5.0)
        self.quantizer.strategy.max_val = torch.tensor(5.0)

        # 设置mock的返回值
        x = torch.tensor([[-5.0, -2.5], [0.0, 2.5], [5.0, 0.0]])
        expected_output = torch.tensor([[-5.0, -2.5], [0.0, 2.5], [5.0, 0.0]])
        mock_fake_quantize.return_value = (expected_output, expected_output)
        mock_linear_quantization_params.return_value = (torch.tensor(0.0), torch.tensor(0.0))

        # 调用量化器
        output = self.quantizer(x)

        # 验证linear_quantization_params被正确调用
        mock_linear_quantization_params.assert_called_once_with(8, torch.tensor(-5.0), torch.tensor(5.0), True, True,
                                                                False)

        # 验证fake_quantize被正确调用
        mock_fake_quantize.assert_called_once_with(x, torch.tensor(0.0), torch.tensor(0.0), 8, True)

        # 验证输出
        self.assertEqual(output.shape, x.shape)
        torch.testing.assert_close(output, expected_output)

    @patch('msmodelslim.quant.quantizer.activation.quantizer.fake_quantize')
    @patch('msmodelslim.quant.quantizer.activation.quantizer.linear_quantization_params')
    def test_positive_only_quantization(self, mock_linear_quantization_params, mock_fake_quantize):
        """测试只有正值的量化"""
        # 设置只有正值的统计数据
        self.quantizer.strategy.min_val = torch.tensor(0.0)
        self.quantizer.strategy.max_val = torch.tensor(10.0)

        # 设置mock的返回值
        x = torch.tensor([[0.0, 2.5], [5.0, 7.5], [10.0, 0.0]])
        expected_output = torch.tensor([[0.0, 2.5], [5.0, 7.5], [10.0, 0.0]])
        mock_fake_quantize.return_value = (expected_output, expected_output)
        mock_linear_quantization_params.return_value = (torch.tensor(0.0), torch.tensor(0.0))

        # 调用量化器
        output = self.quantizer(x)

        # 验证linear_quantization_params被正确调用
        mock_linear_quantization_params.assert_called_once_with(8, torch.tensor(0.0), torch.tensor(10.0), True, True,
                                                                False)

        # 验证fake_quantize被正确调用
        mock_fake_quantize.assert_called_once_with(x, torch.tensor(0.0), torch.tensor(0.0), 8, True)

        # 验证输出
        self.assertEqual(output.shape, x.shape)
        torch.testing.assert_close(output, expected_output)


if __name__ == '__main__':
    unittest.main()

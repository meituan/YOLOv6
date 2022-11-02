import torch
from torch import nn
from yolov6.layers.common import RepBlock, RepVGGBlock, BottleRep, BepC3, SimConv, Transpose

# _QUANT=False
class RepPANNeck(nn.Module):
    """RepPANNeck Module
    EfficientRep is the default backbone of this model.
    RepPANNeck has the balance of feature fusion ability and hardware efficiency.
    """

    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        self.Rep_p4 = RepBlock(
            in_channels=channels_list[3] + channels_list[5],
            out_channels=channels_list[5],
            n=num_repeats[5],
            block=block
        )

        self.Rep_p3 = RepBlock(
            in_channels=channels_list[2] + channels_list[6],
            out_channels=channels_list[6],
            n=num_repeats[6],
            block=block
        )

        self.Rep_n3 = RepBlock(
            in_channels=channels_list[6] + channels_list[7],
            out_channels=channels_list[8],
            n=num_repeats[7],
            block=block
        )

        self.Rep_n4 = RepBlock(
            in_channels=channels_list[5] + channels_list[9],
            out_channels=channels_list[10],
            n=num_repeats[8],
            block=block
        )

        self.reduce_layer0 = SimConv(
            in_channels=channels_list[4],
            out_channels=channels_list[5],
            kernel_size=1,
            stride=1
        )

        self.upsample0 = Transpose(
            in_channels=channels_list[5],
            out_channels=channels_list[5],
        )

        self.reduce_layer1 = SimConv(
            in_channels=channels_list[5],
            out_channels=channels_list[6],
            kernel_size=1,
            stride=1
        )

        self.upsample1 = Transpose(
            in_channels=channels_list[6],
            out_channels=channels_list[6]
        )

        self.downsample2 = SimConv(
            in_channels=channels_list[6],
            out_channels=channels_list[7],
            kernel_size=3,
            stride=2
        )

        self.downsample1 = SimConv(
            in_channels=channels_list[8],
            out_channels=channels_list[9],
            kernel_size=3,
            stride=2
        )

    def upsample_enable_quant(self, num_bits, calib_method):
        print("Insert fakequant after upsample")
        # Insert fakequant after upsample op to build TensorRT engine
        from pytorch_quantization import nn as quant_nn
        from pytorch_quantization.tensor_quant import QuantDescriptor
        conv2d_input_default_desc = QuantDescriptor(num_bits=num_bits, calib_method=calib_method)
        self.upsample_feat0_quant = quant_nn.TensorQuantizer(conv2d_input_default_desc)
        self.upsample_feat1_quant = quant_nn.TensorQuantizer(conv2d_input_default_desc)
        # global _QUANT
        self._QUANT = True

    def forward(self, input):

        (x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        upsample_feat0 = self.upsample0(fpn_out0)
        if hasattr(self, '_QUANT') and self._QUANT is True:
            upsample_feat0 = self.upsample_feat0_quant(upsample_feat0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        upsample_feat1 = self.upsample1(fpn_out1)
        if hasattr(self, '_QUANT') and self._QUANT is True:
            upsample_feat1 = self.upsample_feat1_quant(upsample_feat1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        outputs = [pan_out2, pan_out1, pan_out0]

        return outputs


class CSPRepPANNeck(nn.Module):
    """
    CSPRepPANNeck module.
    """

    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=BottleRep,
        csp_e=float(1)/2
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        self.Rep_p4 = BepC3(
            in_channels=channels_list[3] + channels_list[5], # 512 + 256
            out_channels=channels_list[5], # 256
            n=num_repeats[5],
            e=csp_e,
            block=block
        )

        self.Rep_p3 = BepC3(
            in_channels=channels_list[2] + channels_list[6], # 256 + 128
            out_channels=channels_list[6], # 128
            n=num_repeats[6],
            e=csp_e,
            block=block
        )

        self.Rep_n3 = BepC3(
            in_channels=channels_list[6] + channels_list[7], # 128 + 128
            out_channels=channels_list[8], # 256
            n=num_repeats[7],
            e=csp_e,
            block=block
        )

        self.Rep_n4 = BepC3(
            in_channels=channels_list[5] + channels_list[9], # 256 + 256
            out_channels=channels_list[10], # 512
            n=num_repeats[8],
            e=csp_e,
            block=block
        )

        self.reduce_layer0 = SimConv(
            in_channels=channels_list[4], # 1024
            out_channels=channels_list[5], # 256
            kernel_size=1,
            stride=1
        )

        self.upsample0 = Transpose(
            in_channels=channels_list[5], # 256
            out_channels=channels_list[5], # 256
        )

        self.reduce_layer1 = SimConv(
            in_channels=channels_list[5], # 256
            out_channels=channels_list[6], # 128
            kernel_size=1,
            stride=1
        )

        self.upsample1 = Transpose(
            in_channels=channels_list[6], # 128
            out_channels=channels_list[6] # 128
        )

        self.downsample2 = SimConv(
            in_channels=channels_list[6], # 128
            out_channels=channels_list[7], # 128
            kernel_size=3,
            stride=2
        )

        self.downsample1 = SimConv(
            in_channels=channels_list[8], # 256
            out_channels=channels_list[9], # 256
            kernel_size=3,
            stride=2
        )

    def forward(self, input):

        (x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        upsample_feat1 = self.upsample1(fpn_out1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        outputs = [pan_out2, pan_out1, pan_out0]

        return outputs

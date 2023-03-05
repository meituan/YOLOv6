from pickle import FALSE
from torch import nn
from yolov6.layers.common import BottleRep, RepVGGBlock, RepBlock, BepC3, SimSPPF, SPPF, SimCSPSPPF, CSPSPPF, ConvWrapper


class EfficientRep(nn.Module):
    '''EfficientRep Backbone
    EfficientRep is handcrafted by hardware-aware neural network design.
    With rep-style struct, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    '''

    def __init__(
        self,
        in_channels=3,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock,
        fuse_P2=False,
        cspsppf=False
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None
        self.fuse_P2 = fuse_P2

        self.stem = block(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2
        )

        self.ERBlock_2 = nn.Sequential(
            block(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1],
                block=block,
            )
        )

        self.ERBlock_3 = nn.Sequential(
            block(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                block=block,
            )
        )

        self.ERBlock_4 = nn.Sequential(
            block(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                block=block,
            )
        )

        channel_merge_layer = SPPF if block == ConvWrapper else SimSPPF
        if cspsppf:
            channel_merge_layer = CSPSPPF if block == ConvWrapper else SimCSPSPPF

        self.ERBlock_5 = nn.Sequential(
            block(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2,
            ),
            RepBlock(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                block=block,
            ),
            channel_merge_layer(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                kernel_size=5
            )
        )

    def forward(self, x):

        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        if self.fuse_P2:
            outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)

        return tuple(outputs)


class EfficientRep6(nn.Module):
    '''EfficientRep+P6 Backbone
    EfficientRep is handcrafted by hardware-aware neural network design.
    With rep-style struct, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    '''

    def __init__(
        self,
        in_channels=3,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock,
        fuse_P2=False,
        cspsppf=False
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None
        self.fuse_P2 = fuse_P2

        self.stem = block(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2
        )

        self.ERBlock_2 = nn.Sequential(
            block(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1],
                block=block,
            )
        )

        self.ERBlock_3 = nn.Sequential(
            block(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                block=block,
            )
        )

        self.ERBlock_4 = nn.Sequential(
            block(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                block=block,
            )
        )

        self.ERBlock_5 = nn.Sequential(
            block(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2,
            ),
            RepBlock(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                block=block,
            )
        )

        channel_merge_layer = SimSPPF if not cspsppf else SimCSPSPPF

        self.ERBlock_6 = nn.Sequential(
            block(
                in_channels=channels_list[4],
                out_channels=channels_list[5],
                kernel_size=3,
                stride=2,
            ),
            RepBlock(
                in_channels=channels_list[5],
                out_channels=channels_list[5],
                n=num_repeats[5],
                block=block,
            ),
            channel_merge_layer(
                in_channels=channels_list[5],
                out_channels=channels_list[5],
                kernel_size=5
            )
        )

    def forward(self, x):

        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        if self.fuse_P2:
            outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)
        x = self.ERBlock_6(x)
        outputs.append(x)

        return tuple(outputs)


class CSPBepBackbone(nn.Module):
    """
    CSPBepBackbone module.
    """

    def __init__(
        self,
        in_channels=3,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock,
        csp_e=float(1)/2,
        fuse_P2=False,
        cspsppf=False
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None
        self.fuse_P2 = fuse_P2

        self.stem = block(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2
        )

        self.ERBlock_2 = nn.Sequential(
            block(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2
            ),
            BepC3(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1],
                e=csp_e,
                block=block,
            )
        )

        self.ERBlock_3 = nn.Sequential(
            block(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            BepC3(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                e=csp_e,
                block=block,
            )
        )

        self.ERBlock_4 = nn.Sequential(
            block(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            BepC3(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                e=csp_e,
                block=block,
            )
        )

        channel_merge_layer = SPPF if block == ConvWrapper else SimSPPF
        if cspsppf:
            channel_merge_layer = CSPSPPF if block == ConvWrapper else SimCSPSPPF

        self.ERBlock_5 = nn.Sequential(
            block(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2,
            ),
            BepC3(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                e=csp_e,
                block=block,
            ),
            channel_merge_layer(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                kernel_size=5
            )
        )

    def forward(self, x):

        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        if self.fuse_P2:
            outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)

        return tuple(outputs)


class CSPBepBackbone_P6(nn.Module):
    """
    CSPBepBackbone+P6 module.
    """

    def __init__(
        self,
        in_channels=3,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock,
        csp_e=float(1)/2,
        fuse_P2=False,
        cspsppf=False
    ):
        super().__init__()
        assert channels_list is not None
        assert num_repeats is not None
        self.fuse_P2 = fuse_P2

        self.stem = block(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2
        )

        self.ERBlock_2 = nn.Sequential(
            block(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2
            ),
            BepC3(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1],
                e=csp_e,
                block=block,
            )
        )

        self.ERBlock_3 = nn.Sequential(
            block(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            BepC3(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                e=csp_e,
                block=block,
            )
        )

        self.ERBlock_4 = nn.Sequential(
            block(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            BepC3(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                e=csp_e,
                block=block,
            )
        )

        channel_merge_layer = SPPF if block == ConvWrapper else SimSPPF
        if cspsppf:
            channel_merge_layer = CSPSPPF if block == ConvWrapper else SimCSPSPPF

        self.ERBlock_5 = nn.Sequential(
            block(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2,
            ),
            BepC3(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                e=csp_e,
                block=block,
            ),
        )
        self.ERBlock_6 = nn.Sequential(
            block(
                in_channels=channels_list[4],
                out_channels=channels_list[5],
                kernel_size=3,
                stride=2,
            ),
            BepC3(
                in_channels=channels_list[5],
                out_channels=channels_list[5],
                n=num_repeats[5],
                e=csp_e,
                block=block,
            ),
            channel_merge_layer(
                in_channels=channels_list[5],
                out_channels=channels_list[5],
                kernel_size=5
            )
        )

    def forward(self, x):

        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)
        x = self.ERBlock_6(x)
        outputs.append(x)

        return tuple(outputs)

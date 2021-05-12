import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule


class PointNet2SemSegSSG(nn.Module):
    r"""Parameters
        ----------
        in_channels: int
            Number of input channels of per point feature except point coordinate.
            For point cloud of size (N, xyz; RGB; normal), in_channel=6(RGB; normal)

        out_channels: int
            Number of output channel of the network, usually the number of class 
            of the dataset.

        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature

        Note: modify npoint and radii to fit different data.
    """
    def __init__(self, in_channels, out_channels, use_xyz=True, as_backbone=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_xyz = use_xyz
        self.as_backbone = as_backbone
        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[self.in_channels, 32, 32, 64],
                use_xyz=self.use_xyz
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=self.use_xyz
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=self.use_xyz
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=self.use_xyz
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + self.in_channels, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        if not self.as_backbone:
            self.fc_layer = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Conv1d(128, self.out_channels, kernel_size=1),
            )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + in_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
                returns: if as_backbone, out_channel=128.
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        features = l_features[0]

        if not self.as_backbone:
            network_output = self.fc_layer(features)
        else:
            network_output = features

        return network_output
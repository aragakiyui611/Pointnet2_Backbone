from typing import Sequence
import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG


class PointNet2ClassificationMSG(nn.Module):
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
    """
    def __init__(self, in_channels, out_channels, use_xyz=True):
        super().__init__()
        self.in_channels = in_channels
        self.use_xyz = use_xyz
        self.out_channels = out_channels
        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.1, 0.2, 0.4],
                nsamples=[16, 32, 128],
                mlps=[[self.in_channels, 32, 32, 64],
                      [self.in_channels, 64, 64, 128],
                      [self.in_channels, 64, 96, 128]],
                use_xyz=self.use_xyz
            )
        )

        input_channels = 64 + 128 + 128
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[
                    [input_channels, 64, 64, 128],
                    [input_channels, 128, 128, 256],
                    [input_channels, 128, 128, 256],
                ],
                use_xyz=self.use_xyz
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[128 + 256 + 256, 256, 512, 1024],
                use_xyz=self.use_xyz
            )
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.out_channels)
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
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        features = features.squeeze(-1)

        network_output = self.fc_layer(features)
        
        return network_output
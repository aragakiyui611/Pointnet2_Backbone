from models.pointnet2_ssg_sem import PointNet2SemSegSSG as modelsem
from models.pointnet2_ssg_cls import PointNet2ClassificationSSG as modelcls
from models.pointnet2_msg_cls import PointNet2ClassificationMSG as modelmsgcls
from models.pointnet2_msg_sem import PointNet2SemSegMSG as modelmsgsem
import torch
# for network debug

pc = torch.ones(10,100,9).cuda()*1.0
net1 = modelcls(6,40,True).cuda()
net2 = modelsem(6,40,True).cuda()
net3 = modelmsgcls(6,40,True).cuda()
net4 = modelmsgsem(in_channels=6,out_channels=40,use_xyz=True, as_backbone=True).cuda()

z =net1(pc)
print(z.shape)

z =net2(pc)
print(z.shape)

z =net3(pc)
print(z.shape)

print(net4)
z =net4(pc)
print(z.shape)
from torch import nn
from torch import cat
import torch
import torch.nn.functional as F
import time
import torch.nn.init as init
###MFI block
class Attention_code_block(nn.Module):
    def __init__(self, in_channel):
        super(Attention_code_block, self).__init__()
        
        self.attention1 = nn.Sequential(
            nn.Conv3d(in_channel*2, in_channel,kernel_size=1, stride=1),
            nn.InstanceNorm3d(in_channel),
            nn.LeakyReLU(),
            nn.Conv3d(in_channel, in_channel, kernel_size=1, stride=1),
            nn.InstanceNorm3d(in_channel),
            nn.LeakyReLU()
        )
        
        self.attention2 =  nn.Sequential(
            nn.Conv3d(in_channel*2, in_channel,kernel_size=1, stride=1),
            nn.InstanceNorm3d(in_channel),
            nn.LeakyReLU(),
            nn.Conv3d(in_channel, in_channel, kernel_size=1, stride=1),
            nn.InstanceNorm3d(in_channel),
            nn.LeakyReLU()
        )
        
        self.attention3 =  nn.Sequential(
            nn.Conv3d(in_channel*2, in_channel,kernel_size=1, stride=1),
            nn.InstanceNorm3d(in_channel),
            nn.LeakyReLU(),
            nn.Conv3d(in_channel, in_channel, kernel_size=1, stride=1),
            nn.InstanceNorm3d(in_channel),
            nn.LeakyReLU()
        )
        
        self.attention4 =  nn.Sequential(
            nn.Conv3d(in_channel*2, in_channel,kernel_size=1, stride=1),
            nn.InstanceNorm3d(in_channel),
            nn.LeakyReLU(),
            nn.Conv3d(in_channel, in_channel, kernel_size=1, stride=1),
            nn.InstanceNorm3d(in_channel),
            nn.LeakyReLU()
        )
        self.attention5 =  nn.Sequential(
            nn.Conv3d(in_channel*2, in_channel,kernel_size=1, stride=1),
            nn.InstanceNorm3d(in_channel),
            nn.LeakyReLU(),
            nn.Conv3d(in_channel, in_channel, kernel_size=1, stride=1),
            nn.InstanceNorm3d(in_channel),
            nn.LeakyReLU()
        )
        self.attention6 =  nn.Sequential(
            nn.Conv3d(in_channel*2, in_channel,kernel_size=1, stride=1),
            nn.InstanceNorm3d(in_channel),
            nn.LeakyReLU(),
            nn.Conv3d(in_channel, in_channel, kernel_size=1, stride=1),
            nn.InstanceNorm3d(in_channel),
            nn.LeakyReLU()
        )
        self.attention7 =  nn.Sequential(
            nn.Conv3d(in_channel*2, in_channel,kernel_size=1, stride=1),
            nn.InstanceNorm3d(in_channel),
            nn.LeakyReLU(),
            nn.Conv3d(in_channel, in_channel, kernel_size=1, stride=1),
            nn.InstanceNorm3d(in_channel),
            nn.LeakyReLU()
        )
    def forward(self, fusion_map, x1, x2, x3, x4,x5,x6,x7): 
        
        # Calculate/Compute每个 attention
        # print(fusion_map.shape, x1.shape)
        attention1 = self.attention1(torch.cat((fusion_map, x1), 1))
        # print(attention1.shape)
        attention2 = self.attention2(torch.cat((fusion_map, x2), 1))
        attention3 = self.attention3(torch.cat((fusion_map, x3), 1))
        attention4 = self.attention4(torch.cat((fusion_map, x4), 1))
        attention5 = self.attention5(torch.cat((fusion_map, x5), 1))
        attention6 = self.attention6(torch.cat((fusion_map, x6), 1))
        attention7 = self.attention7(torch.cat((fusion_map, x7), 1))

        attention_stack = torch.stack((attention1, attention2, attention3, attention4, attention5, attention6, attention7), dim=1)
        # print(attention_stack.shape)
        # 对拼接后的张量沿着第1维应用 softmax (每个 attention 的贡献度)
        attention_softmax = F.softmax(attention_stack, dim=1)  # 在新dimension dim=1 上应用 softmax
        # print( attention_softmax.shape)
        # 将 softmax 结果分离成单独的 attention
        attention1, attention2, attention3, attention4, attention5, attention6, attention7= torch.chunk(attention_softmax, 7, dim=1)
       
        return attention1.squeeze(dim=1), attention2.squeeze(dim=1), attention3.squeeze(dim=1), attention4.squeeze(dim=1),\
        attention5.squeeze(dim=1), attention6.squeeze(dim=1), attention7.squeeze(dim=1)

###Unet Encoder block
class EnBlock(nn.Module):
    def __init__(self,in_c):
        super(EnBlock,self).__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(in_c//8,in_c),
            nn.ReLU(True),
            nn.Conv3d(in_c, in_c,3,padding=1),
            nn.GroupNorm(in_c//8,in_c),
            nn.ReLU(True),
            nn.Conv3d(in_c, in_c,3,padding=1),
        )
    def forward(self, x):
        res = x
        out = self.conv(x)
        return (out+res)

class EnDown(nn.Module):
    def __init__(self,in_c,ou_c):
        super(EnDown,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_c,ou_c,kernel_size=3,stride=2,padding=1)
        )
    def forward(self, x):
        out = self.conv(x)
        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.init_conv = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3,
                                   padding=1)  # 64,128,128,128
        self.en1 = EnBlock(32)
        self.ed1 = EnDown(32, 64)
        self.en2 = EnBlock(64)
        self.ed2 = EnDown(64, 128)
        self.en3 = EnBlock(128)
        self.ed3 = EnDown(128, 256)
        self.en4 = EnBlock(256)
    def forward(self, x):
        x = self.init_conv(x)
        c1 = self.en1(x)
        p1 = self.ed1(c1)
        c2 = self.en2(p1)
        p2 = self.ed2(c2)
        c3 = self.en3(p2)
        p3 = self.ed3(c3)
        c4 = self.en4(p3)
        return c1,c2,c3,c4

###Unet Dncoder block

class DnUp(nn.Module):
    def __init__(self, in_c, ou_c):
        super(DnUp, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_c,in_c,1),
            nn.ConvTranspose3d(in_c,ou_c,2,2)
        )

    def forward(self, x,en_x):
        out = self.conv(x)


        return (out+en_x)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv3d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class no_share_unet(nn.Module):
    def __init__(self,in_channel,out_channel,diff=False,deepSupvision=False):
        super(no_share_unet, self).__init__()
        class_nums = out_channel
        self.is_diff = diff
        self.deepSupvision = deepSupvision


        self.diff1 = Attention_code_block(64)
        self.diff2 = Attention_code_block(128)
        self.diff3 = Attention_code_block(256)
        self.diff4 = Attention_code_block(128)
        self.diff5 = Attention_code_block(64)


        self.init_conv_1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3,
                                   padding=1)  # 64,128,128,128
        self.en1_1 = EnBlock(32)
        self.ed1_1 = EnDown(32, 64)
        self.en2_1 = EnBlock(64)
        self.ed2_1 = EnDown(64, 128)
        self.en3_1 = EnBlock(128)
        self.ed3_1 = EnDown(128, 256)
        self.en4_1 = EnBlock(256)

        self.init_conv_2 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3,
                                     padding=1)  # 64,128,128,128
        self.en1_2 = EnBlock(32)
        self.ed1_2 = EnDown(32, 64)
        self.en2_2 = EnBlock(64)
        self.ed2_2 = EnDown(64, 128)
        self.en3_2 = EnBlock(128)
        self.ed3_2 = EnDown(128, 256)
        self.en4_2 = EnBlock(256)

        self.init_conv_3 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3,
                                     padding=1)  # 64,128,128,128
        self.en1_3 = EnBlock(32)
        self.ed1_3 = EnDown(32, 64)
        self.en2_3 = EnBlock(64)
        self.ed2_3 = EnDown(64, 128)
        self.en3_3 = EnBlock(128)
        self.ed3_3 = EnDown(128, 256)
        self.en4_3 = EnBlock(256)

        self.init_conv_4 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3,
                                     padding=1)  # 64,128,128,128
        self.en1_4 = EnBlock(32)
        self.ed1_4 = EnDown(32, 64)
        self.en2_4 = EnBlock(64)
        self.ed2_4 = EnDown(64, 128)
        self.en3_4 = EnBlock(128)
        self.ed3_4 = EnDown(128, 256)
        self.en4_4 = EnBlock(256)
        
        
        self.init_conv_5 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # 64,128,128,128
        self.en1_5 = EnBlock(32)
        self.ed1_5 = EnDown(32, 64)
        self.en2_5 = EnBlock(64)
        self.ed2_5 = EnDown(64, 128)
        self.en3_5 = EnBlock(128)
        self.ed3_5 = EnDown(128, 256)
        self.en4_5 = EnBlock(256)

        self.init_conv_6 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # 64,128,128,128
        self.en1_6 = EnBlock(32)
        self.ed1_6 = EnDown(32, 64)
        self.en2_6 = EnBlock(64)
        self.ed2_6 = EnDown(64, 128)
        self.en3_6 = EnBlock(128)
        self.ed3_6 = EnDown(128, 256)
        self.en4_6 = EnBlock(256)

        self.init_conv_7 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # 64,128,128,128
        self.en1_7 = EnBlock(32)
        self.ed1_7 = EnDown(32, 64)
        self.en2_7 = EnBlock(64)
        self.ed2_7 = EnDown(64, 128)
        self.en3_7 = EnBlock(128)
        self.ed3_7 = EnDown(128, 256)
        self.en4_7 = EnBlock(256)


        self.ud1_1 = DnUp(256, 128)  #
        self.un1_1 = EnBlock(128)
        self.ud2_1 = DnUp(128, 64)
        self.un2_1 = EnBlock(64)
        self.ud3_1 = DnUp(64, 32)
        self.un3_1 = EnBlock(32)

        self.ud1_2 = DnUp(256, 128)  #
        self.un1_2 = EnBlock(128)
        self.ud2_2 = DnUp(128, 64)
        self.un2_2 = EnBlock(64)
        self.ud3_2 = DnUp(64, 32)
        self.un3_2 = EnBlock(32)

        self.ud1_3 = DnUp(256, 128)  #
        self.un1_3 = EnBlock(128)
        self.ud2_3 = DnUp(128, 64)
        self.un2_3 = EnBlock(64)
        self.ud3_3 = DnUp(64, 32)
        self.un3_3 = EnBlock(32)

        self.ud1_4 = DnUp(256, 128)  #
        self.un1_4 = EnBlock(128)
        self.ud2_4 = DnUp(128, 64)
        self.un2_4 = EnBlock(64)
        self.ud3_4 = DnUp(64, 32)
        self.un3_4 = EnBlock(32)
        
        self.ud1_5 = DnUp(256, 128)  #
        self.un1_5 = EnBlock(128)
        self.ud2_5 = DnUp(128, 64)
        self.un2_5 = EnBlock(64)
        self.ud3_5 = DnUp(64, 32)
        self.un3_5 = EnBlock(32)

        self.ud1_6 = DnUp(256, 128)  #
        self.un1_6 = EnBlock(128)
        self.ud2_6 = DnUp(128, 64)
        self.un2_6 = EnBlock(64)
        self.ud3_6 = DnUp(64, 32)
        self.un3_6 = EnBlock(32)

        self.ud1_7 = DnUp(256, 128)  #
        self.un1_7 = EnBlock(128)
        self.ud2_7 = DnUp(128, 64)
        self.un2_7 = EnBlock(64)
        self.ud3_7 = DnUp(64, 32)
        self.un3_7 = EnBlock(32)

        self.out_conv1 = nn.Conv3d(32,class_nums,1)
        self.out_conv2 = nn.Conv3d(32, class_nums, 1)
        self.out_conv3 = nn.Conv3d(32, class_nums, 1)
        self.out_conv4 = nn.Conv3d(32, class_nums, 1)
        self.out_conv5 = nn.Conv3d(32, class_nums, 1)
        self.out_conv6 = nn.Conv3d(32, class_nums, 1)
        self.out_conv7 = nn.Conv3d(32, class_nums, 1)
        
        
        self.Attention_1 = nn.Sequential(
            nn.Conv3d(in_channel*2,in_channel,kernel_size=3,padding=1),
            nn.InstanceNorm3d(in_channel),
            nn.LeakyReLU(),
            nn.Conv3d(in_channel,in_channel,kernel_size=1,padding=1),
            nn.InstanceNorm3d(in_channel),
            nn.LeakyReLU(),
            nn.Sigmoid()
        )

        self.cat_out_conv = nn.Sequential(
            nn.Conv3d(7*class_nums,class_nums,1)
        )

        if self.deepSupvision:
            self.stage1Out = nn.Sequential(
                nn.Conv3d(128,class_nums,kernel_size=3,padding=1),
                nn.Upsample(scale_factor=4)

            )
            self.stage2Out = nn.Sequential(
                nn.Conv3d(64,class_nums,kernel_size=3,padding=1),
                nn.Upsample(scale_factor=2),
            )

        self.fc1 = nn.Linear(7,7)
        self.fc2 = nn.Linear(7,7)
        self.fc3 = nn.Linear(7,7)
        self.fc4 = nn.Linear(7,7)
        self.fc5 = nn.Linear(7,7)
    def weight_init(self):
        initializer = kaiming_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, x1,x2,x3,x4,x5,x6,x7,mask_code):
        x1 = self.init_conv_1(x1)
        x2 = self.init_conv_2(x2)
        x3 = self.init_conv_3(x3)
        x4 = self.init_conv_4(x4)
        x5 = self.init_conv_3(x5)
        x6 = self.init_conv_4(x6)
        x7 = self.init_conv_3(x7)
       


        c1_1 = self.en1_1(x1)
        c1_2 = self.en1_2(x2)
        c1_3 = self.en1_3(x3)
        c1_4 = self.en1_4(x4)
        c1_5 = self.en1_5(x5)
        c1_6 = self.en1_6(x6)
        c1_7 = self.en1_7(x7)
        

        p1_1 = self.ed1_1(c1_1)
        p1_2 = self.ed1_2(c1_2)
        p1_3 = self.ed1_3(c1_3)
        p1_4 = self.ed1_4(c1_4)
        p1_5 = self.ed1_5(c1_5)
        p1_6 = self.ed1_6(c1_6)
        p1_7 = self.ed1_7(c1_7)
        
        mask_code = mask_code.float()
        weight_1 = mask_code*torch.softmax(self.fc1(mask_code),dim=-1)
        weight_1 = weight_1/weight_1.sum(dim=1, keepdim=True)
      
        adaptive_aggrevate_feature_1 = (
            p1_1 * weight_1[:, 0:1, None, None, None] +
            p1_2 * weight_1[:, 1:2, None, None, None] +
            p1_3 * weight_1[:, 2:3, None, None, None] +
            p1_4 * weight_1[:, 3:4, None, None, None] +
            p1_5 * weight_1[:, 4:5, None, None, None] +
            p1_6 * weight_1[:, 5:6, None, None, None] +
            p1_7 * weight_1[:, 6:7, None, None, None])
        
        if self.is_diff:     
            attention1_1, attention1_2, attention1_3, attention1_4, attention1_5, attention1_6, attention1_7 = self.diff1(adaptive_aggrevate_feature_1,p1_1, p1_2, p1_3, p1_4,p1_5,p1_6,p1_7)
            
        p1_1, p1_2, p1_3, p1_4,p1_5,p1_6,p1_7 = p1_1*attention1_1, p1_2*attention1_2, p1_3*attention1_3, p1_4*attention1_4, p1_5*attention1_5, p1_6*attention1_6, p1_7*attention1_7

        c2_1 = self.en2_1(p1_1)
        c2_2 = self.en2_2(p1_2)
        c2_3 = self.en2_3(p1_3)
        c2_4 = self.en2_4(p1_4)
        c2_5 = self.en2_5(p1_5)
        c2_6 = self.en2_6(p1_6)
        c2_7 = self.en2_7(p1_7)


        p2_1 = self.ed2_1(c2_1)
        p2_2 = self.ed2_2(c2_2)
        p2_3 = self.ed2_3(c2_3)
        p2_4 = self.ed2_4(c2_4)
        p2_5 = self.ed2_2(c2_5)
        p2_6 = self.ed2_6(c2_6)
        p2_7 = self.ed2_7(c2_7)
        
        weight_2 = mask_code*torch.softmax(self.fc2(mask_code),dim=-1)
        weight_2 = weight_2/weight_2.sum(dim=1, keepdim=True)
        adaptive_aggrevate_feature_2 = (
            p2_1 * weight_2[:, 0:1, None, None, None] +
            p2_2 * weight_2[:, 1:2, None, None, None] +
            p2_3 * weight_2[:, 2:3, None, None, None] +
            p2_4 * weight_2[:, 3:4, None, None, None] +
            p2_5 * weight_2[:, 4:5, None, None, None] +
            p2_6 * weight_2[:, 5:6, None, None, None] +
            p2_7 * weight_2[:, 6:7, None, None, None])
        
        if self.is_diff:
            attention2_1, attention2_2, attention2_3, attention2_4, attention2_5, attention2_6, attention2_7 = self.diff2(adaptive_aggrevate_feature_2,p2_1, p2_2, p2_3, p2_4,p2_5, p2_6, p2_7)
            
        # print(attention1_1.shape,p1_1.shape)
        p2_1, p2_2, p2_3, p2_4,p2_5, p2_6, p2_7 = p2_1*attention2_1, p2_2*attention2_2, p2_3*attention2_3, p2_4*attention2_4,p2_5*attention2_5, p2_6*attention2_6, p2_7*attention2_7

        c3_1 = self.en3_1(p2_1)
        c3_2 = self.en3_2(p2_2)
        c3_3 = self.en3_3(p2_3)
        c3_4 = self.en3_4(p2_4)
        c3_5 = self.en3_5(p2_5)
        c3_6 = self.en3_6(p2_6)
        c3_7 = self.en3_7(p2_7)


        p3_1 = self.ed3_1(c3_1)
        p3_2 = self.ed3_2(c3_2)
        p3_3 = self.ed3_3(c3_3)
        p3_4 = self.ed3_4(c3_4)
        p3_5 = self.ed3_5(c3_5)
        p3_6 = self.ed3_6(c3_6)
        p3_7 = self.ed3_7(c3_7)
        
        weight_3 = mask_code*torch.softmax(self.fc3(mask_code),dim=-1)
        weight_3 = weight_3/weight_3.sum(dim=1, keepdim=True)
        adaptive_aggrevate_feature_3 = (
            p3_1 * weight_3[:, 0:1, None, None, None] +
            p3_2 * weight_3[:, 1:2, None, None, None] +
            p3_3 * weight_3[:, 2:3, None, None, None] +
            p3_4 * weight_3[:, 3:4, None, None, None] +
            p3_5 * weight_3[:, 4:5, None, None, None] +
            p3_6 * weight_3[:, 5:6, None, None, None] +
            p3_7 * weight_3[:, 6:7, None, None, None])
        if self.is_diff:
           attention3_1, attention3_2, attention3_3, attention3_4,attention3_5, attention3_6, attention3_7 = self.diff3(adaptive_aggrevate_feature_3,p3_1, p3_2, p3_3, p3_4,p3_5, p3_6, p3_7)
            
        
        p3_1, p3_2, p3_3, p3_4,p3_5, p3_6, p3_7 = p3_1*attention3_1, p3_2*attention3_2, p3_3*attention3_3, p3_4*attention3_4,p3_5*attention3_5, p3_6*attention3_6, p3_7*attention3_7
        
        c4_1 = self.en4_1(p3_1)
        c4_2 = self.en4_2(p3_2)
        c4_3 = self.en4_3(p3_3)
        c4_4 = self.en4_4(p3_4)
        c4_5 = self.en4_5(p3_5)
        c4_6 = self.en4_6(p3_6)
        c4_7 = self.en4_7(p3_7)

        up5_1 = self.ud1_1(c4_1, c3_1)
        up5_2 = self.ud1_2(c4_2, c3_2)
        up5_3 = self.ud1_3(c4_3, c3_3)
        up5_4 = self.ud1_4(c4_4, c3_4)
        
        up5_5 = self.ud1_5(c4_5, c3_5)
        up5_6 = self.ud1_6(c4_6, c3_6)
        up5_7 = self.ud1_7(c4_7, c3_7)
        
        if self.deepSupvision:
            stage1Out1 = self.stage1Out(up5_1)
            stage1Out2 = self.stage1Out(up5_2)
            stage1Out3 = self.stage1Out(up5_3)
            stage1Out4 = self.stage1Out(up5_4)
            stage1Out5 = self.stage1Out(up5_5)
            stage1Out6 = self.stage1Out(up5_6)
            stage1Out7 = self.stage1Out(up5_7)


        un5_1 = self.un1_1(up5_1)
        un5_2 = self.un1_2(up5_2)
        un5_3 = self.un1_3(up5_3)
        un5_4 = self.un1_4(up5_4)
        un5_5 = self.un1_5(up5_5)
        un5_6 = self.un1_6(up5_6)
        un5_7 = self.un1_7(up5_7)
        
        weight_4 = mask_code*torch.softmax(self.fc4(mask_code),dim=-1)
        weight_4 = weight_4/weight_4.sum(dim=1, keepdim=True)
        
        adaptive_aggrevate_feature_4 = (
            un5_1 * weight_4[:, 0:1, None, None, None] +
            un5_2 * weight_4[:, 1:2, None, None, None] +
            un5_3 * weight_4[:, 2:3, None, None, None] +
            un5_4 * weight_4[:, 3:4, None, None, None] +
            un5_5 * weight_4[:, 4:5, None, None, None] +
            un5_6 * weight_4[:, 5:6, None, None, None] +
            un5_7 * weight_4[:, 6:7, None, None, None] )

        if self.is_diff:
            attention4_1, attention4_2, attention4_3, attention4_4,attention4_5, attention4_6, attention4_7 = self.diff4(adaptive_aggrevate_feature_4,un5_1, un5_2, un5_3, un5_4,un5_5, un5_6, un5_7)

        up5_1, up5_2, up5_3, up5_4,up5_5, up5_6, up5_7 = up5_1*attention4_1, up5_2*attention4_2, up5_3*attention4_3, up5_4*attention4_4,up5_5*attention4_5, up5_6*attention4_6, up5_7*attention4_7
        
        up6_1 = self.ud2_1(un5_1, c2_1)
        up6_2 = self.ud2_2(un5_2, c2_2)
        up6_3 = self.ud2_3(un5_3, c2_3)
        up6_4 = self.ud2_4(un5_4, c2_4)
        up6_5 = self.ud2_5(un5_5, c2_5)
        up6_6 = self.ud2_6(un5_6, c2_6)
        up6_7 = self.ud2_7(un5_7, c2_7)
        if self.deepSupvision:
            stage2Out1 = self.stage2Out(up6_1)
            stage2Out2 = self.stage2Out(up6_2)
            stage2Out3 = self.stage2Out(up6_3)
            stage2Out4 = self.stage2Out(up6_4)
            stage2Out5 = self.stage2Out(up6_5)
            stage2Out6 = self.stage2Out(up6_6)
            stage2Out7 = self.stage2Out(up6_7)

        un6_1 = self.un2_1(up6_1)
        un6_2 = self.un2_2(up6_2)
        un6_3 = self.un2_3(up6_3)
        un6_4 = self.un2_4(up6_4)
        un6_5 = self.un2_5(up6_5)
        un6_6 = self.un2_6(up6_6)
        un6_7 = self.un2_7(up6_7)
        
        
        weight_5 = mask_code*torch.softmax(self.fc5(mask_code),dim=-1)
        weight_5 = weight_5/weight_5.sum(dim=1, keepdim=True)
        adaptive_aggrevate_feature_5 = (
            un6_1 * weight_5[:, 0:1, None, None, None] +
            un6_2 * weight_5[:, 1:2, None, None, None] +
            un6_3 * weight_5[:, 2:3, None, None, None] +
            un6_4 * weight_5[:, 3:4, None, None, None] +
            un6_5 * weight_5[:, 4:5, None, None, None] +
            un6_6 * weight_5[:, 5:6, None, None, None] +
            un6_7 * weight_5[:, 6:7, None, None, None])
        if self.is_diff:
            attention5_1, attention5_2, attention5_3, attention5_4,attention5_5, attention5_6, attention5_7 = self.diff5(adaptive_aggrevate_feature_5,un6_1, un6_2, un6_3, un6_4,un6_5, un6_6, un6_7)

        un6_1, un6_2, un6_3, un6_4,un6_5, un6_6, un6_7 = un6_1*attention5_1, un6_2*attention5_2, un6_3*attention5_3, un6_4*attention5_4,un6_5*attention5_5, un6_6*attention5_6, un6_7*attention5_7
        
        up7_1 = self.ud3_1(un6_1, c1_1)
        up7_2 = self.ud3_2(un6_2, c1_2)
        up7_3 = self.ud3_3(un6_3, c1_3)
        up7_4 = self.ud3_4(un6_4, c1_4)
        up7_5 = self.ud3_5(un6_5, c1_5)
        up7_6 = self.ud3_6(un6_6, c1_6)
        up7_7 = self.ud3_7(un6_7, c1_7)

        un7_1 = self.un3_1(up7_1)
        un7_2 = self.un3_2(up7_2)
        un7_3 = self.un3_3(up7_3)
        un7_4 = self.un3_4(up7_4)
        un7_5 = self.un3_5(up7_5)
        un7_6 = self.un3_6(up7_6)
        un7_7 = self.un3_7(up7_7)

        out1 = self.out_conv1(un7_1)
        out2 = self.out_conv2(un7_2)
        out3 = self.out_conv3(un7_3)
        out4 = self.out_conv4(un7_4)
        out5 = self.out_conv5(un7_5)
        out6 = self.out_conv6(un7_6)
        out7 = self.out_conv7(un7_7)

        cat_out = self.cat_out_conv(torch.cat((out1,out2 ,out3 , out4, out5 ,out6, out7),dim=1))

        if not self.deepSupvision:
            return out1, out2,out3 ,out4,out5, out6, out7,cat_out
        else:
            return stage1Out1,stage1Out2,stage1Out3,stage1Out4,stage1Out5,stage1Out6,stage1Out7,\
            stage2Out1,stage2Out2,stage2Out3,stage2Out4,stage2Out5,stage2Out6,stage2Out7,out1,out2, out3, out4,out5, out6, out7,cat_out

if __name__ == '__main__':
    # Set/Setup第7张GPU
    torch.cuda.set_device(7)

    # Createinput张量并移动到第7张GPU
    x1 = torch.rand(1, 1, 128, 128, 128).cuda()
    x2 = torch.rand(1, 1, 128, 128, 128).cuda()
    x3 = torch.rand(1, 1, 128, 128, 128).cuda()
    x4 = torch.rand(1, 1, 128, 128, 128).cuda()
    x5 = torch.rand(1, 1, 128, 128, 128).cuda()
    x6 = torch.rand(1, 1, 128, 128, 128).cuda()
    x7 = torch.rand(1, 1, 128, 128, 128).cuda()
    mask = torch.tensor([[1, 1, 0, 0, 0, 0, 0]]).cuda()

    # Create模型并移动到第7张GPU
    model = no_share_unet(in_channel=1, out_channel=3, diff=True, deepSupvision=True).cuda()

    # 运行模型并Printoutputshape
    res = model(x1, x2, x3, x4, x5, x6, x7, mask)
    print(res)


# from torch import nn
# from torch import cat
# import torch
# import torch.nn.functional as F
# import time
# import torch.nn.init as init

# ###MFI block
# class fianl_diff_code_block(nn.Module):
#     def __init__(self,in_channel):
#         super(fianl_diff_code_block,self).__init__()
#         self.Relation1 = nn.Sequential(
#             nn.Linear(in_channel * 4, in_channel*2),
#             nn.LeakyReLU(),
#             nn.Linear(in_channel*2, in_channel)
#             # nn.LeakyReLU(),
#         )
#         self.Relation2 = nn.Sequential(
#             nn.Linear(in_channel * 4, in_channel*2),
#             nn.LeakyReLU(),
#             nn.Linear(in_channel*2, in_channel)
#             # nn.LeakyReLU(),
#         )
#         self.Relation3 = nn.Sequential(
#             nn.Linear(in_channel * 4, in_channel*2),
#             nn.LeakyReLU(),
#             nn.Linear(in_channel*2, in_channel)
#             # nn.LeakyReLU(),
#         )
#         self.Relation4 = nn.Sequential(
#             nn.Linear(in_channel * 4, in_channel*2),
#             nn.LeakyReLU(),
#             nn.Linear(in_channel*2, in_channel)
#             # nn.LeakyReLU(),
#         )
#     def forward(self, x1,x2,x3,x4,mod_code):     ####mod_code: b*4
#         b,c,h,w,l = x1.shape
#         X_ori = torch.cat((x1.unsqueeze(1),x2.unsqueeze(1),x3.unsqueeze(1),x4.unsqueeze(1)),1)  ###(b*4*c*h*w*l)
#         X = torch.mean(X_ori.view(b,4,c,h*w*l),-1) ##b*4*c
#         X1 = X.unsqueeze(1).repeat(1,4,1,1)  ##b*4*4*c
#         X2 = X.unsqueeze(2).repeat(1,1,4,1)
#         X_R = torch.cat((X1, X2),-1)   ###b*4*4*2c

#         mod_code = mod_code.unsqueeze(-1).repeat(1, 1, 2 * c)
#         X_R_1, X_R_2, X_R_3, X_R_4 = self.Relation1(torch.cat((X_R[:,0,:,:],mod_code),dim=-1)),self.Relation1(torch.cat((X_R[:,1,:,:],mod_code),dim=-1)),\
#                                      self.Relation1(torch.cat((X_R[:,2,:,:],mod_code),dim=-1)),self.Relation1(torch.cat((X_R[:,3,:,:],mod_code),dim=-1)),


#         X_R_1,X_R_2,X_R_3,X_R_4 = F.softmax(X_R_1, 1),F.softmax(X_R_2, 1),F.softmax(X_R_3, 1),F.softmax(X_R_4, 1)
#         X_1_out = torch.matmul(X_ori.view(b,4,c,h*w*l).permute(0,2,3,1),X_R_1.permute(0,2,1).unsqueeze(-1)).squeeze(-1)  ##b*c*(h*w*l)*4 and b*c*4*1 -> b*c*(h*w*l)*1
#         X_2_out = torch.matmul(X_ori.view(b, 4, c, h * w * l).permute(0, 2, 3, 1),
#                                X_R_2.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1)
#         X_3_out = torch.matmul(X_ori.view(b, 4, c, h * w * l).permute(0, 2, 3, 1),
#                                X_R_3.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1)
#         X_4_out = torch.matmul(X_ori.view(b, 4, c, h * w * l).permute(0, 2, 3, 1),
#                                X_R_4.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1)

#         X_1_out,X_2_out,X_3_out,X_4_out = X_1_out.reshape(b,c,h,w,l),X_2_out.reshape(b,c,h,w,l),X_3_out.reshape(b,c,h,w,l),X_4_out.reshape(b,c,h,w,l)

#         return x1+X_1_out,x2+X_2_out,x3+X_3_out,x4+X_4_out


# ###Unet Encoder block
# class EnBlock(nn.Module):
#     def __init__(self,in_c):
#         super(EnBlock,self).__init__()
#         self.conv = nn.Sequential(
#             nn.GroupNorm(in_c//8,in_c),
#             nn.ReLU(True),
#             nn.Conv3d(in_c, in_c,3,padding=1),
#             nn.GroupNorm(in_c//8,in_c),
#             nn.ReLU(True),
#             nn.Conv3d(in_c, in_c,3,padding=1),
#         )
#     def forward(self, x):
#         res = x
#         out = self.conv(x)
#         return (out+res)

# class EnDown(nn.Module):
#     def __init__(self,in_c,ou_c):
#         super(EnDown,self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv3d(in_c,ou_c,kernel_size=3,stride=2,padding=1)
#         )
#     def forward(self, x):
#         out = self.conv(x)
#         return out

# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder,self).__init__()
#         self.init_conv = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3,
#                                    padding=1)  # 64,128,128,128
#         self.en1 = EnBlock(32)
#         self.ed1 = EnDown(32, 64)
#         self.en2 = EnBlock(64)
#         self.ed2 = EnDown(64, 128)
#         self.en3 = EnBlock(128)
#         self.ed3 = EnDown(128, 256)
#         self.en4 = EnBlock(256)
#     def forward(self, x):
#         x = self.init_conv(x)
#         c1 = self.en1(x)
#         p1 = self.ed1(c1)
#         c2 = self.en2(p1)
#         p2 = self.ed2(c2)
#         c3 = self.en3(p2)
#         p3 = self.ed3(c3)
#         c4 = self.en4(p3)
#         return c1,c2,c3,c4

# ###Unet Dncoder block

# class DnUp(nn.Module):
#     def __init__(self, in_c, ou_c):
#         super(DnUp, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv3d(in_c,in_c,1),
#             nn.ConvTranspose3d(in_c,ou_c,2,2)
#         )

#     def forward(self, x,en_x):
#         out = self.conv(x)


#         return (out+en_x)

# def kaiming_init(m):
#     if isinstance(m, (nn.Linear, nn.Conv3d)):
#         init.kaiming_normal_(m.weight)
#         if m.bias is not None:
#             m.bias.data.fill_(0)
#     elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
#         m.weight.data.fill_(1)
#         if m.bias is not None:
#             m.bias.data.fill_(0)


# class no_share_unet(nn.Module):
#     def __init__(self,in_channel,out_channel,diff=False,deepSupvision=False):
#         super(no_share_unet, self).__init__()
#         class_nums = out_channel
#         self.is_diff = diff
#         self.deepSupvision = deepSupvision


#         self.diff1 = fianl_diff_code_block(64)
#         self.diff2 = fianl_diff_code_block(128)
#         self.diff3 = fianl_diff_code_block(256)
#         self.diff4 = fianl_diff_code_block(128)
#         self.diff5 = fianl_diff_code_block(64)


#         self.init_conv_1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3,
#                                    padding=1)  # 64,128,128,128
#         self.en1_1 = EnBlock(32)
#         self.ed1_1 = EnDown(32, 64)
#         self.en2_1 = EnBlock(64)
#         self.ed2_1 = EnDown(64, 128)
#         self.en3_1 = EnBlock(128)
#         self.ed3_1 = EnDown(128, 256)
#         self.en4_1 = EnBlock(256)

#         self.init_conv_2 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3,
#                                      padding=1)  # 64,128,128,128
#         self.en1_2 = EnBlock(32)
#         self.ed1_2 = EnDown(32, 64)
#         self.en2_2 = EnBlock(64)
#         self.ed2_2 = EnDown(64, 128)
#         self.en3_2 = EnBlock(128)
#         self.ed3_2 = EnDown(128, 256)
#         self.en4_2 = EnBlock(256)

#         self.init_conv_3 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3,
#                                      padding=1)  # 64,128,128,128
#         self.en1_3 = EnBlock(32)
#         self.ed1_3 = EnDown(32, 64)
#         self.en2_3 = EnBlock(64)
#         self.ed2_3 = EnDown(64, 128)
#         self.en3_3 = EnBlock(128)
#         self.ed3_3 = EnDown(128, 256)
#         self.en4_3 = EnBlock(256)

#         self.init_conv_4 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3,
#                                      padding=1)  # 64,128,128,128
#         self.en1_4 = EnBlock(32)
#         self.ed1_4 = EnDown(32, 64)
#         self.en2_4 = EnBlock(64)
#         self.ed2_4 = EnDown(64, 128)
#         self.en3_4 = EnBlock(128)
#         self.ed3_4 = EnDown(128, 256)
#         self.en4_4 = EnBlock(256)


#         self.ud1_1 = DnUp(256, 128)  #
#         self.un1_1 = EnBlock(128)
#         self.ud2_1 = DnUp(128, 64)
#         self.un2_1 = EnBlock(64)
#         self.ud3_1 = DnUp(64, 32)
#         self.un3_1 = EnBlock(32)

#         self.ud1_2 = DnUp(256, 128)  #
#         self.un1_2 = EnBlock(128)
#         self.ud2_2 = DnUp(128, 64)
#         self.un2_2 = EnBlock(64)
#         self.ud3_2 = DnUp(64, 32)
#         self.un3_2 = EnBlock(32)

#         self.ud1_3 = DnUp(256, 128)  #
#         self.un1_3 = EnBlock(128)
#         self.ud2_3 = DnUp(128, 64)
#         self.un2_3 = EnBlock(64)
#         self.ud3_3 = DnUp(64, 32)
#         self.un3_3 = EnBlock(32)

#         self.ud1_4 = DnUp(256, 128)  #
#         self.un1_4 = EnBlock(128)
#         self.ud2_4 = DnUp(128, 64)
#         self.un2_4 = EnBlock(64)
#         self.ud3_4 = DnUp(64, 32)
#         self.un3_4 = EnBlock(32)

#         self.out_conv1 = nn.Conv3d(32,class_nums,1)
#         self.out_conv2 = nn.Conv3d(32, class_nums, 1)
#         self.out_conv3 = nn.Conv3d(32, class_nums, 1)
#         self.out_conv4 = nn.Conv3d(32, class_nums, 1)

#         self.cat_out_conv = nn.Sequential(
#             nn.Conv3d(4*class_nums,class_nums,1)
#         )

#         if self.deepSupvision:
#             self.stage1Out = nn.Sequential(
#                 nn.Conv3d(128,class_nums,kernel_size=3,padding=1),
#                 nn.Upsample(scale_factor=4)

#             )
#             self.stage2Out = nn.Sequential(
#                 nn.Conv3d(64,class_nums,kernel_size=3,padding=1),
#                 nn.Upsample(scale_factor=2),
#             )


#     def weight_init(self):
#         initializer = kaiming_init

#         for block in self._modules:
#             for m in self._modules[block]:
#                 initializer(m)

#     def forward(self, x1,x2,x3,x4,mask):
#         x1 = self.init_conv_1(x1)
#         x2 = self.init_conv_2(x2)
#         x3 = self.init_conv_3(x3)
#         x4 = self.init_conv_4(x4)


#         c1_1 = self.en1_1(x1)
#         c1_2 = self.en1_2(x2)
#         c1_3 = self.en1_3(x3)
#         c1_4 = self.en1_4(x4)



#         p1_1 = self.ed1_1(c1_1)
#         p1_2 = self.ed1_2(c1_2)
#         p1_3 = self.ed1_3(c1_3)
#         p1_4 = self.ed1_4(c1_4)
#         if self.is_diff:
#             p1_1, p1_2, p1_3, p1_4 = self.diff1(p1_1, p1_2, p1_3, p1_4,mask)


#         c2_1 = self.en2_1(p1_1)
#         c2_2 = self.en2_2(p1_2)
#         c2_3 = self.en2_3(p1_3)
#         c2_4 = self.en2_4(p1_4)



#         p2_1 = self.ed2_1(c2_1)
#         p2_2 = self.ed2_2(c2_2)
#         p2_3 = self.ed2_3(c2_3)
#         p2_4 = self.ed2_4(c2_4)
#         if self.is_diff:
#             p2_1, p2_2, p2_3, p2_4 = self.diff2(p2_1, p2_2, p2_3, p2_4,mask)


#         c3_1 = self.en3_1(p2_1)
#         c3_2 = self.en3_2(p2_2)
#         c3_3 = self.en3_3(p2_3)
#         c3_4 = self.en3_4(p2_4)


#         p3_1 = self.ed3_1(c3_1)
#         p3_2 = self.ed3_2(c3_2)
#         p3_3 = self.ed3_3(c3_3)
#         p3_4 = self.ed3_4(c3_4)
#         if self.is_diff:
#             p3_1, p3_2, p3_3, p3_4 = self.diff3(p3_1, p3_2, p3_3, p3_4,mask)


#         c4_1 = self.en4_1(p3_1)
#         c4_2 = self.en4_2(p3_2)
#         c4_3 = self.en4_3(p3_3)
#         c4_4 = self.en4_4(p3_4)

#         up5_1 = self.ud1_1(c4_1, c3_1)
#         up5_2 = self.ud1_2(c4_2, c3_2)
#         up5_3 = self.ud1_3(c4_3, c3_3)
#         up5_4 = self.ud1_4(c4_4, c3_4)
#         if self.deepSupvision:
#             stage1Out1 = self.stage1Out(up5_1)
#             stage1Out2 = self.stage1Out(up5_2)
#             stage1Out3 = self.stage1Out(up5_3)
#             stage1Out4 = self.stage1Out(up5_4)

#         un5_1 = self.un1_1(up5_1)
#         un5_2 = self.un1_2(up5_2)
#         un5_3 = self.un1_3(up5_3)
#         un5_4 = self.un1_4(up5_4)

#         if self.is_diff:
#             un5_1, un5_2, un5_3, un5_4 = self.diff4(un5_1, un5_2, un5_3, un5_4,mask)


#         up6_1 = self.ud2_1(un5_1, c2_1)
#         up6_2 = self.ud2_2(un5_2, c2_2)
#         up6_3 = self.ud2_3(un5_3, c2_3)
#         up6_4 = self.ud2_4(un5_4, c2_4)
#         if self.deepSupvision:
#             stage2Out1 = self.stage2Out(up6_1)
#             stage2Out2 = self.stage2Out(up6_2)
#             stage2Out3 = self.stage2Out(up6_3)
#             stage2Out4 = self.stage2Out(up6_4)

#         un6_1 = self.un2_1(up6_1)
#         un6_2 = self.un2_2(up6_2)
#         un6_3 = self.un2_3(up6_3)
#         un6_4 = self.un2_4(up6_4)
#         if self.is_diff:
#             un6_1, un6_2, un6_3, un6_4 = self.diff5(un6_1, un6_2, un6_3, un6_4,mask)


#         up7_1 = self.ud3_1(un6_1, c1_1)
#         up7_2 = self.ud3_2(un6_2, c1_2)
#         up7_3 = self.ud3_3(un6_3, c1_3)
#         up7_4 = self.ud3_4(un6_4, c1_4)

#         un7_1 = self.un3_1(up7_1)
#         un7_2 = self.un3_2(up7_2)
#         un7_3 = self.un3_3(up7_3)
#         un7_4 = self.un3_4(up7_4)

#         out1 = self.out_conv1(un7_1)
#         out2 = self.out_conv2(un7_2)
#         out3 = self.out_conv3(un7_3)
#         out4 = self.out_conv4(un7_4)

#         cat_out = self.cat_out_conv(torch.cat((out1,out2 ,out3 , out4),dim=1))

#         if not self.deepSupvision:
#             return out1 , out2 , out3 , out4,cat_out
#         else:
#             return stage1Out1,stage1Out2,stage1Out3,stage1Out4,stage2Out1,stage2Out2,stage2Out3,stage2Out4,out1 , out2 , out3 , out4,cat_out
# if __name__ == '__main__':
#     x1 = torch.rand(1,1,128,128,128).cuda()
#     x2 = torch.rand(1,1,128,128,128).cuda()
#     x3 = torch.rand(1,1,128,128,128).cuda()
#     x4 = torch.rand(1,1,128,128,128).cuda()
#     mask = torch.rand(1,4).cuda()
#     model = no_share_unet(in_channel=1, out_channel=3, diff=True,deepSupvision=True).cuda()
#     res = model(x1,x2,x3,x4,mask)
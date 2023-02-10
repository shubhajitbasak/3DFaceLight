import torch.nn as nn
import block as B
import torch



class myNet(nn.Module):
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3):
        super(myNet, self).__init__()
        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = B.HBCT(in_channels=nf)
        self.B2 = B.HBCT(in_channels=nf)
        self.B3 = B.HBCT(in_channels=nf)
        self.B4 = B.HBCT(in_channels=nf)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')
        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

    def forward(self, input):
        out_fea = self.fea_conv(input)

        out_B1 = self.B1(out_fea)  # BX50X64X64
        out_B2 = self.B2(out_B1)  # BX50X64X64
        out_B3 = self.B3(out_B2)  # BX50X64X64
        out_B4 = self.B4(out_B3)  # BX50X64X64
        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))  # BX50X64X64
        out_lr = self.LR_conv(out_B) + out_fea  # BX50X64X64

        return out_lr


if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    net = myNet()
    out = net(x)
    print(out.shape)

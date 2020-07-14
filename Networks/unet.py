import torch
from torch import nn
import sys
sys.path.append('..')
from Config import config

class Net_basic(nn.Module):
    '''基础网络，仅包含保存、加载模型的功能'''
    def __init__(self):
        super(Net_basic,self).__init__()
        
    def load(self,path):
        '''加载指定模型'''
        self.load_state_dict(torch.load(path))

            
        
    def save(self,path):
        '''保存模型'''
        torch.save(self.state_dict(),path)

# 实现double conv
class double_conv(nn.Module):
    ''' Conv => Batch_Norm => ReLU => Conv2d => Batch_Norm => ReLU
    '''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv.apply(self.init_weights)
    
    def forward(self, x):
        x = self.conv(x)
        return x

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2d:
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias,0)

# 实现input conv
class inconv(nn.Module):
    ''' input conv layer
        let input 3 channels image to 64 channels
        The oly difference between `inconv` and `down` is maxpool layer 
    '''
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x
        
# 实现output conv
class outconv(nn.Module):
    ''' out conv layer
        1*1 conv
    '''
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv=nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    ''' normal down path 
        MaxPool2d => double_conv
    '''
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    ''' up path
        conv_transpose => double_conv
    '''
    def __init__(self, in_ch, out_ch, Transpose=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if Transpose:
            self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        else:
            # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_ch, in_ch//2, kernel_size=1, padding=0),
                                    nn.ReLU(inplace=True))
        self.conv = double_conv(in_ch, out_ch)
        self.up.apply(self.init_weights)

    def forward(self, x1, x2):
        ''' 
            conv output shape = (input_shape - Filter_shape + 2 * padding)/stride + 1
        '''

        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX//2,
                                    diffY // 2, diffY - diffY//2))

        x = torch.cat([x2,x1], dim=1)
        x = self.conv(x)
        return x

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2d:
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias,0)
            
            
class Unet(Net_basic):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()
        self.inc = inconv(in_ch, 64)
        self.down1 = down(64, 128)
        # print(list(self.down1.parameters()))
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.drop3 = nn.Dropout2d(0.5)
        self.down4 = down(512, 1024)
        self.drop4 = nn.Dropout2d(0.5)


        
        self.up1 = up(1024, 512, False)
        self.up2 = up(512, 256, False)
        self.up3 = up(256, 128, False)
        self.up4 = up(128, 64, False)
        self.outc = outconv(64, out_ch)
        




    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.drop3(x4)
        x5 = self.down4(x4)
        x5 = self.drop4(x5)
        
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        #x = self.outc(x)
        

        y = torch.sigmoid(x)

        

        return y
        
        

if __name__=='__main__':
    model=Unet(1,1)
    img=torch.ones((1,1,256,256))
    result,x5=model(img)
    print(result.size())
    print(x5.size())

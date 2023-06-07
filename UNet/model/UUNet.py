import torch
import torch.nn as nn

from .UNet import EncoderBlock, DecoderBlock, ConvBlock

class UUNet(nn.Module):
    def __init__(self):
        super(UUNet, self).__init__()

        """ Encoder 1"""
        self.enc1 = EncoderBlock(3, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        """ Bottleneck """
        self.bottleneck1 = ConvBlock(512, 1024)

        """ Decoder 1 """
        self.dec1 = DecoderBlock(1024, 512)
        self.dec2 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec4 = DecoderBlock(128, 64)

        """ Bridge """
        self.bridge = nn.Conv2d(64, 8, kernel_size=1, padding=0)

        """ Encoder 2 """
        self.encA = EncoderBlock(8, 64)
        self.encB = EncoderBlock(64, 128)
        self.encC = EncoderBlock(128, 256)
        self.encD = EncoderBlock(256, 512)

        """ Bottleneck """
        self.bottleneck2 = ConvBlock(512, 1024)

        """ Decoder 2 """
        self.decA = DecoderBlock(1024, 512)
        self.decB = DecoderBlock(512, 256)
        self.decC = DecoderBlock(256, 128)
        self.decD = DecoderBlock(128, 64)

        self.classifier = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)

        bottleneck1 = self.bottleneck1(p4)

        d1 = self.dec1(bottleneck1, s4)
        d2 = self.dec2(d1, s3)
        d3 = self.dec3(d2, s2)
        d4 = self.dec4(d3, s1)

        bridge = self.bridge(d4)

        sA, pA = self.encA(bridge)
        sB, pB = self.encB(pA)
        sC, pC = self.encC(pB)
        sD, pD = self.encD(pC)

        bottleneck2 = self.bottleneck2(pD)

        dA = self.decA(bottleneck2, sD)
        dB = self.decB(dA, sC)
        dC = self.decC(dB, sB)
        dD = self.decD(dC, sA)
        
        return self.sigmoid(self.classifier(dD))
    
    def predict(self, image, out_threshold=0.5):
        self.eval()
        with torch.no_grad():
            prediction = self.forward(image.unsqueeze(0))[0].squeeze(0).numpy()
            mask = (prediction > out_threshold).astype('uint8')

        return mask
    
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def save_weights(self, path):
        torch.save(self.state_dict(), path)
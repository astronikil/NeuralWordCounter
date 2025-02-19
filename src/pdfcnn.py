import torch
from torch import nn

class pdfcnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        #
        self.cnn_1_1 = nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.cnn_1_2 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(8)
        self.blk_1 = nn.AvgPool2d((2, 2), stride=(2,2))
        #
        self.cnn_2_1 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.cnn_2_2 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(8)
        self.blk_2 = nn.AvgPool2d((2, 2), stride=(2,2))
        #
        self.cnn_3 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.bn_3 = nn.BatchNorm2d(8)
        self.blk_3 = nn.AvgPool2d((4, 4), stride=(4,4))
        #
        self.cnn_4 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.bn_4 = nn.BatchNorm2d(8)
        self.blk_4 = nn.AvgPool2d((4, 4), stride=(4,4))
        #
        self.cnn_5 = nn.Conv2d(8, 32, 3, stride=2, padding=1)
        self.bn_5 = nn.BatchNorm2d(32)
        self.blk_5 = nn.AvgPool2d((2, 2))
        #
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc_classfier = nn.Linear(32, 30)

    def forward(self, x):
        # Normalize data;()"0.99" instead of 1 is to avoid singular input to
        # ReLU)
        x = (0.99 - x/255.0)*1.0
        # 3x3 CNN -> 3X3 CNN -> B.Norm -> /2 Downsample
        x = self.relu( self.cnn_1_1(x) )
        x = self.relu( self.bn_1( self.cnn_1_2(x) ) )
        x = self.blk_1(x)
        # 3x3 CNN -> 3X3 CNN -> B.Norm -> /2 Downsample
        x = self.relu( self.cnn_2_1(x) )
        x = self.relu( self.bn_2( x + self.cnn_2_2(x) ) )
        x = self.blk_2(x)
        # 3x3 CNN -> B.Norm -> /4 Downsample
        x = self.relu( self.bn_3( x + self.cnn_3(x) ) )
        x = self.blk_3(x)
        # 3x3 CNN -> B.Norm -> /4 Downsample
        x = self.relu( self.bn_4( x + self.cnn_4(x) ) )
        x = self.blk_4(x)
        # 3x3 CNN -> B.Norm -> /2 Downsample
        x = self.relu( self.bn_5( self.cnn_5(x) ) )
        x = self.blk_5(x)
        #Linearize 32 channels
        x = self.flatten(x)
        # MLP block
        x = self.relu( self.fc2(x) )
        x = self.relu( self.fc3(x) )
        # 30 Logits to be fed into softmax/cross-entropy loss
        x = self.fc_classfier(x)
        return x

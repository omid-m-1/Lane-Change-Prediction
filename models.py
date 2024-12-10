import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import params

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout=nn.Dropout(p=dropout)
        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp((torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model)))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(1)
        self.register_buffer('pe',pe)

    def forward(self,x):
        x=x+self.pe[:x.size(0)]
        return self.dropout(x)

class SingleCNNSixLayer(nn.Module):
    def __init__(self):
        super(SingleCNNSixLayer, self).__init__()
        # 6 conv layers, every 2 followed by pooling:
        # Stage 1:
        self.conv1=nn.Conv3d(1,16,(3,3,3),padding=1)
        self.relu1=nn.ReLU()
        self.conv2=nn.Conv3d(16,32,(3,3,3),padding=1)
        self.relu2=nn.ReLU()
        self.pool1=nn.MaxPool3d((1,2,2)) # halving W,H: (200,80)->(100,40)

        # Stage 2:
        self.conv3=nn.Conv3d(32,48,(3,3,3),padding=1)
        self.relu3=nn.ReLU()
        self.conv4=nn.Conv3d(48,64,(3,3,3),padding=1)
        self.relu4=nn.ReLU()
        self.pool2=nn.MaxPool3d((1,2,2)) # (100,40)->(50,20)

        # Stage 3:
        self.conv5=nn.Conv3d(64,64,(3,3,3),padding=1)
        self.relu5=nn.ReLU()
        self.conv6=nn.Conv3d(64,64,(3,3,3),padding=1)
        self.relu6=nn.ReLU()
        self.pool3=nn.MaxPool3d((1,2,2)) # (50,20)->(25,10)

        self.adaptive_pool=nn.AdaptiveAvgPool3d((5,24,10)) # from (5,25,10) to (5,24,10)
        self.reduce_conv=nn.Conv3d(64,1,kernel_size=1)

    def forward(self,x):
        # x:(B,1,5,200,80)
        # stage 1:
        x=self.conv1(x);x=self.relu1(x)
        x=self.conv2(x);x=self.relu2(x)
        x=self.pool1(x) # (B,32,5,100,40)

        # stage 2:
        x=self.conv3(x);x=self.relu3(x)
        x=self.conv4(x);x=self.relu4(x)
        x=self.pool2(x) # (B,64,5,50,20)

        # stage 3:
        x=self.conv5(x);x=self.relu5(x)
        x=self.conv6(x);x=self.relu6(x)
        x=self.pool3(x) # (B,64,5,25,10)

        x=self.adaptive_pool(x) # (B,64,5,24,10)
        B,C,D,H,W=x.size()

        x=self.reduce_conv(x) # (B,1,5,24,10)
        x=x.squeeze(1) # (B,5,24,10)
        x=x.view(B,5,2,12,2,5)
        x=x.permute(0,1,2,4,3,5)
        x=x.reshape(B,5*2*2,12*5)
        return x

class CNNTransformerModel(nn.Module):
    def __init__(self, batch_size, device, hyperparams_dict, parameters, drop_prob=0.5):
        super(CNNTransformerModel, self).__init__()
        self.batch_size=batch_size
        self.device=device
        self.task=hyperparams_dict['task']

        # 5 parallel pipelines
        self.cnn_pipelines=nn.ModuleList([SingleCNNSixLayer() for _ in range(5)])

        self.d_model=60
        self.nhead=4
        self.num_layers_transformer=4
        self.dim_feedforward=256
        encoder_layer=TransformerEncoderLayer(d_model=self.d_model,nhead=self.nhead,
                                              dim_feedforward=self.dim_feedforward,dropout=drop_prob)
        self.transformer_encoder=TransformerEncoder(encoder_layer,self.num_layers_transformer)
        self.pos_encoder=PositionalEncoding(self.d_model,drop_prob)

        self.fc1 = nn.Linear(self.d_model*2*100, 512)
        self.dropout1 = nn.Dropout(drop_prob)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(drop_prob)

        self.num_classes=12
        self.fc_class=nn.Linear(256,self.num_classes)
        self.fc_ttlc=nn.Linear(256,1)

    def forward(self,x_in):
        # x_in:(B,25,200,80)
        B=x_in.size(0)
        # reshape to (B,5,5,200,80)
        x_5g=x_in.view(B,5,5,200,80)

        # Process each group with a pipeline:
        seq_feats=[]
        for i in range(5):
            group_i=x_5g[:,i] # (B,5,200,80)
            group_i=group_i.unsqueeze(1) # (B,1,5,200,80)
            gf=self.cnn_pipelines[i](group_i) # (B,20,60)
            seq_feats.append(gf)

        # concat (B,20,60)*5 => (B,100,60)
        out_final=torch.cat(seq_feats,dim=1) # (B,100,60)

        # Transformer: (100,B,60)
        x_trans=out_final.permute(1,0,2)
        x_trans=self.pos_encoder(x_trans)
        transformer_output=self.transformer_encoder(x_trans) # (100,B,60)

        # concat input/output: (100,B,120)
        combined=torch.cat([transformer_output,x_trans],dim=2)
        flattened =combined.permute(1,0,2).contiguous().view(B,-1) # (B,120*100)

        # Pass through the new dense layers with dropout
        x = F.relu(self.fc1(flattened))  # (B, 512)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))  # (B, 256)
        x = self.dropout2(x)

        lc_pred=self.fc_class(x)  # (B,12)
        ttlc_pred=self.fc_ttlc(x) # (B,1)
        return {'lc_pred':lc_pred,'ttlc_pred':ttlc_pred,'features':x}


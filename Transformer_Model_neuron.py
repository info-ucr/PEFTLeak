import torch
from torch import nn
import numpy as np
import random
import math
from tqdm import tqdm

IMG_SIZE=32

class Create_Patch(nn.Module):
    def __init__(self,embedding_dim,patch_dim, patch_size, num_patch, dropout, in_channels):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(size=(1,embedding_dim)), requires_grad=True)
        #torch.save(self.cls_token, "cls_token.pt")
        self.position_embeddings = nn.Parameter(torch.randn(size=(1, num_patch+1, embedding_dim)), requires_grad=True)
        #self.dropout = nn.Dropout(p=dropout)
        self.fc1=nn.Linear(patch_dim,embedding_dim, bias=False)
        self.flat=nn.Flatten(1)
        self.NUM_PATCHES=num_patch
        self.PATCH_SIZE=patch_size
        self.PATCH_DIM=patch_dim
    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        s=cls_token
        row,col=0,0
        for i in range(self.NUM_PATCHES):
            patch=x[:,:,:,row:row+self.PATCH_SIZE, col:col+self.PATCH_SIZE]
            patch=self.flat(patch)
            patch=patch.reshape(len(patch),1,self.PATCH_DIM)
            patch=self.fc1(patch)
            s=torch.cat([s,patch],dim=1)
    #print(s[0][0])
            col=col+self.PATCH_SIZE

            if col==IMG_SIZE:
                col=0
                row=row+self.PATCH_SIZE
       
        x = self.position_embeddings + s 
        return x
    
class Residual(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        
        
    def forward(self, x, y):
        res= y+x #x is the attention output and y in the first Layer Norm's input
        return res

class Attention(nn.Module):
    def __init__(self, num_patch, num_head, embedding_dim, dim_head,r):
        super().__init__()
        self.LN1= nn.LayerNorm(embedding_dim,eps=1e-6)
        self.QKV= nn.Linear(embedding_dim, 3*embedding_dim, bias=True)
        self.sc= math.sqrt(dim_head)
        self.msa= nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.NUM_PATCHES=num_patch
        self.NUM_HEADS=num_head
        self.HEAD_DIM=dim_head
        self.adapt1=nn.Linear(embedding_dim,r)
        self.relu=nn.ReLU()
        self.adapt2=nn.Linear(r,embedding_dim)
        self.res=Residual(embedding_dim)
    def forward(self, y):
#         print("Layer Norm 1 Input")
#         print(y)
#         print("mean")
#         print(torch.mean(y))
#         print("std")
#         print(torch.std(y))
        x=self.LN1(y)
#         print("Layer Norm 1 Output")
#         print(x)
        QKV= self.QKV(x)
        QKV=QKV.reshape(x.shape[0],self.NUM_PATCHES +1, 3, self.NUM_HEADS, self.HEAD_DIM)
        QKV=QKV.permute(2,0,3,1,4) #(3,batch_size,num_head,num_patch+1,dim_head)
        Q, K, V= QKV[0], QKV[1], QKV[2]
        Q=Q.transpose(-2,-1)
        K=K.transpose(-2,-1)
        V=V.transpose(-2,-1)

        Q_Transpose=Q.transpose(-2,-1)
        #K_Transpose= K.transpose(-2,-1) #-1 is the last dimension, -2 is the second last dimension, swaps this two dimension
        #dot_prod= (Q@K_Transpose)//self.sc #batch_size, num_heads, num_patch+1, num_patch+1
        dot_prod= (Q_Transpose@K)/self.sc
        score_matrix=dot_prod.softmax(dim=-1)
#         print("score matrix")
#         #print(score_matrix)
#         print(score_matrix.shape)
        # print("value")
        # print(V.transpose(-2,-1))
        # print(V.transpose(-2,-1).shape)
        valued_proj= score_matrix@V.transpose(-2,-1) #batch_size, num_heads, num_patches+1, dim_head
        
        valued_proj=valued_proj.transpose(1,2) #batch_size,num_patches+1,  num_heads, dim_head
        concat_heads= valued_proj.flatten(2) #batch_size, num_patches+1, embedding_dim
        
        MSA_output=self.msa(concat_heads)
#         print("MSA output")
#         print(MSA_output)
        MSA_output2=self.adapt1(MSA_output)
        MSA_output2=self.relu(MSA_output2)
            
        MSA_output2=self.adapt2(MSA_output2)
        MSA_output=self.res(MSA_output, MSA_output2)
#         print("adapter 1 output")
#         print(MSA_output)
        return MSA_output
    

class MLP(nn.Module):
    def __init__(self, num_patch, embedding_dim,r):
        super().__init__()
        self.LN2= nn.LayerNorm(embedding_dim, eps=1e-6)
        self.fc1=nn.Linear(embedding_dim, 4*embedding_dim)
        self.gelu=nn.GELU()
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(4*embedding_dim, embedding_dim)
        self.adapt1=nn.Linear(embedding_dim,r)
        self.relu=nn.ReLU()
        self.adapt2=nn.Linear(r,embedding_dim)
        self.res=Residual(embedding_dim)
    def forward(self,x):
#         print("Layer Norm 2 Input")
#         print(x)
#         print("mean")
#         print(torch.mean(x))
#         print("std")
#         print(torch.std(x))
        x=self.LN2(x)
#         print("Layer Norm 2 Output")
#         print(x)
        x=self.fc1(x)     
        x=self.gelu(x)
#         print("MLP output")
#         print(x)
        
        
        #np.save( "a.npy", np.array(a))
        y=self.fc2(x)
#         print("MLP output")
#         print(y)
        x=self.adapt1(y)
        x=self.relu(x)       
        x=self.adapt2(x)
        x=self.res(x,y)
        #print("adapter 2 output")
        #print(x)
        return x
        
class MLPHead(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.head=nn.Linear(embedding_dim, num_classes)
    def forward(self,x): 
        cls= x[:,0] #takes the class patch only
        print(cls.shape)
        cls=self.head(cls)
        return cls
    
class Encoder(nn.Module):
    def __init__(self, num_patch, num_head, embedding_dim,dim_head,r):
        super().__init__()
        self.attn= Attention(num_patch, num_head, embedding_dim, dim_head,r)
        self.mlp= MLP(num_patch, embedding_dim,r)
        self.res=Residual(embedding_dim)
    def forward(self, x):
        y=self.attn(x)
        x=self.res(x,y)
        y=self.mlp(x)
        x=self.res(x,y)
        return x
        
class ViT(nn.Module):
    def __init__(self,r,embedding_dim, patch_dim, patch_size, num_patch, num_head, dim_head,dropout, channel,num_classes ):
        super().__init__()
        self.patch=Create_Patch(embedding_dim, patch_dim, patch_size, num_patch, dropout, channel )
#         self.res= Residual(embedding_dim)
#         self.attn=Attention(num_patch, num_head, embedding_dim, dim_head)
#         self.attn2=Attention(num_patch, num_head, embedding_dim, dim_head)
#         self.mlp = MLP(num_patch, embedding_dim)
#         self.mlp2 = MLP(num_patch, embedding_dim)
        self.mlphead=MLPHead(embedding_dim, num_classes)
        self.LN=nn.LayerNorm(embedding_dim, eps=1e-6)
#         self.encoder1=Encoder(num_patch, num_head, embedding_dim, dim_head,r)
#         self.encoder2=Encoder(num_patch, num_head, embedding_dim, dim_head,r)
#         self.encoder3=Encoder(num_patch, num_head, embedding_dim, dim_head,r)
#         self.encoder4=Encoder(num_patch, num_head, embedding_dim, dim_head,r)
        self.block=12
        for i in range(1, self.block+1):
            setattr(self, f'encoder{i}', Encoder(num_patch, num_head, embedding_dim, dim_head, r))


        
    def forward(self,x):
        x=self.patch(x)
        for i in range(1, self.block+1):
            x = getattr(self, f'encoder{i}')(x)
#         x=self.encoder1(x)
#         x=self.encoder2(x)
#         x=self.encoder3(x)
#         x=self.encoder4(x)
        x=self.LN(x)
        x=self.mlphead(x)
        return x
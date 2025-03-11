import torch
from torch import nn
import numpy as np
import random
import math
import copy

class First_Encoder:
    def __init__(self, d, embedding_dim, num_heads):
        self.EMBED_DIM=embedding_dim
        self.NUM_HEADS=num_heads
        self.HEAD_DIM=embedding_dim//num_heads
        self.d=d
    def attack_parameter(self,w):
        Q_head=torch.zeros(self.EMBED_DIM,self.EMBED_DIM)
        K_head=torch.zeros(self.EMBED_DIM,self.EMBED_DIM)
        V_head=torch.zeros(self.EMBED_DIM,self.EMBED_DIM)
        #print(Q_head[1*PATCH_VECTOR_LEN:(1+1)*PATCH_VECTOR_LEN,1*PATCH_VECTOR_LEN:(1+1)*PATCH_VECTOR_LEN].shape)

        for i in range(self.NUM_HEADS):
            if i==0:
                Q_head[i*self.HEAD_DIM:(i+1)*self.HEAD_DIM,(i)*self.HEAD_DIM:(i+1)*self.HEAD_DIM]=0*torch.eye(self.HEAD_DIM)
                K_head[i*self.HEAD_DIM:(i+1)*self.HEAD_DIM,(i)*self.HEAD_DIM:(i+1)*self.HEAD_DIM]=torch.eye(self.HEAD_DIM)
                V_head[i*self.HEAD_DIM:self.d,(i)*self.HEAD_DIM:self.d]=1*torch.eye(self.d)
            # if i==0:   
            #     Q_head[i*self.HEAD_DIM:(i+1)*self.HEAD_DIM,(i)*self.HEAD_DIM:(i+1)*self.HEAD_DIM]=10**3*torch.eye(self.HEAD_DIM)
            #     K_head[i*self.HEAD_DIM:(i+1)*self.HEAD_DIM,(i)*self.HEAD_DIM:(i+1)*self.HEAD_DIM]=torch.eye(self.HEAD_DIM)
            #     V_head[i*self.HEAD_DIM:(i+1)*self.HEAD_DIM,(i)*self.HEAD_DIM:(i+1)*self.HEAD_DIM]=1*torch.eye(self.HEAD_DIM)
            else:    
                Q_head[i*self.HEAD_DIM:(i+1)*self.HEAD_DIM,(i)*self.HEAD_DIM:(i+1)*self.HEAD_DIM]=10**5*torch.eye(self.HEAD_DIM)
                K_head[i*self.HEAD_DIM:(i+1)*self.HEAD_DIM,(i)*self.HEAD_DIM:(i+1)*self.HEAD_DIM]=torch.eye(self.HEAD_DIM)
                V_head[i*self.HEAD_DIM:(i+1)*self.HEAD_DIM,(i)*self.HEAD_DIM:(i+1)*self.HEAD_DIM]=0*torch.eye(self.HEAD_DIM)

        w[0:self.EMBED_DIM]=Q_head
        w[self.EMBED_DIM:2*self.EMBED_DIM]=K_head
        w[2*self.EMBED_DIM:3*self.EMBED_DIM]=V_head
        return w

class Attention_Layer:
    def __init__(self,d, embedding_dim,patch_dim,num_heads):
        self.EMBED_DIM=embedding_dim
        self.PATCH_DIM=patch_dim
        self.NUM_HEADS=num_heads
        self.HEAD_DIM=patch_dim//num_heads
        self.d=d
    def attack_parameter(self, w):
        
#print(Q.shape)
        Q_head=torch.zeros(self.EMBED_DIM,self.EMBED_DIM)
        K_head=torch.zeros(self.EMBED_DIM,self.EMBED_DIM)
        V_head=torch.zeros(self.EMBED_DIM,self.EMBED_DIM)
        #print(Q_head[1*PATCH_VECTOR_LEN:(1+1)*PATCH_VECTOR_LEN,1*PATCH_VECTOR_LEN:(1+1)*PATCH_VECTOR_LEN].shape)

        for i in range(self.NUM_HEADS):    
                Q_head[i*self.HEAD_DIM:(i+1)*self.HEAD_DIM,(i)*self.HEAD_DIM:(i+1)*self.HEAD_DIM]=10**5*torch.eye(self.HEAD_DIM)
                K_head[i*self.HEAD_DIM:(i+1)*self.HEAD_DIM,(i)*self.HEAD_DIM:(i+1)*self.HEAD_DIM]=torch.eye(self.HEAD_DIM)
                V_head[i*self.HEAD_DIM:(i+1)*self.HEAD_DIM,(i)*self.HEAD_DIM:(i+1)*self.HEAD_DIM]=1*torch.eye(self.HEAD_DIM)

        w[0:self.EMBED_DIM]=Q_head
        w[self.EMBED_DIM:2*self.EMBED_DIM]=K_head
        w[2*self.EMBED_DIM:3*self.EMBED_DIM]=V_head
        return w

class Linear_Embedding:
    def __init__(self, embedding_dim, patch_dim, num_patch):
        self.EMBED_DIM=embedding_dim
        self.PATCH_DIM=patch_dim
        self.NUM_PATCHES=num_patch

    def attack_parameter(self, E, coeff):
        D=self.PATCH_DIM
        
        E=coeff*torch.eye(D)
        

        Embedding=E
        # for i in range(self.NUM_PATCHES):
        #     Embedding=torch.cat((Embedding, E), axis=0)
        return Embedding

class MLP_Identity:
    def __init__(self, embedding_dim):
        #self.All_PATCH=num_patch+1
        self.EMBED_DIM=embedding_dim
    def attack_parameter(self,w):
        w=torch.zeros(w.size())
        w[0:self.EMBED_DIM,0:self.EMBED_DIM]=torch.eye(self.EMBED_DIM)
        return w
            

class MLP_Layer:
    def __init__(self, num_bins,embedding_dim, patch_dim, num_patch):
        self.num_bins=num_bins
        self.PATCH_DIM=patch_dim
        self.EMBED_DIM=embedding_dim
        self.All_PATCH=num_patch+1
    def attack_parameter(self, w, w_pos, multiplier):
        for j in range(self.All_PATCH-1):
            #w[j*self.num_bins:(j+1)*self.num_bins]=torch.zeros(self.num_bins,self.EMBED_DIM)
            #for i in range(j*self.PATCH_DIM, (j+1)*self.PATCH_DIM):
                #if i%2==0:
            w[j*self.num_bins+5:(j+1)*self.num_bins+5, :]= multiplier*w_pos[j+1] #float(1/2) #float(1/All_PATCH)
                # else:
                #     w[j*self.num_bins:(j+1)*self.num_bins, i]= -float(2) #-float(1/2)   #-float(1/All_PATCH)
        return w
    
class Adapter:
    def __init__(self,r,gap, embedding_dim):
        #self.num_bins=num_bins
        self.EMBED_DIM=embedding_dim
        self.r=r
        self.gap=gap
    def attack_parameter(self, w, w_pos,patch_id, multiplier):
        w[self.gap:self.r,:]=multiplier*w_pos[patch_id]
        return w
        
class First_Adapter:
    def __init__(self, r, embedding_dim):
        self.EMBED_DIM=embedding_dim
        self.r=r
    def attack_parameter(self, w_layer1, w_layer2, w1,w2):
        w_layer1=copy.deepcopy(w1)
        w_layer2=copy.deepcopy(w2)
        return w_layer1, w_layer2
    
class Design:
    def __init__(self,r,gap, d, num_bins, embedding_dim, patch_dim, num_heads, num_patch):
        super().__init__()
        self.attn_layer=Attention_Layer(d,embedding_dim, patch_dim,num_heads)
        self.first_enc=First_Encoder(d, embedding_dim, num_heads)
        self.embed=Linear_Embedding(embedding_dim, patch_dim, num_patch)
        #self.position=Position_Embedding( embedding_dim, patch_dim, num_patch)
        self.mlp=MLP_Layer(num_bins,embedding_dim, patch_dim, num_patch)
        self.mlp_idn=MLP_Identity(embedding_dim)
        self.adapt=Adapter(r, gap, embedding_dim)
        self.first_adapt=First_Adapter(r,embedding_dim)
        
    def first_encoder(self,w):
        w=self.first_enc.attack_parameter(w)
        return w
    
    def attention(self, w):
        w=self.attn_layer.attack_parameter(w)
        return w
    
    def linear_embed(self, w, coeff):
        w=self.embed.attack_parameter(w, coeff)
        return w
    
    def mlp_identity(self, w):
        w=self.mlp_idn.attack_parameter(w)
        return w
    
    def adapter(self, w,w_pos,patch_id, multiplier):
        w=self.adapt.attack_parameter(w,w_pos, patch_id, multiplier)
        return w
    def first_adapter(self, w_layer1, w_layer2, w1, w2):
        w_layer1, w_layer2=self.first_adapt.attack_parameter(w_layer1, w_layer2, w1, w2)
        return w_layer1, w_layer2
    
    def MLP(self, w,w_pos, multiplier):
        w=self.mlp.attack_parameter(w, w_pos, multiplier)
        return w
        
    
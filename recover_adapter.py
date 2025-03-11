import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
import random
import math
from tqdm import tqdm

def reconstruct(gap,block,high, num_bins, weight_grad, bias_grad, i, j, std_inv, mean_inv, w_pos, scale):
    fig, ax=plt.subplots(4,8)
    ax=ax.ravel()
    rec_patch1=[]
    with torch.no_grad():
        for p in range(block):
            for k in range(gap,len(weight_grad[p])-1):
                rec=(weight_grad[p][k]-weight_grad[p][k+1])/(bias_grad[p][k]-bias_grad[p][k+1])
#                 if p%2==0:
#                     rec=rec-high
#                 print(bias_grad[p][k])
#                 print(bias_grad[p][k+1])
                if bias_grad[p][k]-bias_grad[p][k+1]: #rec.abs().sum!=0:
#                     print("Hurrah")
#                     print(p)
#                     print(k)
                    rec_patch1.append(rec)
    print(len(rec_patch1))
    recons1=[]
    count=0
    for rec in rec_patch1:
       #rec=torch.div(rec,2) #rec+e_mean+var1*var3
       rec=rec-w_pos[i]
       #rec=torch.div(rec,0.05)
       #rec=torch.inverse(E).float()@rec   
       rec=torch.div(rec,scale)
       ans=rec.reshape(3,16,16)
       ans = ans.clone().detach()
       recons1.append(ans)
       ans=(ans*std_inv+mean_inv).clamp(0,1)
       ax[count].imshow(ans.permute(1, 2, 0).cpu())
       ax[count].axis("off")
       count+=1
    
    fig.set_figheight(9)
    fig.set_figwidth(9)
    plt.show()
    return recons1
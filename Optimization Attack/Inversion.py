import torch
from torch import nn
import numpy as np
import random
import torch.nn.functional as F
#from .metrics import total_variation as TV
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
device=torch.device("cpu")

def total_variation(x):
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy

def inversion_attack(im_size, model, index1,index2, target_grad, batch_size, label,iter,dm,ds):
    criterion=nn.CrossEntropyLoss()
    alpha=10**(-6)
    pos_coeff=0.1
    x=torch.randn(batch_size,1, 3, im_size, im_size, requires_grad=True)
    #x=x.reshape(1,1,3,32,32)
    optimizer=optim.Adam([x], lr=0.1)
    #print(x.parameters)
    scheduler=StepLR(optimizer, step_size=500, gamma=0.1)
    dummy_gradient=model.state_dict()
    for i in range(iter):
        print("iter:", i)
        if iter%100==0:
            for param_group in optimizer.param_groups:
                print(f"Current learning rate: {param_group['lr']}")
        model.train()
        output=model(x.to(device))
        #print(output)
        loss=criterion(output.to(device),label.to(device))
        model.zero_grad()
        gradient=torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True, only_inputs=True)
        #loss.backward()
        weight_grad_dummy=[]
        bias_grad_dummy=[]
        count1=0
        count2=0
        for param in gradient:
            if count1 in index1:
                weight_grad_dummy.append(param.view(-1))
                #print(count1)
            count1+=1
            if count2 in index2:
                bias_grad_dummy.append(param.view(-1))
            count2+=1
#         for name, param in model.named_parameters():
#             if any(f"encoder{i}.{layer}.adapt1.weight" == name for i in range(1, 13) for layer in ["attn", "mlp"]):
#                 weight_grad_dummy.append(param.grad)
#         for name, param in model.named_parameters():
#             if any(f"encoder{i}.{layer}.adapt1.bias" == name for i in range(1, 13) for layer in ["attn", "mlp"]):
#                 bias_grad_dummy.append(param.grad)
        
        weight_grad_dummy=torch.cat(weight_grad_dummy)
        bias_grad_dummy=torch.cat(bias_grad_dummy)
        grad_dummy=torch.concat((weight_grad_dummy,bias_grad_dummy),axis=0)
#         print(weight_grad_dummy.shape)
#         print(bias_grad_dummy.shape)
        #recon_loss_pos=-F.cosine_similarity(grad_pos_dummy, grad_pos_target, dim=0)
        # recon_loss+= alpha*total_variation(x)
        #recon_loss=torch.sum((weight_grad_dummy-weight_grad).pow(2))+torch.sum((bias_grad_dummy-bias_grad).pow(2)) 
        recon_loss=1-F.cosine_similarity(grad_dummy, target_grad, dim=0)+alpha*total_variation(x)#+pos_coeff*recon_loss_pos
        optimizer.zero_grad()
        model.zero_grad()
        #x.zero_grad()
        #x.grad= None #zero_grad()
        #x.grad.zero_()
        #recon_loss.backward(retain_graph=True)
        if x.grad is not None:
           x.grad.zero_()
        gradient=torch.autograd.grad(recon_loss, x, retain_graph=True, create_graph=True, only_inputs=True)
        #x.grad.sign_()
        x.grad=gradient[0] #This updates x and prevents error
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
             x.data = torch.max(torch.min(x, (1 - dm) / ds), -dm / ds)
        print("recon loss")
        print(recon_loss)
        del weight_grad_dummy
        del bias_grad_dummy
        del gradient
        #del grad_pos_dummy
        torch.cuda.empty_cache()
    return x.detach(), recon_loss
        
        
                
        
    
    
    
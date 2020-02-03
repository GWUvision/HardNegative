import torch
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F

def distMC(Mat_A, Mat_B, norm=1, sq=True):#N by F
    N_A = Mat_A.size(0)
    N_B = Mat_B.size(0)
    
    DC = Mat_A.mm(torch.t(Mat_B))
    if sq:
        DC.fill_diagonal_(-norm)
            
    return DC

def Mat(Lvec):
    N = Lvec.size(0)
    Mask = Lvec.repeat(N,1)
    Same = (Mask==Mask.t())
    return Same.clone().fill_diagonal_(0), ~Same#same diff

class EPHNLoss(Module):
    def __init__(self):
        super(EPHNLoss, self).__init__()
        self.semi = False
        self.order = 1
        self.lam = 1
        
    def forward(self, fvec, Lvec):
        N = fvec.size(0)
        fvec_norm = F.normalize(fvec, p = 2, dim = 1)
        # matting
        Same, Diff = Mat(Lvec.view(-1))
        
        # Similarity Matrix and Tensor
        Dist = distMC(fvec_norm,fvec_norm)
        
        ############################################
        # finding max similarity on same label pairs
        D_detach_P = Dist.clone().detach()
        D_detach_P[Diff]=-1
        D_detach_P[D_detach_P>0.9999]=-1
        V_pos, I_pos = D_detach_P.max(1)
 
        # prevent duplicated pairs
        Mask_not_drop_pos = (V_pos>0)

        # extracting pos score
        Pos = Dist[torch.arange(0,N), I_pos]
        Pos_log = Pos.clone().detach().cpu()
        
        ############################################
        # finding max similarity on diff label pairs
        D_detach_N = Dist.clone().detach()
        D_detach_N[Same]=-1
        if self.semi:
            V_neg, I_neg = D_detach_N.max(1)
            Neg_tmp = Dist[torch.arange(0,N), I_neg]
            Neg_log = Neg_tmp.clone().detach().cpu()
            
            D_detach_N[(D_detach_N>(V_pos.repeat(N,1).t()))&Diff]=-1#extracting SHN
            
            V_neg, I_neg = D_detach_N.max(1)
            
            # prevent duplicated pairs
            Mask_not_drop_neg = (V_neg>0)

            # extracting neg score
            Neg = Dist[torch.arange(0,N), I_neg]
            
            # Masking
            Mask_not_drop = Mask_not_drop_pos&Mask_not_drop_neg
            Mask1 = (Neg<Pos) & Mask_not_drop
            Mask2 = (Neg_tmp>Pos) & Mask_not_drop
            
        else:
            V_neg, I_neg = D_detach_N.max(1)

            # prevent duplicated pairs
            Mask_not_drop_neg = (V_neg>0)

            # extracting neg score
            Neg = Dist[torch.arange(0,N), I_neg]
            Neg_log = Neg.clone().detach().cpu()
        
            # Masking
            Mask_not_drop = Mask_not_drop_pos&Mask_not_drop_neg
            Mask1 = (Neg<Pos) & Mask_not_drop
            Mask2 = (Neg>Pos) & Mask_not_drop
        
        if self.order==1:
            # triplets
            T = torch.stack([Pos,Neg],1)[Mask_not_drop,:]
            
            # loss
            loss = -F.log_softmax(T/0.1,dim=1)[:,0].mean()
        elif self.order==2:
            # triplets
            T = torch.stack([Pos-0.5*(Pos).pow(2),0.5*(Neg).pow(2)],1)[Mask_not_drop,:]
            
            # loss
            loss = -F.log_softmax(T*10,dim=1)[:,0].mean()
        elif self.order==0:
            # triplets
            T1 = torch.stack([Pos,Neg],1)[Mask1,:]
            T2 = torch.stack([Neg.clone().detach()+0.1,Neg],1)[Mask2,:]

            # loss
            loss = -F.log_softmax(T1/0.1,dim=1)[:,0].mean()-F.log_softmax(T2,dim=1)[:,0].mean()*self.lam
        else:
            # triplets
            T1 = torch.stack([Pos,Neg],1)[Mask1,:]
            T2 = torch.stack([Pos-0.5*(Pos).pow(2),0.5*(Neg).pow(2)],1)[Mask2,:]

            # loss
            loss = -F.log_softmax(T1/0.1,dim=1)[:,0].mean()-F.log_softmax(T2,dim=1)[:,0].mean()*self.lam
            
        print('loss:{:.3f} rt:{:.3f}'.format(loss.item(), Mask_not_drop.float().mean().item()), end='\r')

        return loss, Pos_log[Mask2], Neg_log[Mask2], Pos_log.mean()-Neg_log.mean()
    
    
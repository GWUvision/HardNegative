from _code_Hotel.Train import learn
import os, torch

lam = 0.1
Data = 'HOTEL'
model = 'R50'
dim = 256
data_dict = torch.load('data_dict_emb.pth')
i=0
for order in [0]:
    for LR in [1e-2]:
        if order==1:
            dst = '_result/{}_{}/Order1_LR_{:.0e}_MP/{}/'.format(Data,model,LR,i)
        elif order==2:
            dst = '_result/{}_{}/Order2_LR_{:.0e}_MP/{}/'.format(Data,model,LR,i)
        elif order==3:
            dst = '_result/{}_{}/Comb_LR_{:.0e}_lam{}_MP/{}/'.format(Data,model,LR,lam,i)
        else:
            dst = '_result/{}_{}/Order1C_LR_{:.0e}_lam{}_MP/{}/'.format(Data,model,LR,lam,i)

        print(dst)
        x = learn(dst, Data, data_dict)
        x.batch_size = 512
        x.Graph_size = 2
        x.init_lr = LR
        x.criterion.order = order
        x.criterion.lam = lam
        x.feature(dim, model, num_epochs=50)
        print(dst)
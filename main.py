import argparse
from _code.Train import learn
import os, torch
    
parser = argparse.ArgumentParser(description='running parameters')
parser.add_argument('--Data', type=str, help='dataset name: CUB, CAR, SOP or ICR')
parser.add_argument('--model', type=str, help='backbone model: R18 or R50')
parser.add_argument('--dim', type=int, help='embedding dimension')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--order', type=int, help='order')
parser.add_argument('--lam', type=float, help='lambda')
parser.add_argument('--g', type=int, help='times')
parser.add_argument('--semi', type=int, help='semi')
parser.add_argument('--ep', type=int, help='epochs')
args = parser.parse_args()

order = args.order
lam = args.lam
Data, model, dim, LR = args.Data, args.model, args.dim, args.lr
Gsize = args.g
ep = args.ep
semi = (args.semi==1)

if Data=='HOTEL':
    data_dict = torch.load('/SEAS/home/xuanhong/ICML2020/data_dict_emb.pth'.format(Data))
else:
    data_dict = torch.load('/home/xuanhong/datasets/{}/data_dict_emb.pth'.format(Data))

print(order)

if order==1:
    if semi:
        dst = '_result/{}_{}/SemiO1_LR_{:.0e}_MP_ep{}/G{}/'.format(Data,model,LR,ep,Gsize)
    else:
        dst = '_result/{}_{}/Order1_LR_{:.0e}_MP_ep{}/G{}/'.format(Data,model,LR,ep,Gsize)
elif order==2:
    dst = '_result/{}_{}/Order2_LR_{:.0e}_MP_ep{}/G{}/'.format(Data,model,LR,ep,Gsize)
elif order==3:
    dst = '_result/{}_{}/Comb_LR_{:.0e}_lam{}_MP_ep{}/G{}/'.format(Data,model,LR,lam,ep,Gsize)
elif order==0:
    dst = '_result/{}_{}/Order1C_LR_{:.0e}_lam{}_MP_ep{}/G{}/'.format(Data,model,LR,lam,ep,Gsize)
else:
    print('error in order')


print(dst)
x = learn(dst, Data, data_dict)
x.batch_size = 512
x.Graph_size = Gsize
x.init_lr = LR
x.criterion.order = order
x.criterion.lam = lam
if semi and order==1:
    x.criterion.semi=True
    print('semi')
x.run(dim, model, num_epochs=ep)
print(dst)
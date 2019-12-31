import argparse
from _code_Hotel.Train import learn
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
args = parser.parse_args()

order = args.order
lam = args.lam
Data, model, dim, LR = args.Data, args.model, args.dim, args.lr
Gsize = args.g

if Data=='HOTEL':
    data_dict = torch.load('/SEAS/home/xuanhong/ICML2020/data_dict_emb.pth'.format(Data))
else:
    data_dict = torch.load('/home/xuanhong/datasets/{}/data_dict_emb.pth'.format(Data))

print(order)

if order==1:
    if args.semi==1:
        dst = '_result/{}_{}/SemiO1_LR_{:.0e}_MP_ep100/G{}/'.format(Data,model,LR,Gsize)
    else:
        dst = '_result/{}_{}/Order1_LR_{:.0e}_MP_ep100/G{}/'.format(Data,model,LR,Gsize)
elif order==2:
    dst = '_result/{}_{}/Order2_LR_{:.0e}_MP/G{}/'.format(Data,model,LR,Gsize)
elif order==3:
    dst = '_result/{}_{}/Comb_LR_{:.0e}_lam{}_MP_ep200/G{}/'.format(Data,model,LR,lam,Gsize)
else:
    dst = '_result/{}_{}/Order1C_LR_{:.0e}_lam{}_MP_ep200/G{}/'.format(Data,model,LR,lam,Gsize)


print(dst)
x = learn(dst, Data, data_dict)
x.batch_size = 512
x.Graph_size = 2
x.init_lr = LR
x.criterion.order = order
x.criterion.lam = lam
if args.semi==1 and order==1:
    x.criterion.semi=True
x.run(dim, model, num_epochs=200)
print(dst)
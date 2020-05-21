from _code_Hotel.Reader import ImageReader
import os, torch
from torchvision import models, transforms, datasets
import torch.nn as nn
from _code_Hotel.Utils import recall, recall2, recall2_batch, eva

Data = 'HOTEL'
model = 'R50'
out_dim = 256
imgsize = 256
RGBmean = [0.5838, 0.5146, 0.4470]
RGBstdv = [0.6298, 0.6112, 0.4445]
data_dict = torch.load('data_dict_emb.pth')

dst = '_result/{}_{}/'.format(Data,model)
print(dst)

# model
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, out_dim)

# image transformer
transforms = transforms.Compose([transforms.Resize(imgsize),
                                 transforms.CenterCrop(imgsize),
                                 transforms.ToTensor(),
                                 transforms.Normalize(RGBmean, RGBstdv)])

# dataset
dsets_tra = ImageReader(data_dict['tra'], transforms) 
dsets_val = ImageReader(data_dict['val'], transforms)

# extract the tra feature
Fvec_tra = eva(dsets_tra, model)
torch.save(Fvec_tra, dst + 'traFvecs.pth')

# extract the query feature
Fvec_val = eva(dsets_val, model)
torch.save(Fvec_val, dst + 'valFvecs.pth')

# extract the 100NN index for each query
_,pre100 = recall2_batch(Fvec_val, Fvec_tra, dsets_val.idx_to_class, dsets_tra.idx_to_class)
torch.save(pre100, dst + 'pre100.pth')
        
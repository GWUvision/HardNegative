import os, time

from torchvision import models, transforms, datasets
from torch.utils.data.sampler import SequentialSampler
import torch.optim as optim
import torch.nn as nn
import torch

from .Sampler import BalanceSampler, BalanceSampler2
from .Reader import ImageReader
from .Loss import EPHNLoss
from .Utils import recall, recall2, recall2_batch, eva
from .color_lib import RGBmean, RGBstdv

from torch.utils.tensorboard import SummaryWriter

PHASE = ['tra','val']

class learn():
    def __init__(self, dst, Data, data_dict):
        self.dst = dst
        self.gpuid = [0]
            
        self.imgsize = 256
        self.batch_size = 128
        self.num_workers = 48
        
        self.decay_time = [False,False]
        self.init_lr = 0.001
        self.decay_rate = 0.1
        self.avg = 8
        
        self.Data = Data
        self.data_dict = data_dict
        
        self.RGBmean = RGBmean[Data]
        self.RGBstdv = RGBstdv[Data]
        
        self.criterion = EPHNLoss() 
        self.Graph_size = 2
        self.test_freq = 20
        self.chck_freq = 10
        
        self.gpu_size = 4
        self.writer = SummaryWriter(dst)
        self.global_it = 0
        self.test = False
        if not self.setsys(): print('system error'); return
        
    def run(self, emb_dim, model_name, num_epochs=20):
        self.out_dim = emb_dim
        self.num_epochs = num_epochs
        self.loadData()
        self.setModel(model_name)
        print('output dimension: {}'.format(emb_dim))
        print('train')
        self.opt()
        
    def feature(self, emb_dim, model_name, num_epochs=20):
        self.out_dim = emb_dim
        self.num_epochs = num_epochs
        self.loadData()
#         self.setModel(model_name)
        print('output dimension: {}'.format(emb_dim))
        self.evaluation()

    ##################################################
    # step 0: System check
    ##################################################
    def setsys(self):
        if not torch.cuda.is_available(): print('No GPU detected'); return False
        if not os.path.exists(self.dst): os.makedirs(self.dst)
        return True
    
    ##################################################
    # step 1: Loading Data
    ##################################################
    def loadData(self):
        self.tra_transforms = transforms.Compose([transforms.Resize(int(self.imgsize*1.1)),
                                                  transforms.RandomCrop(self.imgsize),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(self.RGBmean, self.RGBstdv)])
        
        self.val_transforms = transforms.Compose([transforms.Resize(self.imgsize),
                                                  transforms.CenterCrop(self.imgsize),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(self.RGBmean, self.RGBstdv)])

        self.dsets = ImageReader(self.data_dict['tra'], self.tra_transforms) 
        self.intervals = self.dsets.intervals
        self.classSize = len(self.intervals)
        print('number of classes: {}'.format(self.classSize))

        return
    
    ##################################################
    # step 2: Set Model
    ##################################################
    def setModel(self, model_name, epoch=None):
        if model_name == 'R18':
            self.model = models.resnet18(pretrained=True)
            print('Setting model: resnet18')
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.out_dim)
            self.model.avgpool = nn.AvgPool2d(self.avg)
        elif model_name == 'R50':
            self.model = models.resnet50(pretrained=True)
            print('Setting model: resnet50')
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.out_dim)
            self.model.avgpool = nn.AvgPool2d(self.avg)
        else:
            pass
            # self.model = googlenet(pretrained=True)
            # self.model.aux_logits=False
            # num_ftrs = self.model.fc.in_features
            # self.model.fc = nn.Linear(num_ftrs, self.classSize)
            # print('Setting model: GoogleNet')

        print('Training on Single-GPU')
        print('LR is set to {}'.format(self.init_lr))
        
        if epoch:
            self.model.load_state_dict(torch.load(self.dst+str(epoch)+'state_dict.pth'),strict=True)
            print('loaded state dict')
            
        self.model = self.model.cuda()
        if self.gpu_size>1:
            self.model = torch.nn.DataParallel(self.model, device_ids=[i for i in range(self.gpu_size)], output_device=0)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.init_lr, momentum=0.0)
        return
    
    def lr_scheduler(self, epoch):
        if epoch>0.4*self.num_epochs and not self.decay_time[0]: 
            self.decay_time[0] = True
            lr = self.init_lr*self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        if epoch>0.7*self.num_epochs and not self.decay_time[1]: 
            self.decay_time[1] = True
            lr = self.init_lr*self.decay_rate*self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        return
            
    ##################################################
    # step 3: Learning
    ##################################################
    def opt(self):
        if self.Data in ['SOP','ICR']:
            if self.gpu_size>1:
                batch_limit = 80
            else:
                batch_limit = 120
        elif self.Data=='HOTEL':
            batch_limit = int(195*self.Graph_size/2)
            print(batch_limit)
        else:
            batch_limit = 120
            
        # recording time
        since = time.time()
        acc_list = []
        for epoch in range(self.num_epochs+1): 
            # adjust the learning rate
            print('Epoch {}/{} \n '.format(epoch, self.num_epochs) + '-' * 40)
            self.lr_scheduler(epoch)
            tra_loss, N_sample, B_iter = 0,0,0
            
            # train 
            while B_iter<batch_limit:
                tra_loss_tmp, N_sample_tmp, B_iter_tmp = self.tra(N_limit=batch_limit-B_iter)
                tra_loss+=tra_loss_tmp
                N_sample+=N_sample_tmp
                B_iter+=B_iter_tmp

            self.writer.add_scalar('loss', tra_loss/N_sample, epoch)
                
            if epoch%self.test_freq==0 and epoch>0:
                pre100 = self.recall_val2tra(epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
        return
    
    def tra(self, N_limit=-1):
        self.model.module.train(True)  # Set model to training mode
        if self.Data in ['CUB','CAR']:
            dataLoader = torch.utils.data.DataLoader(self.dsets, batch_size=self.batch_size, sampler=BalanceSampler(self.intervals, GSize=self.Graph_size), num_workers=self.num_workers, drop_last=True)
        else: 
            dataLoader = torch.utils.data.DataLoader(self.dsets, batch_size=self.batch_size, sampler=BalanceSampler2(self.intervals, GSize=self.Graph_size), num_workers=self.num_workers, drop_last=True)
        
        L_data, N_data, B_data = 0.0, 0,- 0
        Pos_data, Neg_data, Margin_data = list(), list(), list()
        HN_num, N_data_tmp = 0, 0
        # iterate batch
        for data in dataLoader:
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                inputs_bt, labels_bt = data # <FloatTensor> <LongTensor>
                fvec = self.model(inputs_bt.cuda())
                loss, Pos_log, Neg_log, margin = self.criterion(fvec, labels_bt.cuda())

                loss.backward()
                self.optimizer.step() 
                
                Pos_data.append(Pos_log)
                Neg_data.append(Neg_log)
                Margin_data.append(margin)
                HN_num+=Pos_log.size(0)
                N_data_tmp+=1
                
                if self.global_it%self.chck_freq==0 and self.global_it>0:
                    Pos_data.append(torch.Tensor([0,1]))
                    Neg_data.append(torch.Tensor([0,1]))
                    self.writer.add_histogram(self.Data+'Pos_hist', torch.cat(Pos_data,0).view(-1), self.global_it)
                    self.writer.add_histogram(self.Data+'Neg_hist', torch.cat(Neg_data,0).view(-1), self.global_it)
                    self.writer.add_scalar(self.Data+'Margin', torch.stack(Margin_data,0).mean(), self.global_it)
                    self.writer.add_scalar(self.Data+'HN_num', HN_num/(self.batch_size*N_data_tmp), self.global_it)
                    
                    Pos_data, Neg_data, Margin_data = list(), list(), list()
                    HN_num = 0
                    N_data_tmp = 0
                    
                self.global_it+=1
                
            L_data += loss.item()
            N_data += len(labels_bt)
            B_data += 1
            if B_data>=N_limit and N_limit!=-1: break

        return L_data, N_data, B_data
        
    def evaluation(self):
        # recording time
        since = time.time()
        acc_list = []
        for epoch in range(self.num_epochs+1): 
            print(epoch)
            if epoch%self.test_freq==0 and epoch>0:
                self.setModel('R50', epoch=epoch)
                # calculate the retrieval accuracy
                if self.Data in ['SOP','CUB','CAR']:
                    acc = self.recall_val2val(epoch)
                elif self.Data=='ICR':
                    acc = self.recall_val2gal(epoch)
                elif self.Data=='HOTEL':
                    acc = self.recall_val2tra(epoch)
                else:
                    acc = self.recall_val2tra(epoch)
                    
#                 self.writer.add_scalar(self.Data+'_train_R@1', acc[0], epoch)
#                 self.writer.add_scalar(self.Data+'_test_R@1', acc[1], epoch)
#                 acc_list.append(acc)

        # save model
#         torch.save(acc_list, self.dst + 'acc.pth')
#         torch.save(self.model.cpu(), self.dst + 'model.pth')
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
        return
    
    def recall_val2val(self, epoch):
        self.model.module.train(False)  # Set model to testing mode
        dsets_tra = ImageReader(self.data_dict['tra'], self.val_transforms) 
        dsets_val = ImageReader(self.data_dict['val'], self.val_transforms) 
        Fvec_tra = eva(dsets_tra, self.model)
        Fvec_val = eva(dsets_val, self.model)
        
        if epoch==self.num_epochs:
            torch.save(Fvec_tra, self.dst + str(epoch) + 'traFvecs.pth')
            torch.save(Fvec_val, self.dst + str(epoch) + 'valFvecs.pth')
            torch.save(dsets_tra, self.dst + 'tradsets.pth')
            torch.save(dsets_val, self.dst + 'valdsets.pth')
            
        acc_tra = recall(Fvec_tra, dsets_tra.idx_to_class)
        acc_val = recall(Fvec_val, dsets_val.idx_to_class)
        print('R@1_tra:{:.2f}  R@1_val:{:.2f}'.format(acc_tra*100, acc_val*100)) 
        
        return [acc_tra, acc_val]
    
    def recall_val2tra(self, epoch):
        self.model.train(False)  # Set model to testing mode
        # torch.save(self.model.module.state_dict(), self.dst+str(epoch)+'state_dict.pth')

        dsets_tra = ImageReader(self.data_dict['tra'], self.val_transforms) 
        dsets_val = ImageReader(self.data_dict['val'], self.val_transforms) 
        
        if not os.path.exists(self.dst + str(epoch) + 'traFvecs.pth'):
            Fvec_tra = eva(dsets_tra, self.model)
            torch.save(Fvec_tra, self.dst + str(epoch) + 'traFvecs.pth')
        else: 
            Fvec_tra = torch.load(self.dst + str(epoch) + 'traFvecs.pth')
            
        if not os.path.exists(self.dst + str(epoch) + 'valFvecs.pth'):
            Fvec_val = eva(dsets_val, self.model)
            torch.save(Fvec_val, self.dst + str(epoch) + 'valFvecs.pth')
        else:
            Fvec_val = torch.load(self.dst + str(epoch) + 'valFvecs.pth')
        
        if epoch%5==0:
            torch.save(dsets_tra, self.dst + 'tradsets.pth')
            torch.save(dsets_val, self.dst + 'valdsets.pth')

        if not os.path.exists(self.dst+str(epoch)+'pre100.pth'):
            _,pre100 = recall2_batch(Fvec_val, Fvec_tra, dsets_val.idx_to_class, dsets_tra.idx_to_class)
    #         print('R@1:{:.4f}'.format(acc)) 
            torch.save(pre100, self.dst+str(epoch)+'pre100.pth')
        
        return 0
    
    def recall_val2gal(self, epoch):
        self.model.module.train(False)  # Set model to testing mode
        dsets_gal = ImageReader(self.data_dict['gal'], self.val_transforms) 
        dsets_val = ImageReader(self.data_dict['val'], self.val_transforms) 
        Fvec_gal = eva(dsets_gal, self.model)
        Fvec_val = eva(dsets_val, self.model)
        
        if epoch==self.num_epochs:
            torch.save(Fvec_gal, self.dst + 'galFvecs.pth')
            torch.save(Fvec_val, self.dst + 'valFvecs.pth')
            torch.save(dsets_gal, self.dst + 'galdsets.pth')
            torch.save(dsets_val, self.dst + 'valdsets.pth')
            
        acc = recall2(Fvec_val, Fvec_gal, dsets_val.idx_to_class, dsets_gal.idx_to_class)
        print('R@1:{:.4f}'.format(acc)) 
        
        return [acc,acc]
    
import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, custom_evaluate_synset, get_daparam, custom_match_loss, get_time, TensorDataset, custom_epoch, DiffAugment, ParamDiffAug

# import matplotlib.pyplot as plt 

class Distiller():
    def __init__(self,model = 'ConvNet',method = 'DC',ipc  = 1,eval_mode = "S",num_eval=5,iteration=1000,lr_img = 0.1 ,lr_net = 0.01,batch_real = 256,batch_train = 256,init = 'noise',dis_metric = 'ours'):
        # self.dataset = dataset
        self.model = model

        self.ipc = ipc # images per class
        self.eval_mode = eval_mode #eval mode
        self.lr_img = lr_img #lr update synth
        self.iteration = iteration # lr training model
        self.num_eval = num_eval #
        self.lr_net = lr_net
        self.batch_real = batch_real #batchsize for real data
        self.batch_train = batch_train #  batch training data
        self.init = init # way to init the synth 
        
        self.dis_metric = dis_metric    

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'    
        self.eval_it_pool = np.arange(0, self.iteration+1, 500).tolist() if self.eval_mode == 'S' or self.eval_mode == 'SS' else [self.iteration] # The list of iterations when we evaluate models and record results.
        self.model_eval_pool = get_eval_pool(self.eval_mode, self.model, self.model)

        self.outer_loop, self.inner_loop = get_loops(self.ipc)
        # self.dsa_param = ParamDiffAug()
        # self.dsa = True if self.method == 'DSA' else False      

    def distill(self,dataset = 'MNIST',data_path = 'data',save_path = 'result'):
        if not os.path.exists(data_path):
            os.mkdir(data_path)

        if not os.path.exists(save_path):
            os.mkdir(save_path)        

        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(dataset, data_path)
        
        accs_all_exps = dict() # record performances of all experiments
        for key in self.model_eval_pool:
            accs_all_exps[key] = []

        data_save = []

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)

        images_all = torch.cat(images_all, dim=0).to(self.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=self.device)

        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))
        
        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]
        
        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*self.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=self.device)

        label_syn = torch.tensor([np.ones(self.ipc)*i for i in range(num_classes)], requires_grad=False, device=self.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        if self.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c*self.ipc:(c+1)*self.ipc] = get_images(c, self.ipc).detach().data
        else:
            print('initialize synthetic data from random noise')

        '''training'''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=self.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(self.device)
        print('%s training begins'%get_time())

        for it in range(self.iteration+1):
            if it in self.eval_it_pool:
                for model_eval in self.model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(self.model, model_eval, it))
                    self.epoch_eval_train = 300

                    self.dc_aug_param = get_daparam(dataset, self.model, model_eval, self.ipc)
                    print('DC augmentation parameters: \n', self.dc_aug_param)
                    
                    accs = []

                    for it_eval in range(self.num_eval):
                        net_eval = get_network(model_eval,channel,num_classes, im_size).to(self.device)
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) 

                    _, acc_train, acc_test = custom_evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, self.lr_net,self.device,self.epoch_eval_train,self.batch_train,self.dc_aug_param)
                    accs.append(acc_test)
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))

                ''' visualize and save '''
                save_name = os.path.join(save_path, 'vis_DS_%s_%s_%dipc_iter%d.png'%( dataset, self.model, self.ipc,  it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, save_name, nrow=self.ipc) # Trying normalize = True/False may get better visual effects.

                if it == self.iteration: # record the final results
                    accs_all_exps[model_eval] += accs

            ''' Train synthetic data '''
            net = get_network(self.model, channel, num_classes, im_size).to(self.device) # get a random model
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=self.lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            loss_avg = 0
            self.dc_aug_param = None  # Mute the DC augmentation when learning synthetic data (in inner-loop epoch function) in oder to be consistent with DC paper.
            
            for ol in range(self.outer_loop):

                ''' freeze the running mu and sigma for BatchNorm layers '''
                # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                # This would make the training with BatchNorm layers easier.

                BN_flag = False
                BNSizePC = 16  # for batch normalization
                for module in net.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True
                if BN_flag:
                    img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                    net.train() # for updating the mu, sigma of BatchNorm
                    output_real = net(img_real) # get running mu, sigma
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer
            
                ''' update synthetic data '''
                loss = torch.tensor(0.0).to(self.device)
                for c in range(num_classes):
                    img_real = get_images(c, self.batch_real)
                    lab_real = torch.ones((img_real.shape[0],), device=self.device, dtype=torch.long) * c
                    img_syn = image_syn[c*self.ipc:(c+1)*self.ipc].reshape((self.ipc, channel, im_size[0], im_size[1]))
                    lab_syn = torch.ones((self.ipc,), device=self.device, dtype=torch.long) * c

                    output_real = net(img_real)
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    output_syn = net(img_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                    loss += custom_match_loss(gw_syn, gw_real, self.device,self.dis_metric)

                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
                loss_avg += loss.item()

                if ol == self.outer_loop - 1:
                    break


                ''' update network '''
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=self.batch_train, shuffle=True, num_workers=0)
                for il in range(self.inner_loop):
                    custom_epoch('train', trainloader, net, optimizer_net, criterion,False,self.device,self.dc_aug_param )


            loss_avg /= (num_classes*self.outer_loop)

            if it%10 == 0:
                print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))

            if it == self.iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save }, os.path.join(save_path, 'res_DS_%s_%s_%dipc.pt'%( dataset, self.model, self.ipc)))
        return data_save
x = Distiller()

x.distill()

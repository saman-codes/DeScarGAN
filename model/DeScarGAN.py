# Copyright 2020 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from visdom import Visdom
import os
from utils.tools import npy_loader, normalize, visualize, imshow, standardize, MapTransformOverNumpyArrayChannels, TransposeNumpy, kappa_score, eval_binary_classifier
import torch.nn as nn
from utils.Functions import classification_loss, gradient_penalty, label2onehot
from model.generator_discriminator import Generator, Discriminator
from torch.autograd import Variable

viz = Visdom(server='10.207.1.111/24', port=8097)
if not os.path.exists('save_nets'):
    os.system('mkdir save_nets')

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to run this example."
    )


class Solver(object):
    def __init__(self, dataset_path, dataset):
        super(Solver, self).__init__()
        self.dataset_path = dataset_path
        self.dataset = dataset

    def train(self):

        id = 0

        torch.cuda.set_device(id)
        device = 'cuda'
        print("computations done on ", device)

        netG = Generator().to(device)
        netD = Discriminator().to(device)

        p1 = np.array([np.array(p.shape).prod() for p in netG.parameters()
                       ]).sum()  #number of parameters for G
        p2 = np.array([np.array(p.shape).prod() for p in netD.parameters()
                       ]).sum()  #number of parameters for D
        print(p1, p2, p1 / (256 * 256))

        blank = np.ones((256, 256))
        lossd_window = viz.line(Y=torch.zeros((1)).cpu(),
                                X=torch.zeros((1)).cpu(),
                                opts=dict(xlabel='epoch',
                                          ylabel='Loss',
                                          title='training loss discriminator'))
        lossg_window = viz.line(Y=torch.zeros((1)).cpu(),
                                X=torch.zeros((1)).cpu(),
                                opts=dict(xlabel='epoch',
                                          ylabel='Loss',
                                          title='training loss generator'))
        image_window = viz.image(blank)
        image_window2 = viz.image(blank)
        image_window3 = viz.image(blank)
        image_window4 = viz.image(blank)
        image_window5 = viz.image(blank)
        image_window7 = viz.image(blank)
        val_window = viz.line(
            Y=torch.zeros((1)).cpu(),
            X=torch.zeros((1)).cpu(),
            opts=dict(xlabel='epoch',
                      ylabel='accuracy',
                      title='classification accuracy on validation set'))
        grad_window = viz.line(Y=torch.zeros((1)).cpu(),
                               X=torch.zeros((1)).cpu(),
                               opts=dict(xlabel='epoch',
                                         ylabel='gradient',
                                         title='average gradients'))

        #--------------------------------------------------------------------------------------------------
        #CHOOSE HYPERPARAMETERS
        #------------------------------------------------------------------------------------------
        batchsize = 8
        lambda_id = 50
        lambda_rec = 50
        lambda_gp = 10
        lambda_fake = 20
        lambda_real = 20
        lambda_fake_h = 1
        lambda_cls_d = 5
        lambda_cls_h = 1
        beta1 = 0.5
        beta2 = 0.999
        n_critic = 5

        augmentation_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])

        transformer = transforms.Compose([
            MapTransformOverNumpyArrayChannels(augmentation_transform),
            TransposeNumpy([1, 2, 0]),
            transforms.ToTensor()
        ])

        path = os.path.join(self.dataset_path, 'train')
        print(path, 'PATH')

        path_val = os.path.join(self.dataset_path, 'validate')
        print(path_val, 'PATH_VAL')

        Dataset = torchvision.datasets.DatasetFolder(root=path,
                                                     loader=npy_loader,
                                                     transform=transformer,
                                                     extensions=('.npy', ))
        print('Dataset', len(Dataset))

        train_loader = torch.utils.data.DataLoader(dataset=Dataset,
                                                   batch_size=batchsize,
                                                   shuffle=True,
                                                   num_workers=8)
        val_set = torchvision.datasets.DatasetFolder(root=path_val,
                                                     loader=npy_loader,
                                                     extensions=('.npy', ))

        validate_loader = torch.utils.data.DataLoader(dataset=val_set,
                                                      batch_size=1,
                                                      shuffle=False)

        g_optimizer = torch.optim.Adam(netG.parameters(), 0.0001,
                                       [beta1, beta2])
        d_optimizer = torch.optim.Adam(netD.parameters(), 0.0001,
                                       [beta1, beta2])
        netG, g_optimizer = amp.initialize(netG, g_optimizer, opt_level='O1')
        netD, d_optimizer = amp.initialize(netD, d_optimizer, opt_level='O1')
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()

        netG = torch.nn.DataParallel(netG, device_ids=[id], output_device=id)
        netD = torch.nn.DataParallel(netD, device_ids=[id], output_device=id)

        def weights_init_d(m):

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data,
                                        mode='fan_in',
                                        nonlinearity='relu')

                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        def weights_init_x(m):

            if isinstance(m, nn.Conv2d):  #or isinstance(m, nn.Linear):

                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        netG.apply(weights_init_d)
        netD.apply(weights_init_x)
        try_loading_file = False
        if try_loading_file:

            try:

                netG.load_state_dict(
                    torch.load("./save_nets/netG_synthetic.pt",
                               map_location={
                                   'cuda:0': 'cpu'
                               }))  # does NOT load optimizer state etc.
                netD.load_state_dict(
                    torch.load("./save_nets/netD_synthetic.pt",
                               map_location={'cuda:0': 'cpu'}))

                print("loaded model from file")
            except:
                print("loading model from file failed; created new model")

        c_dim = 2
        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #
        for epoch in range(500):
            running_loss = 0
            sum_loss_real = 0
            sum_loss_fake = 0
            sum_loss_cls = 0
            sum_loss_gp = 0
            sum_h_fake = 0
            sum_h_rec = 0
            running_loss_h = 0
            sum_h_id = 0
            sum_h_cls = 0
            total = 0
            correct = 0
            k = torch.ones(1).long().cuda()
            g = torch.zeros(1).long().cuda()
            netD = netD.train()
            netG = netG.train()

            for i, (X, label_org) in enumerate(train_loader):
                print(X.shape)
                print(label_org)
                num_it = round(len(Dataset) * epoch / batchsize + i, 0)
                print('epoch', epoch, 'iter', i, X.shape)
                if self.dataset == 'Synthetic':
                    x_real = torch.tensor(X[:, :1, :, :]).half().to(device)
                else:
                    # print('x_real before =======>', X.shape)
                    # x_real = np.transpose(np.array(X), (0, 3, 1, 2))
                    # print('x_real after =======>', x_real.shap
                    # viz.image(x_real[0, 0, ...])
                    # x_real = torch.tensor(x_real).half().to(device)
                    x_real = X.half().to(device)

                noise = torch.rand(x_real.shape).to(device)
                noise2 = noise

                x_real = x_real + 0.05 * noise  #add Gaussian noise to input

                inputs = Variable(x_real, requires_grad=True)
                label_org = label_org.to(device)
                v_diseased = torch.eq(label_org, k).nonzero().cuda()
                v_healthy = torch.eq(label_org, g).nonzero().cuda()

                label_h = torch.zeros(len(inputs)).long().cuda()
                label_d = torch.ones(len(inputs)).long().cuda()
                c_d = label2onehot(label_d, c_dim).half().to(device)
                c_h = label2onehot(label_h, c_dim).half().to(device)

                part_d = len(
                    v_diseased
                ) / batchsize  #split the batch into healthy and diseased images
                part_h = len(v_healthy) / batchsize

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #

                # Compute loss with real images.

                if v_healthy.nelement() != 0:
                    (out_h, klasse_h) = netD(inputs[v_healthy][:, 0, :, :, :],
                                             g)
                    d_loss_real_h = -torch.mean(out_h) * part_h
                    d_loss_cls_h = classification_loss(
                        logit=klasse_h,
                        target=label_h[:len(v_healthy)]) * part_h
                else:
                    d_loss_real_h = torch.zeros(1).cuda()
                    d_loss_cls_h = torch.zeros(1).cuda()

                if v_diseased.nelement() != 0:
                    (out_d, klasse_d) = netD(inputs[v_diseased][:, 0, :, :, :],
                                             k)
                    d_loss_real_d = -torch.mean(out_d) * part_d
                    d_loss_cls_d = classification_loss(
                        logit=klasse_d,
                        target=label_d[:len(v_diseased)]) * part_d
                else:
                    d_loss_real_d = torch.zeros(1).cuda()
                    d_loss_cls_d = torch.zeros(1).cuda()

                d_loss_real = d_loss_real_h + d_loss_real_d
                d_loss_cls = d_loss_cls_h + d_loss_cls_d
                (_, klasse) = netD(inputs, g)

                # Compute loss with fake images.
                x_fake_h = netG(inputs, c_h)
                x_fake_d = netG(inputs, c_d)

                x_fake_h += 0.05 * noise2
                x_fake_d += 0.05 * noise2

                (out_h, _) = netD(x_fake_h.detach(), g)
                out_h = torch.mean(out_h, dim=(2, 3))
                (out_d, _) = netD(x_fake_d.detach(), k)
                out_d = torch.mean(out_d, dim=(2, 3))

                d_loss_fake = (torch.mean(out_h) * part_h +
                               torch.mean(out_d) * part_d)

                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real.size(0), 1, 1, 1).half().to(device)
                x_hat_d = alpha * x_real.data + (1 - alpha) * (
                    x_fake_d.data.half().requires_grad_(True))
                x_hat_h = alpha * x_real.data + (1 - alpha) * (
                    x_fake_h.data.half().requires_grad_(True))
                x_hat_d = Variable(x_hat_d, requires_grad=True)
                x_hat_h = Variable(x_hat_h, requires_grad=True)
                (out_d, _) = netD(x_hat_d, k)
                (out_h, _) = netD(x_hat_h, g)

                d_loss_gp = gradient_penalty(out_d.half(),
                                             x_hat_d) + gradient_penalty(
                                                 out_h.half(), x_hat_h)

                # Backward step and optimize.
                d_loss = lambda_real * d_loss_real + lambda_fake * d_loss_fake + lambda_gp * d_loss_gp + lambda_cls_d * d_loss_cls
                g_optimizer.zero_grad()
                d_optimizer.zero_grad()

                with amp.scale_loss(d_loss, d_optimizer) as scaled_loss_d:
                    scaled_loss_d.backward()

                ave_grads = []
                for n, p in netD.named_parameters():
                    if (p.requires_grad) and ("bias"
                                              not in n) and p.grad is not None:
                        ave_grads.append(p.grad.abs().mean())
                gradd = sum(ave_grads)

                nn.utils.clip_grad_norm_(netD.parameters(),
                                         10)  #gradient clipping
                d_optimizer.step()

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                sum_loss_real += d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                sum_loss_fake += d_loss_fake.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                sum_loss_cls += d_loss_cls.item()
                loss['D/loss_gp'] = d_loss_gp.item()
                sum_loss_gp += d_loss_gp.item()
                running_loss += d_loss.item()

                #Plot loss
                if (i + 1) % 5 == 0:
                    viz.line(X=torch.ones((1, 1)).cpu() * num_it,
                             Y=torch.Tensor([running_loss]).unsqueeze(0).cpu(),
                             win=lossd_window,
                             name='total',
                             update='append')
                    viz.line(X=torch.ones((1, 1)).cpu() * num_it,
                             Y=torch.Tensor([sum_loss_real
                                             ]).unsqueeze(0).cpu(),
                             win=lossd_window,
                             name='loss_real',
                             update='append')
                    viz.line(X=torch.ones((1, 1)).cpu() * num_it,
                             Y=torch.Tensor([sum_loss_fake
                                             ]).unsqueeze(0).cpu(),
                             win=lossd_window,
                             name='loss_fake',
                             update='append')
                    viz.line(X=torch.ones((1, 1)).cpu() * num_it,
                             Y=torch.Tensor([sum_loss_cls]).unsqueeze(0).cpu(),
                             win=lossd_window,
                             name='loss_cls',
                             update='append')
                    viz.line(X=torch.ones((1, 1)).cpu() * num_it,
                             Y=torch.Tensor([sum_loss_gp]).unsqueeze(0).cpu(),
                             win=lossd_window,
                             name='loss_gp',
                             update='append')
                    viz.line(X=torch.ones((1, 1)).cpu() * num_it,
                             Y=torch.Tensor([gradd]).unsqueeze(0).cpu(),
                             win=grad_window,
                             name='grad_d',
                             update='append')

                    running_loss = 0
                    sum_loss_real = 0
                    sum_loss_fake = 0
                    sum_loss_gp = 0
                    sum_loss_cls = 0

# =================================================================================== #
#                               3. Train the generator                                #
# =================================================================================== #

                if (i + 1) % n_critic == 0:

                    _, predicted = torch.max(klasse.data, 1)
                    total += 1 * batchsize  # p+=predicted
                    correct += (predicted == label_org).sum().item()
                    accuracy = 100 * correct / total

                    criterion = nn.MSELoss()
                    # classification loss
                    x_fake_h = netG(inputs, c_h) + 0.05 * noise2
                    x_fake_d = netG(inputs, c_d) + 0.05 * noise2
                    (out_h, klasse_h) = netD(x_fake_h, g)
                    out_h = torch.mean(out_h, dim=(2, 3))
                    (out_d, klasse_d) = netD(x_fake_d, k)
                    out_d = torch.mean(out_d, dim=(2, 3))
                    g_loss_cls_d = classification_loss(logit=klasse_d,
                                                       target=label_d)
                    g_loss_cls_h = classification_loss(logit=klasse_h,
                                                       target=label_h)
                    g_loss_cls = g_loss_cls_h + g_loss_cls_d

                    #adversarial loss
                    g_loss_fake = (-torch.mean(out_h) * 1 -
                                   torch.mean(out_d) * 1)

                    #identity loss and reconstruction loss
                    if v_diseased.nelement() != 0:
                        t1_d = (inputs[v_diseased])
                        t2_d = (x_fake_d[v_diseased])
                        loss_id_d = criterion(t1_d, t2_d) * 1

                        x_reconst_d = netG(
                            x_fake_h[v_diseased][:, 0, :, :, :],
                            c_d[:len(v_diseased)]) + 0.05 * noise2[
                                v_diseased]  # diseased- healthy- diseased
                        loss_rec_d = criterion(
                            inputs[v_diseased][:, 0, :, :, :], x_reconst_d) * 1

                    else:
                        loss_id_d = torch.zeros(1).cuda()
                        loss_rec_d = torch.zeros(1).cuda()

                    if v_healthy.nelement() != 0:
                        t1_h = (inputs[v_healthy])
                        t2_h = (x_fake_h[v_healthy])
                        loss_id_h = criterion(t1_h, t2_h) * 1

                        x_reconst_h = netG(
                            x_fake_d[v_healthy][:, 0, :, :, :],
                            c_h[:len(v_healthy)]) + 0.05 * noise2[
                                v_healthy]  # healthy - diseased - healthy
                        loss_rec_h = criterion(
                            inputs[v_healthy][:, 0, :, :, :], x_reconst_h) * 1
                    else:
                        loss_id_h = torch.zeros(1).cuda()
                        loss_rec_h = torch.zeros(1).cuda()

                    g_loss_id = loss_id_d + loss_id_h
                    g_loss_rec = loss_rec_h + loss_rec_d

                    # Backward and optimize.

                    g_loss = lambda_fake_h * g_loss_fake + lambda_rec * g_loss_rec + lambda_id * g_loss_id + lambda_cls_h * g_loss_cls

                    g_optimizer.zero_grad()
                    d_optimizer.zero_grad()

                    with amp.scale_loss(g_loss, g_optimizer) as scaled_loss_h:
                        scaled_loss_h.backward()
                    for n, p in netG.named_parameters():
                        if (p.requires_grad) and (
                                "bias" not in n) and p.grad is not None:
                            ave_grads.append(p.grad.abs().mean())
                    gradg = sum(ave_grads)
                    print('gradd', gradd, 'gradg', gradg)

                    nn.utils.clip_grad_norm_(netG.parameters(), 10)
                    g_optimizer.step()
                    print('memory in MB',
                          torch.cuda.max_memory_allocated() / 1000000)

                    # Logging.
                    running_loss_h += g_loss.item()
                    loss['G/loss_fake'] = g_loss_fake.item()
                    sum_h_fake += g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    sum_h_rec += g_loss_rec.item()
                    loss['G/loss_cls'] = g_loss_cls.item()
                    sum_h_cls += g_loss_cls.item()

                    loss['G/loss_id'] = g_loss_id
                    sum_h_id += g_loss_id.item()

                    #plot loss curves
                    if (i + 1) % 5 == 0:
                        viz.line(X=torch.ones((1, 1)).cpu() * num_it,
                                 Y=torch.Tensor([running_loss_h
                                                 ]).unsqueeze(0).cpu(),
                                 win=lossg_window,
                                 name='total',
                                 update='append')
                        viz.line(X=torch.ones((1, 1)).cpu() * num_it,
                                 Y=torch.Tensor([sum_h_fake
                                                 ]).unsqueeze(0).cpu(),
                                 win=lossg_window,
                                 name='loss_fake',
                                 update='append')
                        viz.line(X=torch.ones((1, 1)).cpu() * num_it,
                                 Y=torch.Tensor([sum_h_rec
                                                 ]).unsqueeze(0).cpu(),
                                 win=lossg_window,
                                 name='loss_reconstruction',
                                 update='append')
                        viz.line(X=torch.ones((1, 1)).cpu() * num_it,
                                 Y=torch.Tensor([sum_h_id]).unsqueeze(0).cpu(),
                                 win=lossg_window,
                                 name='loss_id',
                                 update='append')
                        viz.line(X=torch.ones((1, 1)).cpu() * num_it,
                                 Y=torch.Tensor([sum_h_cls
                                                 ]).unsqueeze(0).cpu(),
                                 win=lossg_window,
                                 name='loss_cls',
                                 update='append')
                        viz.line(X=torch.ones((1, 1)).cpu() * num_it,
                                 Y=torch.Tensor([gradg]).unsqueeze(0).cpu(),
                                 win=grad_window,
                                 name='grad_h',
                                 update='append')

                        image_window = viz.image(
                            visualize(x_fake_d[0, 0, :, :]),
                            win=image_window,
                            opts=dict(caption='generated_diseased'))
                        image_window4 = viz.image(
                            visualize(x_fake_h[0, 0, :, :]),
                            win=image_window4,
                            opts=dict(caption='generated_healthy'))
                        image_window2 = viz.image(
                            visualize(x_real[0, 0, :, :]),
                            win=image_window2,
                            opts=dict(caption="original"))
                        if v_diseased.nelement() != 0:
                            image_window3 = viz.image(
                                visualize(x_reconst_d[0, 0, :, :]),
                                win=image_window3,
                                opts=dict(caption="reconstructed diseased"))
                        diff = inputs - x_fake_h
                        image_window5 = viz.heatmap(
                            (diff[0, 0, :, :]),
                            win=image_window5,
                            opts=dict(caption="difference"))

                        sum_h_fake = 0
                        sum_h_rec = 0
                        sum_h_id = 0
                        running_loss_h = 0
                        sum_h_cls = 0


#=================================================================================== #
#                                 4. Miscellaneous                                    #
# =================================================================================== #

            if (epoch + 1) % 1 == 0:
                if self.dataset == 'Synthetic':
                    torch.save(netG.state_dict(),
                               "./save_nets/netG_synthetic.pt")
                    torch.save(netD.state_dict(),
                               "./save_nets/netD_synthetic.pt")
                else:
                    torch.save(netG.state_dict(),
                               "./save_nets/netG_chexpert.pt")
                    torch.save(netD.state_dict(),
                               "./save_nets/netD_chexpert.pt")

                viz.line(X=torch.ones((1, 1)).cpu() * epoch,
                         Y=torch.Tensor([accuracy]).unsqueeze(0).cpu(),
                         win=val_window,
                         name='accuracy train',
                         update='append')

                correct = 0
                correct2 = 0
                total = 0

                with torch.no_grad():
                    netD = netD.eval()
                    long_pred = torch.zeros(0).long()
                    long_cls = torch.zeros(0).long()
                    for i, (X2, label_org) in enumerate(validate_loader):
                        if self.dataset == 'Synthetic':
                            x_real = torch.tensor(
                                X2[:, :1, :, :]).half().to(device)
                        else:
                            # x_real = np.transpose(np.array(X2), (0, 3, 1, 2))
                            # print('x_real.shape BEFORE j=---000---> ',
                            #       x_real.shape)
                            # x_real = torch.tensor(x_real).half().to(device)
                            x_real = X2.half().to(device)

                        noise = torch.rand(x_real.shape).to(device)
                        x_real = x_real + 0.05 * noise
                        rand_idx = torch.randperm(label_org.size(0))
                        label_trg = label_org[rand_idx]
                        c_trg = label2onehot(label_trg,
                                             c_dim).half().to(device)

                        x_fake = netG(x_real, c_trg) + 0.05 * noise
                        (_, out_cls) = netD(x_real, g)
                        (_, out_cls2) = netD(x_fake, g)
                        _, predicted = torch.max(out_cls.data, 1)
                        _, predicted2 = torch.max(out_cls2.data, 1)
                        total += 1  # p+=predicted
                        correct += (predicted.cpu() == label_org).sum().item()
                        correct2 += (
                            predicted2.cpu() == label_trg).sum().item()
                        accuracy = 100 * correct / total
                        accuracy2 = 100 * correct2 / total
                        long_pred = torch.cat((long_pred, predicted.cpu()),
                                              dim=0)
                        long_cls = torch.cat((long_cls, label_org), dim=0)

                        image_window7 = viz.image(
                            visualize(x_fake[0, 0, :, :]),
                            win=image_window7,
                            opts=dict(caption='validate'))
                    print(
                        'Accuracy of the network on the test images: %d %%' %
                        (accuracy),
                        'Accuracy of the network on the fake images: %d %%' %
                        (accuracy2))
                    viz.line(X=torch.ones((1, 1)).cpu() * epoch,
                             Y=torch.Tensor([accuracy]).unsqueeze(0).cpu(),
                             win=val_window,
                             name='accuracy',
                             update='append')
                    viz.line(X=torch.ones((1, 1)).cpu() * epoch,
                             Y=torch.Tensor([accuracy2]).unsqueeze(0).cpu(),
                             win=val_window,
                             name='accuracy on fake images',
                             update='append')
                    (kappa, upper, lower) = kappa_score(long_pred, long_cls)
                    print('kappa', kappa)
                    viz.line(X=torch.ones((1, 1)).cpu() * epoch,
                             Y=torch.Tensor([kappa * 100]).unsqueeze(0).cpu(),
                             win=val_window,
                             name='kappa score',
                             update='append')

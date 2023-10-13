# -*- coding:utf-8 -*-
# author: Xinge
# @file: segmentator_3d_asymm_spconv.py

import numpy as np
import spconv
import torch
from torch import nn


def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
                             padding=(0, 1, 1), bias=False, indice_key=indice_key)


def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride,
                             padding=(0, 0, 1), bias=False, indice_key=indice_key)


def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride,
                             padding=(0, 1, 0), bias=False, indice_key=indice_key)


def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride,
                             padding=(1, 0, 0), bias=False, indice_key=indice_key)


def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride,
                             padding=(1, 0, 1), bias=False, indice_key=indice_key)


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ResContextBlock, self).__init__()
        self.conv1 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.LeakyReLU()

        self.conv1_2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.LeakyReLU()

        self.conv2 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut.features = self.act1(shortcut.features)
        shortcut.features = self.bn0(shortcut.features)

        shortcut = self.conv1_2(shortcut)
        shortcut.features = self.act1_2(shortcut.features)
        shortcut.features = self.bn0_2(shortcut.features)

        resA = self.conv2(x)
        resA.features = self.act2(resA.features)
        resA.features = self.bn1(resA.features)

        resA = self.conv3(resA)
        resA.features = self.act3(resA.features)
        resA.features = self.bn2(resA.features)
        resA.features = resA.features + shortcut.features

        return resA


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3, 3), stride=1,
                 pooling=True, drop_out=True, height_pooling=False, indice_key=None):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out

        self.conv1 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act1 = nn.LeakyReLU()
        self.bn0 = nn.BatchNorm1d(out_filters)

        self.conv1_2 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act1_2 = nn.LeakyReLU()
        self.bn0_2 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        if pooling:
            if height_pooling:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=2,
                                                padding=1, indice_key=indice_key, bias=False)
            else:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=(2, 2, 1),
                                                padding=1, indice_key=indice_key, bias=False)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut.features = self.act1(shortcut.features)
        shortcut.features = self.bn0(shortcut.features)

        shortcut = self.conv1_2(shortcut)
        shortcut.features = self.act1_2(shortcut.features)
        shortcut.features = self.bn0_2(shortcut.features)

        resA = self.conv2(x)
        resA.features = self.act2(resA.features)
        resA.features = self.bn1(resA.features)

        resA = self.conv3(resA)
        resA.features = self.act3(resA.features)
        resA.features = self.bn2(resA.features)

        resA.features = resA.features + shortcut.features

        if self.pooling:
            resB = self.pool(resA)
            return resB, resA
        else:
            return resA


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), indice_key=None, up_key=None):
        super(UpBlock, self).__init__()
        # self.drop_out = drop_out
        self.trans_dilao = conv3x3(in_filters, out_filters, indice_key=indice_key + "new_up")
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.BatchNorm1d(out_filters)

        self.conv1 = conv1x3(out_filters, out_filters, indice_key=indice_key)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv3x1(out_filters, out_filters, indice_key=indice_key)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x3(out_filters, out_filters, indice_key=indice_key)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(out_filters)
        # self.dropout3 = nn.Dropout3d(p=dropout_rate)

        self.up_subm = spconv.SparseInverseConv3d(out_filters, out_filters, kernel_size=3, indice_key=up_key,
                                                  bias=False)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, skip):
        upA = self.trans_dilao(x)
        upA.features = self.trans_act(upA.features)
        upA.features = self.trans_bn(upA.features)

        ## upsample
        upA = self.up_subm(upA)

        upA.features = upA.features + skip.features

        upE = self.conv1(upA)
        upE.features = self.act1(upE.features)
        upE.features = self.bn1(upE.features)

        upE = self.conv2(upE)
        upE.features = self.act2(upE.features)
        upE.features = self.bn2(upE.features)

        upE = self.conv3(upE)
        upE.features = self.act3(upE.features)
        upE.features = self.bn3(upE.features)

        return upE


class ReconBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ReconBlock, self).__init__()
        self.conv1 = conv3x1x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv1x3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv1x1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_3 = nn.BatchNorm1d(out_filters)
        self.act1_3 = nn.Sigmoid()

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut.features = self.bn0(shortcut.features)
        shortcut.features = self.act1(shortcut.features)

        shortcut2 = self.conv1_2(x)
        shortcut2.features = self.bn0_2(shortcut2.features)
        shortcut2.features = self.act1_2(shortcut2.features)

        shortcut3 = self.conv1_3(x)
        shortcut3.features = self.bn0_3(shortcut3.features)
        shortcut3.features = self.act1_3(shortcut3.features)
        shortcut.features = shortcut.features + shortcut2.features + shortcut3.features

        shortcut.features = shortcut.features * x.features

        return shortcut


class Asymm_3d_spconv(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)

        self.weight = torch.zeros(20).cuda()
        self.weight[[1,7,9,11,17,18]] = 1.0
        
        self.thing_label = [2,3,4,5,6,7,8]

    def forward(self, voxel_features, coors, batch_size, voxel_features2=None, coors2=None, point_label_tensor=None):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)

        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        
        up2e_sig = up2e
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e.features = torch.cat((up0e.features, up1e.features), 1)

        logits = self.logits(up0e)
        y = logits.dense()

        
        coors_old = coors.clone().long()
        ###################################
        #with torch.no_grad():
        if True:
            coors = coors2
            voxel_features = voxel_features2
            coors = coors.int()
            # import pdb
            # pdb.set_trace()
            ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                          batch_size)
            ret = self.downCntx(ret)
            down1c, down1b = self.resBlock2(ret)
            down2c, down2b = self.resBlock3(down1c)
            down3c, down3b = self.resBlock4(down2c)
            down4c, down4b = self.resBlock5(down3c)

            up4e = self.upBlock0(down4c, down4b)
            up3e = self.upBlock1(up4e, down3b)
            up2e = self.upBlock2(up3e, down2b)
            up2e_ms = up2e

            up1e = self.upBlock3(up2e, down1b)

            up0e = self.ReconNet(up1e)

            up0e.features = torch.cat((up0e.features, up1e.features), 1)

            logits = self.logits(up0e)
            y2 = logits.dense()
       

        up2e_dens = up2e_sig.dense()
        up2e_ms_dens = up2e_ms.dense()
        #print(up2e.indices.size(), "* " )
        
        #print(up2e_dens.size(), "* " * 10)

        indices = up2e.indices.long()
        f1 = up2e_dens[indices[:,0], :, indices[:,1], indices[:,2], indices[:,3]]
        f2 = up2e_ms_dens[indices[:,0], :, indices[:,1], indices[:,2], indices[:,3]].detach()

        fea_loss = torch.mean((f2-f1)*(f2-f1), dim=1)
        fea_loss = fea_loss[fea_loss > 1.]
        #print(fea_loss.size(), torch.mean(fea_loss), "* " * 10)
        if torch.isnan(torch.mean(fea_loss)):
            fea_loss = torch.tensor(0.0).cuda()


        coors = coors.long()
        #print(y.size(), y2.size(),  torch.max(coors_old[:,0]), torch.max(coors_old[:,1]), torch.max(coors_old[:,2]), torch.max(coors_old[:,3]))
        yy = y[coors_old[:,0], :, coors_old[:,1], coors_old[:,2], coors_old[:,3]]
        yy2 = y2[coors_old[:,0], :, coors_old[:,1], coors_old[:,2], coors_old[:,3]]
        l1 = torch.softmax(yy * .3, dim=1)
        l2 = torch.softmax(yy2 * .3, dim=1).detach()
        #print(l2[:10,:], l1[:10,:], l2[:10,:].size())
        diss_loss = torch.mean(l2*torch.abs(torch.log((l2+1e-6)/(l1+1e-6))), dim=1) 
        if point_label_tensor is not None and False:
            #print(point_label_tensor[coors_old[:,0], coors_old[:,1], coors_old[:,2], coors_old[:,3]].size())
            ww_ind = point_label_tensor[coors_old[:,0], coors_old[:,1], coors_old[:,2], coors_old[:,3]]
            ww = self.weight[ww_ind]
            diss_loss = diss_loss[ww>0.5]
      
        diss_loss = diss_loss[~torch.isnan(diss_loss)]
        diss_loss = diss_loss[diss_loss > 0.01]
        if torch.isnan(torch.mean(diss_loss)):
            diss_loss = torch.tensor(0.0).cuda()
        
        #print(torch.mean(diss_loss), diss_loss.size(), "* " * 10)
        
        #############################    aff loss   #####################################
        #print(y.shape, point_label_tensor.shape, "> " * 10)
        aff_loss = 0.0
        for tl in self.thing_label:
            if point_label_tensor is None:
                break
            feas = torch.masked_select(y[0,...], point_label_tensor[0,...].unsqueeze(0)==tl).view(20, -1).permute(1,0)
            feas2 = torch.masked_select(y2[0,...], point_label_tensor[0,...].unsqueeze(0)==tl).view(20, -1).permute(1,0)
            
            feas_m = torch.masked_select(y[1,...], point_label_tensor[1,...].unsqueeze(0)==tl).view(20, -1).permute(1,0)
            feas_m2 = torch.masked_select(y2[1,...], point_label_tensor[1,...].unsqueeze(0)==tl).view(20, -1).permute(1,0)
            
            if (feas.size(0) > 1) & (feas_m.size(0) > 1):
                feas = torch.cat([feas, feas_m], dim=0)
                feas2 = torch.cat([feas2, feas_m2], dim=0)
            elif (feas.size(0) < 1) & (feas_m.size(0) > 1):
                feas = feas_m
                feas2 = feas_m2
            
            if feas.size(0) > 1:
                sims = torch.cosine_similarity(feas.unsqueeze(1), feas.unsqueeze(0), dim=2)
                sims2 = torch.cosine_similarity(feas2.unsqueeze(1), feas2.unsqueeze(0), dim=2)
                aff_loss += torch.mean(torch.abs(sims-sims2.detach()))

        #print(diss_loss.size(), torch.mean(diss_loss), torch.mean(fea_loss), aff_loss)
        diss_loss = torch.mean(diss_loss) * 5. + torch.mean(fea_loss) * .00 + aff_loss * 1.0
        if torch.isnan(diss_loss):
            diss_loss = 0.0
        #diss_loss = torch.mean(diss_loss * (1.0-torch.exp(-diss_loss * 1.).detach())) * 2e1
        
        return y, y2, diss_loss

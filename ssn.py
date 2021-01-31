import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from skimage import io, color
import math
import torch.optim as optim
import sys
from utils import get_spixel_init, compute_neighbor_spix_idx, compute_init_spixel_feat

#####################
# 1. Define the model
#####################

class E_step(nn.Module):
    def __init__(self, valid_spixel, adjust_dist, std):
        super(E_step, self).__init__()
        
        self.register_buffer('valid_spixel', valid_spixel)
        self.register_buffer('adjust_dist', adjust_dist)
        self.std = std

    def forward(self, mean, features):
        # Uses current parameter estimates to evaluate the responsibilities
        #
        # INPUTS:
        # 1) mean:          (torch.FloatTensor: shape = [B, K, C]) mean that defines the likelihood function
        # 2) features:      (torch.FloatTensor: shape = [B, N, C]) defines for each iamge the set of pixel features
        #
        # RETURNS:
        # 1) posteriors:    (torch.FloatTensor: shape = [B, N, K]) contains the responsibilities of each mixture for every pixel in every image)

        feat_sq = torch.sum(torch.pow(features, 2), dim = -1)                                   # shape = [B, N]
        mean_sq = torch.sum(torch.pow(mean, 2), dim = -1)                                       # shape = [B, K]
        dot_prod = torch.bmm(features, mean.transpose(1, 2))                                    # shape = [B, N, K]

        dist_sq = feat_sq[:, :, None] - 2 * dot_prod + mean_sq[:, None, :]                      # shape = [B, N, K]

        # setting distance w.r.t. invalid superpixels to 10000 -> their posteriors become 0
        dist_sq = dist_sq * self.valid_spixel + self.adjust_dist                                # shape = [B, N, K]
        posteriors = F.softmax(- dist_sq / (2.0 * self.std**2), dim = 2)                        # shape = [B, N, K]
        
        return posteriors

class M_step(nn.Module):

    def __init__(self):
        super(M_step, self).__init__()
    
    def forward(self, features, posteriors):
        # Re-estimates parameters using current reponsibilities (posteriors) and parameter estimates
        #
        # INPUTS:
        # 1) features:      (torch.FloatTensor: shape = [B, N, C]) defines for each image the set of pixel features
        # 2) posteriors:    (torch.FloatTensor: shape = [B, N, K]) contains the reponsibilities of each mixture for every pixel in every image
        #
        # RETURNS:
        # 1) mean_new:      (torch.FloatTensor: shape = [N, K, C]) new estimate of the Gaussian mean

        # since the entries of "posteriors" are all positive, performing L_1 normalization of inputs over the specified dimension
        # is equivalent to dividing by each entry by the sum of entries across the specified dimension
        post_norm = F.normalize(posteriors, p=1, dim = 1)
        mean_new = torch.bmm(post_norm.transpose(1, 2), features)

        return mean_new

class EM_step(nn.Module):

    def __init__(self, valid_spixel, adjust_dist, std = 1 / math.sqrt(2.0)):
        super(EM_step, self).__init__()
        self.E_step = E_step(valid_spixel, adjust_dist, std)
        self.M_step = M_step()

    def forward(self, mean, features):
        posteriors = self.E_step(mean, features)
        mean_new = self.M_step(features, posteriors)
        return mean_new

class xylab(nn.Module):
    
    def __init__(self, color_scale, pos_scale):
        super(xylab, self).__init__()
        self.color_scale = color_scale
        self.pos_scale = pos_scale

    def forward(self, Lab):
        ########## compute the XYLab features of the batch of images in Lab ########
        # 1. rgb2Lab
        # 2. create meshgrid of X, Y and expand it along thje mini-batch dimension
        #
        # Lab:   tensor (shape = [N, 3, H, W]): the input image is already opened in LAB format via the Dataloader defined #        in "cityscapes.py" 
        # XY:    tensor (shape = [N, 2, H, W])
        # XYLab: tensor (shape = [N, 5, H, W])
        
        N = Lab.shape[0]
        H = Lab.shape[2]
        W = Lab.shape[3]
        
        Y, X = torch.meshgrid([torch.arange(0, H, out = torch.cuda.FloatTensor()), torch.arange(0, W, out = torch.cuda.FloatTensor())])
        #Y, X = torch.meshgrid([torch.arange(0, H, out = torch.FloatTensor()), torch.arange(0, W, out = torch.FloatTensor())])
        X = self.pos_scale *  X[None, None, :, :].expand(N, -1, -1, -1)                            # shape = [N, 1, H, W]
        Y = self.pos_scale *  Y[None, None, :, :].expand(N, -1, -1, -1)                            # shape = [N, 1, H, W]
        Lab = self.color_scale * Lab.to(torch.float)                                               # requires casting as all input tensors to torch.cat must be of the same dtype

        return torch.cat((X, Y, Lab), dim = 1), X, Y

class crop(nn.Module):
    # all dimensions up to but excluding 'axis' are preserved
    # while the dimensions including and trailing 'axis' are cropped
    # (since the standard dimensions are N, C, H, W,  the default is a spatial crop)

    def __init__(self, axis = 2, offset = 0):
        super(crop, self).__init__()
        self.axis = axis
        self.offset = offset
        
    def forward(self, x, ref):
        for axis in range(self.axis, x.dim()):
            ref_size = ref.size(axis)
            indices = torch.arange(self.offset, self.offset + ref_size)
            indices = x.data.new().resize_(indices.size()).copy_(indices)
            x = x.index_select(axis, indices.long())
        return x

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_pixel_features):
        super(CNN, self).__init__()

        ##############################################
        ########## 1st convolutional layer ###########
        self.conv1_bn_relu_layer = nn.Sequential()
        self.conv1_bn_relu_layer.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1))
        self.conv1_bn_relu_layer.add_module("batchnorm", nn.BatchNorm2d(out_channels))
        self.conv1_bn_relu_layer.add_module("relu", nn.ReLU())

        ##############################################
        ###### 2nd/4th/6th convolutional layers ######
        self.conv2_bn_relu_layer = nn.Sequential()
        self.conv2_bn_relu_layer.add_module("conv", nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1))
        self.conv2_bn_relu_layer.add_module("batchnorm", nn.BatchNorm2d(out_channels))
        self.conv2_bn_relu_layer.add_module("relu", nn.ReLU())

        self.conv4_bn_relu_layer = nn.Sequential()
        self.conv4_bn_relu_layer.add_module("conv", nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1))
        self.conv4_bn_relu_layer.add_module("batchnorm", nn.BatchNorm2d(out_channels))
        self.conv4_bn_relu_layer.add_module("relu", nn.ReLU())

        self.conv6_bn_relu_layer = nn.Sequential()
        self.conv6_bn_relu_layer.add_module("conv", nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1))
        self.conv6_bn_relu_layer.add_module("batchnorm", nn.BatchNorm2d(out_channels))
        self.conv6_bn_relu_layer.add_module("relu", nn.ReLU())
        
        ##############################################
        ######## 3rd/5th convolutional layers ########
        self.pool_conv3_bn_relu_layer = nn.Sequential()
        self.pool_conv3_bn_relu_layer.add_module("maxpool", nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
        self.pool_conv3_bn_relu_layer.add_module("conv", nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1))
        self.pool_conv3_bn_relu_layer.add_module("batchnorm", nn.BatchNorm2d(out_channels)) # the gamma and betas are trainable parameters of Batchnorm
        self.pool_conv3_bn_relu_layer.add_module("relu", nn.ReLU())

        self.pool_conv5_bn_relu_layer = nn.Sequential()
        self.pool_conv5_bn_relu_layer.add_module("maxpool", nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
        self.pool_conv5_bn_relu_layer.add_module("conv", nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1))
        self.pool_conv5_bn_relu_layer.add_module("batchnorm", nn.BatchNorm2d(out_channels))
        self.pool_conv5_bn_relu_layer.add_module("relu", nn.ReLU())

        ##############################################
        ####### 7th (Last) convolutional layer #######
        self.conv7_relu_layer = nn.Sequential()
        self.conv7_relu_layer.add_module("conv", nn.Conv2d(3 * out_channels + in_channels, num_pixel_features - in_channels, kernel_size = 3, stride = 1, padding = 1))
        self.conv7_relu_layer.add_module("relu", nn.ReLU())

        ##############################################
        ################### crop #####################
        self.crop = crop()

    def forward(self, x):

        conv1 = self.conv1_bn_relu_layer(x)
        conv2 = self.conv2_bn_relu_layer(conv1)
        conv3 = self.pool_conv3_bn_relu_layer(conv2)
        conv4 = self.conv4_bn_relu_layer(conv3)
        conv5 = self.pool_conv5_bn_relu_layer(conv4)
        conv6 = self.conv6_bn_relu_layer(conv5)

        # torch.nn.functional.interpolate(size=None, scale_factor = None, mode = 'nearest', align_corners = None)
        # the input data is assumed to be of the form minibatch x channels x [Optinal depth] x [optional height] x width
        # hence, for spatial inputs, we expect a 4D Tensor
        # one can EITHER give a "scale_factor" or a the target output "size" to calculate thje output size (cannot give both, as it's ambiguous)
        conv4_upsample_crop = self.crop(F.interpolate(conv4, scale_factor = 2, mode = 'bilinear'), conv2)
        conv6_upsample_crop = self.crop(F.interpolate(conv6, scale_factor = 4, mode = 'bilinear'), conv2)

        conv7_input = torch.cat((x, conv2, conv4_upsample_crop, conv6_upsample_crop), dim = 1)
        conv7 = self.conv7_relu_layer(conv7_input)
        
        return torch.cat((x, conv7), dim = 1)

class SSN(nn.Module):
    def __init__(self, in_channels, out_channels, C, K_max, iter_EM, color_scale, p_scale, H, W, num_classes):
        super(SSN, self).__init__()
        
        # compute the initial superpixel assignments for each pixel
        spixel_init, feat_spixel_init, num_spixels_w, num_spixels_h = get_spixel_init(K_max, W, H)

        # compute neighboring superpixels for each pixel
        valid_spixel, adjust_dist = compute_neighbor_spix_idx(spixel_init, num_spixels_w, num_spixels_h)
        
        self.spixel_init = spixel_init
        self.feat_spixel_init = feat_spixel_init
        self.valid_spixel = valid_spixel
        self.adjust_dist = adjust_dist
        self.iter_EM = iter_EM
        self.C = C
        self.color_scale = color_scale
        self.p_scale = p_scale

        self.K = num_spixels_w * num_spixels_h
        pos_scale_w = (1.0 * num_spixels_w) / (float(p_scale) * W)
        pos_scale_h = (1.0 * num_spixels_h) / (float(p_scale) * H)
        pos_scale = np.max([pos_scale_h, pos_scale_w])

        self.xylab = xylab(color_scale, pos_scale)
        self.CNN = CNN(in_channels, out_channels, self.C)
        self.EM_step = EM_step(valid_spixel, adjust_dist)
        self.criterion = SegmentationLoss(num_classes = num_classes)
   
    def forward(self, Lab, labels = None):
        
        B = Lab.shape[0]
        XYLab, X, Y = self.xylab(Lab)          
        
        # send the XYLab features through the CNN to obtain the encoded features 
        features = self.CNN(XYLab)
        
        # initialize the (mean) features of each superpixel using the encoded pixel features
        mean_init = compute_init_spixel_feat(features.clone().detach(), self.spixel_init, self.K) 
        mean_new = mean_init                                                                            # shape = [B = batch_size, K, C]

        features = torch.reshape(features, (B, self.C, -1))                                             # shape = [B, C, N]
        features = features.transpose(1, 2)                                                             # shape = [B, N, C]

        # first (T-1) EM iterations
        for t in range(self.iter_EM - 1):
            
            mean_new = self.EM_step(mean_new, features)                                                 # shape = [B, K, C]

        # final EM iteration (for computing just the posteriors)
        posteriors = self.EM_step.E_step(mean_new, features)

        if self.training and labels is not None:
            Ixy = torch.cat((X, Y), dim = 1)    # positional features
            scaled_compact_loss, recon_loss = self.criterion(posteriors, Ixy, labels)
            return scaled_compact_loss, recon_loss
        else:
            return posteriors
    
    def reset_xylab_Estep(self, K_max, W, H):
        # compute the initial superpixel assignments for each pixel
        spixel_init, feat_spixel_init, num_spixels_w, num_spixels_h = get_spixel_init(K_max, W, H)

        # compute neighboring superpixels for each pixel
        valid_spixel, adjust_dist = compute_neighbor_spix_idx(spixel_init, num_spixels_w, num_spixels_h)

        self.K = num_spixels_w * num_spixels_h
        pos_scale_w = (1.0 * num_spixels_w) / (float(self.p_scale) * W)
        pos_scale_h = (1.0 * num_spixels_h) / (float(self.p_scale) * H)
        pos_scale = np.max([pos_scale_h, pos_scale_w])

        self.spixel_init = spixel_init
        self.xylab = xylab(self.color_scale, pos_scale)
        self.EM_step = EM_step(valid_spixel.cuda(), adjust_dist.cuda())

class SegmentationLoss(nn.Module):
    def __init__(self, num_classes, lamb = 1e-5, ignore_index = 255):
        super(SegmentationLoss, self).__init__()
        self.crossentropy_loss = nn.NLLLoss(ignore_index = 255)
        self.mse_loss = nn.MSELoss(reduction = 'sum')
        self.num_classes = num_classes
        self.lamb = lamb

    def forward(self, posteriors, Ixy, labels):
        # computes the (superpixel compactness loss) and (label reconstruction loss)
        #
        # INPUTS:
        # 1) posteriors:    (torch.FloatTensor: shape = [B, N, K]) contains the reponsibilities of each mixture for every pixel in image
        # 2) Ixy:           (torch.FloatTensor: shape = [B, 2, H, W]) contains the positional coordinates of each pixel in image
        # 3) labels:        (torch.FloatTensor: shape = [B, 1, H, W]) contains the label indices for each pixel in image
        #
        # RETURNS:
        # 1) compactness loss scaled by lambda
        # 2) reconstruction loss
        
        B = posteriors.shape[0]

        ########################
        # 1. reconstruction loss
        ########################
        
        Q_row_norm = posteriors                                 # "posteriors" is already normalized across dim = 2 due to the softmax
        Q_col_norm = F.normalize(posteriors, p=1, dim = 1)      # performing L_1 normalization over the specified dimension is equivalent to the desired column normalization

        # reconstructed label
        # the NLLLoss takes as input class probabilities (N, num_classes, d1, d2, ...)
        # and targets in non-one-hot format
        # so we need to turn the "labels" into one-hot format for the reconstructed portion

        labels = labels.reshape((labels.shape[0], labels.shape[1], -1)).squeeze(dim = 1)                            # shape = [B, 1, H, W] -> [B, 1, N = H*W] -> [B, N]
        labels_onehot = F.one_hot(labels, num_classes = self.num_classes).transpose(1,2).to(dtype = torch.float32)  # shape = [B, num_classes, N]

        R_superpixel = torch.bmm(labels_onehot, Q_col_norm)                                                         # shape = [B, K, num_classes]                              
        R_reconstructed = torch.bmm(R_superpixel, Q_row_norm.transpose(1, 2))                                       # shape = [B, num_classes, N]    
        log_R_reconstructed = torch.log(R_reconstructed + 1e-10)                                                    # shape = [B, num_classes, N]
        
        recon_loss = self.crossentropy_loss(log_R_reconstructed, labels.long())

        #####################
        # 2. compactness loss
        #####################
        Ixy = Ixy.reshape((Ixy.shape[0], Ixy.shape[1], -1))                                                         # shape = [B, 2, H, W] -> [B, 2, N = H x W]
        Sxy = torch.bmm(Ixy, Q_col_norm)                                                                            # shape = [B, 2, K]
        
        # hard associations
        H = torch.argmax(posteriors, 2).clone().detach()                                                            # shape = [B, N] 
        Ixy_hat = torch.stack([torch.index_select(Sxy[i,:,:], 1, H[i,:]) for i in range(H.shape[0])], dim = 0)      # shape = [B, 2, N]

        compact_loss = self.mse_loss(Ixy_hat, Ixy) / (2 * B)                                                        # we divide by (2 * B) to keep consistent with original caffe implemention of mse loss
        
        return compact_loss * self.lamb, recon_loss

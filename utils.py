import numpy as np
from scipy import interpolate
import torch
import os
import torch_geometric.utils

def get_spixel_init(num_spixels, img_width, img_height):

    k = num_spixels
    k_w = int(np.floor(np.sqrt(k * img_width / img_height)))
    k_h = int(np.floor(np.sqrt(k * img_height / img_width)))

    spixel_height = img_height / (1. * k_h)
    spixel_width = img_width / (1. * k_w)

    h_coords = np.arange(-spixel_height / 2., img_height + spixel_height - 1,
                         spixel_height)
    w_coords = np.arange(-spixel_width / 2., img_width + spixel_width - 1,
                         spixel_width)
    spix_values = np.int32(np.arange(0, k_w * k_h).reshape((k_h, k_w)))
    spix_values = np.pad(spix_values, 1, 'symmetric')
    f = interpolate.RegularGridInterpolator((h_coords, w_coords), spix_values, method='nearest')

    all_h_coords = np.arange(0, img_height, 1)
    all_w_coords = np.arange(0, img_width, 1)
    all_grid = np.array(np.meshgrid(all_h_coords, all_w_coords, indexing = 'ij'))
    all_points = np.reshape(all_grid, (2, img_width * img_height)).transpose()

    spixel_initmap = f(all_points).reshape((img_height,img_width))

    feat_spixel_initmap = spixel_initmap
    return [spixel_initmap, feat_spixel_initmap, k_w, k_h]


def compute_neighbor_spix_idx(spixel_init, num_spixels_w, num_spixels_h):
    # from the (initial) repsective superpixel assignment of every pixel in the input img,
    # computes the indices of the 9 neighboring superpixels of every pixel
    #
    # INPUTS:
    # 1. spixel_init:   (numpy.array of shape [img_height, img_width]) each entry contains the corresponding pixel's spixel index
    # 2. num_spixel_w:  (int) the number of superpixels along the 'width' dimension
    # 3. num_spixel_h:  (int) the number of superpixels alopng the 'height' dimension
    #
    # RETURNS:
    # 1. valid_spixel:      (torch.FloatTensor of shape [N = img_height * img_width, K]) each row contains whether or not the superpixel is a valid superpixel i.e. fits within the img bounds
    # 2. adjust_dist:       (torch.FloatTensor of shape [N = img_height * img_width, K]) each row contains the amount that the softmax logit in the E-step has to be adjusted by

    img_height, img_width = spixel_init.shape
    N = img_width * img_height
    K = num_spixels_w * num_spixels_h
    filename = 'neighbor_spix_idx_N={}_K={}'.format(N, K)
    
    # assuming we're in the '/data_and_code' directory
    if os.path.isfile(filename + '.npz'):
        npzfile = np.load(filename + '.npz')
        valid_spixel = npzfile['valid_spixel']
        adjust_dist = npzfile['adjust_dist']
    else:
        # if the .npy file doesn't exist
        spixel_init = spixel_init.flatten()                             # np.array of shape (H, W) -> (N = H * W)
        valid_spixel = np.zeros((N, K), dtype = 'float32')              # we initially assume that all superpixels are invalid
        adjust_dist = 10000. * np.ones((N, K), dtype = 'float32')       # we initially assume that all superpixels hence need their distances adjusted to 10,000

        for p_index in range(N):
            print(p_index)
            spixel_index = spixel_init[p_index]                 # the index of the spixel that the current pixel belongs to

            for rel_spix_idx in range(9):
                r_idx_h = rel_spix_idx // 3 - 1
                r_idx_w = rel_spix_idx % 3 - 1

                spix_idx_h = spixel_index + r_idx_h * num_spixels_w
                valid_spixel_p = 1          # True
                adjust_dist_p = 0

                # check whether the neighboring spixel specified by "rel_spix_idx" doesn't leave the vertical bounds of the img
                if spix_idx_h >= K or spix_idx_h <= - 1:
                    spix_idx_h = spixel_index
                    valid_spixel_p = 0      # False
                    adjust_dist_p = 10000

                # check whether the neighboring spixel specified by "rel_spix_idx" doesn't leave the "right" bounds of the img
                if (spix_idx_h + 1) % num_spixels_w == 0 and r_idx_w == 1:
                    r_idx_w = 0
                    valid_spixel_p = 0      # False
                    adjust_dist_p = 10000
                    
                # check whether the neighboring spixel specified by "rel_spix_idx" doesn't leave the "left" boudns of the img
                elif spix_idx_h % num_spixels_w == 0 and r_idx_w == -1:
                    r_idx_w = 0
                    valid_spixel_p = 0      # False
                    adjust_dist_p = 10000

                spix_idx_w = spix_idx_h + r_idx_w
                
                # check whether the neighboring spixel specified by "rel_spix_idx " doesn't leave the bounds of the spix_idx: [0, K)
                if spix_idx_w < K and spix_idx_w > -1:
                    neighbor_spix_idx_p = spix_idx_w
                
                else:
                    neighbor_spix_idx_p = spix_idx_h
                    valid_spixel_p = 0      # False
                    adjust_dist_p = 10000

                valid_spixel[p_index, int(neighbor_spix_idx_p)] = valid_spixel_p
                adjust_dist[p_index, int(neighbor_spix_idx_p)] = adjust_dist_p

        np.savez(filename, valid_spixel = valid_spixel, adjust_dist = adjust_dist)

    valid_spixel = torch.from_numpy(valid_spixel)
    adjust_dist = torch.from_numpy(adjust_dist)
    return valid_spixel, adjust_dist

def compute_init_spixel_feat(trans_feature, spixel_init, num_spixels):
    # initializes the (mean) features of each superpixel using the features encoded by the CNN "trans_feature"
    #
    # INPUTS:
    # 1) trans_feature:     (tensor of shape [B, C, H, W])
    # 2) spixel_init:       (tensor of shape [H, W])
    #
    # RETURNS:
    # 1) init_spixel_feat:  (tensor of shape [B, K, C])

    trans_feature = torch.flatten(trans_feature, start_dim = 2)                                                                 # shape = [B, C, N]
    trans_feature = trans_feature.transpose(0, 2)                                                                               # shape = [N, C, N]
    
    spixel_init = torch.from_numpy(spixel_init.flatten()).long().cuda()                                                         # shape = [N]
    spixel_init = spixel_init[:, None, None].expand(trans_feature.size())                                                       # shape = [N, C, B]

    init_spixel_feat = torch_geometric.utils.scatter_('mean', trans_feature, spixel_init, dim_size = num_spixels)               # shape = [K, C, N]
    return init_spixel_feat.transpose(0, 2).transpose(1, 2)                                                                     # shape = [B, K, C]

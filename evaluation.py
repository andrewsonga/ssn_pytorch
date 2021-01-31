import torch
import torch.nn.functional as F
import numpy as np
from skimage.segmentation._slic import _enforce_label_connectivity_cython

def ASA(posteriors, labels, num_classes, K_max, connectivity = False):
    # INPUTS
    # 1. posteriors:    shape = [B, N, K]
    # 2. labels:        shape = [B, 1, H, W]

    B, _, H, W = labels.shape
    labels = labels.reshape((labels.shape[0], labels.shape[1], -1)).squeeze(dim = 1)                # shape = [B, 1, H, W] -> [B, 1, N = H*W] -> [B, N]
    labels_onehot = F.one_hot(labels, num_classes = num_classes).detach().cpu().numpy()             # shape = [B, N, num_classes]

    B, N, K = posteriors.shape

    hard_assoc = torch.argmax(posteriors, 2).detach().cpu().numpy()                                 # shape = [B, N]
    hard_assoc_hw = hard_assoc.reshape((B, H, W))    
    max_no_pixel_overlap = np.zeros((B, K))

    segment_size = (H * W) / (int(K_max) * 1.0)
    min_size = int(0.06 * segment_size)
    max_size = int(3 * segment_size)

    hard_assoc_hw = hard_assoc.reshape((B, H, W))
    for b in range(hard_assoc.shape[0]):
        for k in range(posteriors.shape[2]):
            if connectivity:
                spix_index_connect = _enforce_label_connectivity_cython(hard_assoc_hw[None, b, :, :], min_size, max_size)[0]
            else:
                spix_index_connect = hard_assoc[b, :]

            indices_k = np.where(spix_index_connect == k)                                                   # indices along the N-dimension
            gt_k = labels_onehot[b, indices_k, :]                                                           # shape = [len(indices_k), num_classes]


            num_gt_k = np.sum(gt_k, 1)                                                                      # shape = [num_classes]
            max_no_pixel_overlap[b, k] = np.max(num_gt_k)

    return np.sum(max_no_pixel_overlap) / (B * N)

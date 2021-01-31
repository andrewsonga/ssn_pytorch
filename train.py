import os
import os.path as osp
import random
import numpy as np
import torch

from torch.nn import DataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import config
from ssn import SSN
from bsds500 import BSDS500, transform_patch_data_train, transform_convert_label
from evaluation import ASA

logger = config.logger

def ensure_dir(dir_path):
    if not osp.isdir(dir_path):
        os.makedirs(dir_path)

class Session():
    def __init__(self, split):
        random.seed(config.RAND_SEED)
        np.random.seed(config.RAND_SEED)
        torch.manual_seed(config.RAND_SEED)
        torch.cuda.manual_seed_all(config.RAND_SEED)
        torch.cuda.set_device(config.DEVICE)
        #########################################
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        #########################################

        self.log_dir = config.LOG_DIR
        self.model_dir = config.MODEL_DIR
        ensure_dir(self.log_dir)
        ensure_dir(self.model_dir)
        logger.info('set log dir as %s' % self.log_dir)
        logger.info('set model dir as %s' % self.model_dir)

        self.step = 1
        self.writer = SummaryWriter(osp.join(self.log_dir, 'train.events'))

        train_transform = transform_patch_data_train([config.TRAIN_H, config.TRAIN_W], config.NUM_CLASSES)
        val_transform = transform_convert_label(config.NUM_CLASSES)

        self.TrainDataset = BSDS500(root = config.ROOT, split = split, transform = train_transform)
        self.TrainLoader = DataLoader(self.TrainDataset,
                                     batch_size = config.BATCH_SIZE,
                                     num_workers = config.NUM_WORKERS,
                                     shuffle = True,
                                     drop_last= True)

        self.ValDataset = BSDS500(root = config.ROOT, split = 'val', transform = val_transform)
        self.ValLoader = DataLoader(self.ValDataset,
                                     batch_size = 1, 
                                     shuffle=False, 
                                     drop_last = False)

        self.net = SSN(config.IN_CHANNELS, config.OUT_CHANNELS, config.C, config.K_MAX, config.ITER_EM, config.COLOR_SCALE, config.P_SCALE, config.TRAIN_H, config.TRAIN_W, config.NUM_CLASSES).cuda()
        self.opt = Adam(self.net.parameters(), lr = config.LR)
        self.net = DataParallel(self.net, device_ids = config.DEVICES)
    
    def write(self, out):
        for k, v in out.items():
            self.writer.add_scalar(k, v, self.step)
        
        out['step'] = self.step
        outputs = [
            '{}: {:.4g}'.format(k, v) 
            for k, v in out.items()]
        logger.info(' '.join(outputs))

    def save_checkpoints(self, name):
        ckp_path = osp.join(self.model_dir, name)
        obj = {
            'net': self.net.module.state_dict(),
            'step': self.step,
        }
        torch.save(obj, ckp_path)
    
    def load_checkpoints(self, name):
        ckp_path = osp.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path, 
                             map_location=lambda storage, loc: storage.cuda())
            logger.info('Load checkpoint %s' % ckp_path)
        except FileNotFoundError:
            logger.error('No checkpoint %s!' % ckp_path)
            return

        self.net.module.load_state_dict(obj['net'])
        self.step = obj['step']
    
    def train_batch(self, image_lab, label):
        scaled_compact_loss, recon_loss = self.net(image_lab, label)
        scaled_compact_loss = scaled_compact_loss.mean()
        recon_loss = recon_loss.mean()        

        loss = scaled_compact_loss + recon_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item(), scaled_compact_loss.item(), recon_loss.item()

def main(ckp_name='latest.pth'):
    sess = Session(split='train')
    sess.load_checkpoints(ckp_name)

    sess.net.train()

    while sess.step < config.ITER_MAX:

        for train_data in sess.TrainLoader:
            images_lab, labels = train_data
            
            loss, scaled_compact_loss, recon_loss = sess.train_batch(images_lab, labels)
            out = {'loss': loss, 'compact loss': scaled_compact_loss, 'recon loss': recon_loss}
            #out = {'loss': loss}

            if sess.step % config.ITER_TEST == 0:
                sess.net.eval()
                sess.net.module.reset_xylab_Estep(config.K_MAX_VAL, sess.ValDataset.height, sess.ValDataset.width)
                asa = 0.
                for val_data in sess.ValLoader:
                    images_lab_val, labels_val = val_data

                    with torch.no_grad():
                        posteriors_val = sess.net(images_lab_val)
                        asa += ASA(posteriors_val, labels_val, config.K_MAX)
                
                out['asa'] = asa / float(sess.ValDataset.__len__())
                del posteriors_val
                torch.cuda.empty_cache()
                
                sess.net.train()
                sess.net.module.reset_xylab_Estep(config.K_MAX, config.TRAIN_H, config.TRAIN_W)

            if sess.step % config.ITER_SAVE == 0:
                sess.save_checkpoints('step_%d.pth' % sess.step)
            
            if sess.step % (config.ITER_SAVE // 100) == 0:
                sess.save_checkpoints('latest.pth')
            
            sess.write(out)
            sess.step += 1

            if sess.step == config.ITER_MAX:
                break
    
    sess.save_checkpoints('final.pth')

if __name__ == "__main__":
    main()

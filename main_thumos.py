import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import json
import matplotlib.pyplot as plt
from collections import OrderedDict
# from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F

from inference_thumos import inference
from utils import misc_utils
from torch.utils.data import Dataset
from dataset.thumos_features import ThumosFeature

from models.model_x4 import Model
from utils.loss import CrossEntropyLoss, GeneralizedCE, BCE, BCE1, Focal, CategoryCrossEntropy, GeneralizedCE_Mask
from config.config_thumos import Config, parse_args, class_dict

from utils.edl_loss import EvidenceLoss
import time

np.set_printoptions(formatter={'float_kind': "{:.2f}".format})
torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)


def load_weight(net, config):
    if config.load_weight:
        model_file = os.path.join(config.model_path, "CAS_Only.pkl")
        print("loading from file for training: ", model_file)
        pretrained_params = torch.load(model_file)

        selected_params = OrderedDict()
        for k, v in pretrained_params.items():
            if 'base_module' in k:
                selected_params[k] = v

        model_dict = net.state_dict()
        model_dict.update(selected_params)
        net.load_state_dict(model_dict)


def get_dataloaders(config):
    train_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='train',
                      modal=config.modal, feature_fps=config.feature_fps,
                      num_segments=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='random', supervision='strong'),
        batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers)

    test_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='test',
                      modal=config.modal, feature_fps=config.feature_fps,
                      num_segments=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='uniform', supervision='strong'),
        batch_size=1,
        shuffle=False, num_workers=config.num_workers)

    return train_loader, test_loader


def set_seed(config):
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.deterministic = True
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = False


class ThumosTrainer():
    def __init__(self, config):
        # config
        self.config = config

        # network
        # self.net = ModelFactory.get_model(config.model_name, config)
        self.net = Model(config.len_feature, config.num_classes, config.num_segments)
        self.net = self.net.cuda()

        # data
        # import pdb; pdb.set_trace()
        self.train_loader, self.test_loader = get_dataloaders(self.config)

        # loss, optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr, betas=(0.9, 0.999), weight_decay=0.0005)   # 0.0005
        self.criterion = CrossEntropyLoss()
        self.Lgce = GeneralizedCE(q=self.config.q_val)
        self.Lgce_mask = GeneralizedCE_Mask(q=self.config.q_val)
        self.bce = BCE()
        self.bce1 = BCE1()
        self.focal = Focal(gamma=2)
        self.ce = CategoryCrossEntropy()

        # parameters
        self.best_mAP = -1 # init
        self.best_acc = 0
        self.step = 0
        self.total_loss_per_epoch = 0
        self.hp_lambda = config.hp_lambda

        self.ulb_prob_t = torch.ones((config.num_classes)).cuda() / config.num_classes
        self.prob_max_mu_t = 1.0 / config.num_classes
        self.prob_max_var_t = 1.0
        self.ema_p = 0.7   # 0.70

    @torch.no_grad()
    def update_prob_t(self, ulb_probs):
        # import pdb;pdb.set_trace()
        ulb_prob_t = ulb_probs.mean(0)
        self.ulb_prob_t = self.ema_p * self.ulb_prob_t + (1 - self.ema_p) * ulb_prob_t

        max_probs, max_idx = ulb_probs.max(dim=-1)
        prob_max_mu_t = torch.mean(max_probs)
        prob_max_var_t = torch.var(max_probs, unbiased=True)
        self.prob_max_mu_t = self.ema_p * self.prob_max_mu_t + (1 - self.ema_p) * prob_max_mu_t.item()
        self.prob_max_var_t = self.ema_p * self.prob_max_var_t + (1 - self.ema_p) * prob_max_var_t.item()
        # print(self.prob_max_mu_t, self.prob_max_var_t)

    @torch.no_grad()
    def calculate_mask(self, probs):
        max_probs, max_idx = probs.max(dim=-1)
        # compute weight
        mu = self.prob_max_mu_t
        var = self.prob_max_var_t
        mask = torch.exp(-((torch.clamp(max_probs - mu, max=0.0) ** 2) / (2 * var / 4)))   # 4
        return max_probs.detach(), mask.detach()

    def test(self):
        self.net.eval()
        with torch.no_grad():
            model_filename = "CAS_Only.pkl"
            self.config.model_file = os.path.join(self.config.model_path, model_filename)
            _mean_ap, test_acc, _ = inference(self.net, self.config, self.test_loader, model_file=self.config.model_file)
            print("cls_acc={:.5f} map={:.5f}".format(test_acc*100, _mean_ap*100))

    def calculate_pesudo_target(self, batch_size, label, topk_indices):
        cls_agnostic_gt = []
        cls_agnostic_neg_gt = []
        for b in range(batch_size):
            label_indices_b = torch.nonzero(label[b, :])[:,0]
            topk_indices_b = topk_indices[b, :, label_indices_b] # topk, num_actions
            cls_agnostic_gt_b = torch.zeros((1, 1, self.config.num_segments)).cuda()

            # positive examples
            for gt_i in range(len(label_indices_b)):
                cls_agnostic_gt_b[0, 0, topk_indices_b[:, gt_i]] = 1
            cls_agnostic_gt.append(cls_agnostic_gt_b)

        return torch.cat(cls_agnostic_gt, dim=0)  # B, 1, num_segments
    
    def calculate_pesudo_target2(self, batch_size, label, topk_indices):
        cls_agnostic_gt = []
        cls_agnostic_neg_gt = []
        for b in range(batch_size):
            label_indices_b = torch.nonzero(label[b, :])[:,0]
            topk_indices_b = topk_indices[b, :, label_indices_b] # topk, num_actions
            cls_agnostic_gt_b = torch.zeros((1, 1, self.config.num_segments//3)).cuda()

            # positive examples
            for gt_i in range(len(label_indices_b)):
                cls_agnostic_gt_b[0, 0, topk_indices_b[:, gt_i]] = 1
            cls_agnostic_gt.append(cls_agnostic_gt_b)

        return torch.cat(cls_agnostic_gt, dim=0)  # B, 1, num_segments

    def calculate_pesudo_target1(self, batch_size, topk_indices):
        cls_agnostic_gt = []
        for b in range(batch_size):
            topk_indices_b = topk_indices[b, :]          # [75, 1]
            cls_agnostic_gt_b = torch.zeros((1, 1, self.config.num_segments)).cuda()

            cls_agnostic_gt_b[0, 0, topk_indices_b[:, 0]] = 1
            cls_agnostic_gt.append(cls_agnostic_gt_b)

        return torch.cat(cls_agnostic_gt, dim=0)         # B, 1, num_segments
        
    def evaluate(self, epoch=0):
        if self.step % self.config.detection_inf_step == 0:
            self.total_loss_per_epoch /= self.config.detection_inf_step

            with torch.no_grad():
                self.net = self.net.eval()
                mean_ap, test_acc, tiou_mAP = inference(self.net, self.config, self.test_loader, model_file=None)
                self.net = self.net.train()

            if mean_ap > self.best_mAP:
                self.best_mAP = mean_ap

                torch.save(self.net.state_dict(), os.path.join(self.config.model_path, "CAS_Only.pkl"))
                # torch.save(self.best_mAP, os.path.join(self.config.model_path, "best_mAP.txt"))
                with open(os.path.join(self.config.model_path, "best_mAP.txt"), 'a') as file:
                    write_str = '%f\n' % (self.best_mAP)
                    file.write(write_str)

            if test_acc > self.best_acc:
                self.best_acc = test_acc

            print("epoch={:5d} step={:5d} Loss={:.4f} cls_acc={:5.2f} best_acc={:5.2f} mean_ap={:5.2f} best_map={:5.2f}".format(
                    epoch, self.step, self.total_loss_per_epoch, test_acc * 100, self.best_acc * 100, mean_ap*100, self.best_mAP * 100))

            self.total_loss_per_epoch = 0
    

    def train(self):
        # resume training
        load_weight(self.net, self.config)

        # training
        # l = []
        for epoch in range(self.config.num_epochs):
            # start_time = time.time()
            for _data, _label, temp_anno, vid_name, vid_num_seg in self.train_loader:
                batch_size = _data.shape[0]
                _data, _label = _data.cuda(), _label.cuda()
                self.optimizer.zero_grad()
                # forward pass
                cas, cas_bg, action_flow, action_rgb, action_flow_aug, action_rgb_aug, sc, sc_flow, sc_rgb = self.net(_data, self.config)
                combined_cas = misc_utils.instance_selection_function(torch.softmax(cas, -1), action_flow.permute(0, 2, 1), action_rgb.permute(0, 2, 1)) 
                _, topk_indices = torch.topk(combined_cas, self.config.num_segments // self.config.r, dim=1)  # //8  torch.softmax(cas, -1)
                cas_top = torch.mean(torch.gather(cas, 1, topk_indices), dim=1)   #
                action = ((1-self.config.hp_lambda)*action_flow + self.config.hp_lambda*action_rgb).permute(0, 2, 1)
                bg = 1 - action

                _, topk_indices_bg = torch.topk(torch.softmax(cas_bg,-1), self.config.num_segments // self.config.r, dim=1)  # //8  torch.softmax(cas, -1)
                cas_top_bg = torch.mean(torch.gather(cas_bg, 1, topk_indices_bg), dim=1)   
               
                _, topk_indices_agno = torch.topk(action, self.config.num_segments // self.config.r + 17, dim=1)  #
                topk_indices_agno = topk_indices_agno.repeat(1, 1, topk_indices.shape[2])   
                cls_agnostic_gt = self.calculate_pesudo_target(batch_size, _label, topk_indices_agno)

                confidence = torch.sum(torch.relu(cas), dim=-1, keepdim=True)
                _, topk_indices_conf = torch.topk(confidence, self.config.num_segments // 2, dim=1)
                conf_gt = self.calculate_pesudo_target1(batch_size, topk_indices_conf)
                cls_agnostic_gt = cls_agnostic_gt * conf_gt  
                
                self.update_prob_t(torch.softmax(cas,dim=-1).reshape(-1, cas.shape[-1]))
                max_probs, mask = self.calculate_mask(torch.softmax(cas,dim=-1).reshape(-1, cas.shape[-1]))
                mask = mask.reshape(cas.shape[0], -1)
                mask_aug = (mask + torch.cat((mask[-1:], mask[:-1]),dim=0))/2

                cls_agnostic_gt_aug = (cls_agnostic_gt + torch.cat((cls_agnostic_gt[-1:], cls_agnostic_gt[:-1]),dim=0))
                cls_agnostic_gt_aug = torch.where(cls_agnostic_gt_aug>=1, 1, 0)

                # losses
                mil_loss = self.criterion(cas_top, _label)    
                label_bg = torch.ones_like(_label).cuda()

                cas_top_bg = torch.softmax(cas_top_bg, -1)
                label_bg = torch.softmax(label_bg, -1)
                bg_loss = 10*F.kl_div(cas_top_bg.log(), label_bg, reduction='batchmean')
                
                act_loss = self.Lgce_mask(action_flow.squeeze(1), cls_agnostic_gt.squeeze(1), mask) + self.Lgce_mask(action_rgb.squeeze(1), cls_agnostic_gt.squeeze(1),mask)
                mix_loss = self.Lgce_mask(action_flow_aug.squeeze(1), cls_agnostic_gt_aug.squeeze(1), mask_aug) + self.Lgce_mask(action_rgb_aug.squeeze(1), cls_agnostic_gt_aug.squeeze(1), mask_aug)                
                mc_loss = 5*F.mse_loss(action_rgb, action_flow)
                se_loss = self.criterion(sc, _label) + self.criterion(sc_flow, _label) + self.criterion(sc_rgb, _label) 
                cost =  mil_loss + mc_loss + bg_loss + act_loss + 3*se_loss + mix_loss    # 3   

                cost.backward()
                self.optimizer.step()
                self.total_loss_per_epoch += cost.cpu().item()
                self.step += 1

                # evaluation
                if epoch > 0:               
                    self.evaluate(epoch=epoch)


def main():
    args = parse_args()
    config = Config(args)
    set_seed(config)

    trainer = ThumosTrainer(config)

    if args.inference_only:
        trainer.test()
    else:
        trainer.train()


if __name__ == '__main__':
    main()

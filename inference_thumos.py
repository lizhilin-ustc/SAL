import os
import json
import torch
from tqdm import tqdm
import numpy as np
from utils import misc_utils
import torch.nn.functional as F
from run_eval import evaluate
import matplotlib.pyplot as plt
import scipy.special
import pickle
from collections import defaultdict
from config.config_thumos import class_dict
from scipy.interpolate import interp1d
import time


def load_weight(model_file, net):
    if model_file is not None:
        print("loading from file: ", model_file)
        net.load_state_dict(torch.load(model_file), strict=False)


def inference(net, config, test_loader, model_file=None):
    np.set_printoptions(formatter={'float_kind': "{:.4f}".format})

    with torch.no_grad():
        net.eval()

        # load weights
        load_weight(model_file, net)

        final_res = {'version': 'VERSION 1.3', 'results': {},
                     'external_data': {'used': True, 'details': 'Features from I3D Network'}}

        num_correct = 0.
        num_total = 0.

        s = time.time()
        for _data, _label, temp_anno, vid_name, vid_num_seg in test_loader:
            
            batch_size = _data.shape[0]
            vid = vid_name[0]
            _data = _data.cuda()
            _label = _label.cuda()
            cas, cas_bg, action_flow, action_rgb, action_flow_aug, action_rgb_aug, sc, sc_flow, sc_rgb = net(_data, config) 
            combined_cas = misc_utils.instance_selection_function(torch.softmax(cas, -1), action_flow.permute(0, 2, 1), action_rgb.permute(0, 2, 1))
            _, topk_indices = torch.topk(combined_cas, config.num_segments // config.r, dim=1)    # combined_cas cas_supp
            beta = 0.35  
            # class prediction
            cas_top = torch.gather(cas, 1, topk_indices)
            cas_top = torch.mean(cas_top, dim=1)
            cas_top = beta*cas_top + (1-beta)*sc
            score_supp = F.softmax(cas_top, dim=1)

            label_np = _label.cpu().numpy()
            score_np = score_supp[0, :].cpu().numpy()

            score_np[np.where(score_np < config.class_thresh)] = 0
            score_np[np.where(score_np >= config.class_thresh)] = 1

            if np.all(score_np == 0):
                arg = np.argmax(score_supp[0, :].cpu().data.numpy())
                score_np[arg] = 1

            correct_pred = np.sum(label_np == score_np, axis=1)
            num_correct += np.sum((correct_pred == config.num_classes).astype(np.float32))
            num_total += correct_pred.shape[0]
            pred = np.where(score_np > config.class_thresh)[0]

            # action prediction
            if len(pred) != 0:
                cas_pred = combined_cas[0].cpu().numpy()[:, pred] 
                cas_pred = np.reshape(cas_pred, (config.num_segments, -1, 1))      
                cas_pred = misc_utils.upgrade_resolution(cas_pred, config.scale)   

                proposal_dict = {}

                for t in range(len(config.act_thresh)):
                    cas_temp = cas_pred.copy()
                    zero_location = np.where(cas_temp[:, :, 0] < config.act_thresh[t])
                    cas_temp[zero_location] = 0

                    cas_input = cas_pred.copy()

                    seg_list = []
                    for c in range(len(pred)):
                        pos = np.where(cas_temp[:, c, 0] > 0)
                        seg_list.append(pos)

                    proposals = misc_utils.get_proposal_oic(seg_list, cas_pred.copy(), score_supp[0, :].cpu().data.numpy(),
                                                            pred, config.scale, vid_num_seg[0].cpu().item(), config.feature_fps,
                                                            config.num_segments, config.gamma)

                    for j in range(len(proposals)):
                        if not proposals[j]:
                            continue
                        class_id = proposals[j][0][0]

                        if class_id not in proposal_dict.keys():
                            proposal_dict[class_id] = []

                        proposal_dict[class_id] += proposals[j]

                final_proposals = []
                for class_id in proposal_dict.keys():
                    final_proposals.append(misc_utils.basnet_nms(proposal_dict[class_id], config.nms_thresh,
                                                                 config.soft_nms, config.nms_alpha))

                final_res['results'][vid_name[0]] = misc_utils.result2json(final_proposals)
        t = time.time() - s
        print(t)
        test_acc = num_correct / num_total

        json_path = os.path.join(config.model_path, 'temp_result.json')

        with open(json_path, 'w') as f:
            json.dump(final_res, f)
            f.close()

        mean_ap, tiou_mAp = evaluate(config.gt_path, json_path, None, tiou_thresholds=np.linspace(0.1, 0.7, 7), plot=False,
                                     subset='test', verbose=config.verbose)
        return mean_ap, test_acc, tiou_mAp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
import math

#### 50.6

class Model(nn.Module):
    def __init__(self, len_feature, num_classes, num_segments, pooling_type=""):
        super(Model, self).__init__()
        self.len_feature = len_feature
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5), # 0.5
            # nn.Conv1d(in_channels=512, out_channels=self.num_classes, kernel_size=1, padding=0)
        )

        self.action_rgb = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5), # 0.5
            # nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0),
        )

        self.action_flow = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5), # 0.5
            # nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0),
        )

        self.cls = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=self.num_classes, kernel_size=1, padding=0)
        )

        self.cls_rgb = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0),
        )

        self.cls_flow = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0),
        )

        self.fc = torch.nn.Linear(512**2, self.num_classes)
        # torch.nn.init.kaiming_normal_(self.fc.weight.data)
        # if self.fc.bias is not None:
        #     torch.nn.init.constant_(self.fc.bias.data, val=0)

        self.fc1 = torch.nn.Linear(512**2, 512)
        torch.nn.init.kaiming_normal_(self.fc1.weight.data)
        if self.fc1.bias is not None:
            torch.nn.init.constant_(self.fc1.bias.data, val=0)

    def BilinearPooling(self, A, B):
        C = torch.bmm(A, torch.transpose(B,1,2)) / (A.shape[2]**2)
        C = C.view(A.shape[0], A.shape[1]**2)
        C = torch.sign(C) * torch.sqrt(torch.abs(C) + 1e-5)
        C = F.normalize(C)
        return C

    def forward(self, x, config):
        input = x.permute(0, 2, 1)  # x.shape=[16,750,2048]     input.shape=[16,2048,750]
        # input = self.add_gaussian_noise(input, mean=0, std=0.15)
        flow = input[:, 1024:, :]
        rgb = input[:, :1024, :]

        flow_aug = (flow + torch.cat((flow[-1:], flow[:-1]),dim=0))
        rgb_aug = (rgb + torch.cat((rgb[-1:], rgb[:-1]), dim=0))

        emb_flow = self.action_flow(flow)
        emb_rgb = self.action_rgb(rgb)
        emb_flow_aug = self.action_flow(flow_aug)
        emb_rgb_aug = self.action_rgb(rgb_aug)
        # emb_flow_aug1 = self.action_flow(flow_aug1)
        # emb_rgb_aug1 = self.action_rgb(rgb_aug1)

        action_flow = torch.sigmoid(self.cls_flow(emb_flow))
        action_rgb = torch.sigmoid(self.cls_rgb(emb_rgb))
        action_flow_aug = torch.sigmoid(self.cls_flow(emb_flow_aug))
        action_rgb_aug = torch.sigmoid(self.cls_rgb(emb_rgb_aug))
        # action = 0.9*action_flow + 0.1*action_rgb    # [B, 1, T]
        action = (1-config.hp_lambda)*action_flow + config.hp_lambda*action_rgb    # [B, 1, T]

        emb = self.encoder(input)

        sc = self.BilinearPooling(emb, emb)
        # import pdb; pdb.set_trace()

        semantic = self.fc1(sc).unsqueeze(-1)  # [B,D,1]

        sema_act = torch.einsum("bkt,bdk->bdt", [action, semantic])
        # norm_fact = 1 / math.sqrt(semantic.shape[-1])
        # sema_act = torch.softmax(sema_act, dim=-1) * norm_fact

        emb = emb + sema_act 
        
        sc = self.fc(sc)

        sc_rgb = self.BilinearPooling(emb_rgb, emb_rgb)
        sc_rgb = self.fc(sc_rgb)

        sc_flow = self.BilinearPooling(emb_flow, emb_flow)
        sc_flow = self.fc(sc_flow)

        cas = self.cls(emb).permute(0, 2, 1)

        bg = 1 - action    
        cas_bg = cas * bg.permute(0,2,1) 

        return cas,cas_bg, action_flow, action_rgb, action_flow_aug, action_rgb_aug, sc, sc_flow, sc_rgb

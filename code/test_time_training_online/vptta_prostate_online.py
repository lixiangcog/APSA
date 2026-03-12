
import argparse
import time
import os
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd # 🔥 新增: 用于读取 CSV
from torch.utils.data import DataLoader
from datasets.dataloaders.RIGA_dataloader import RIGA_labeled_set # 🔥 新增: 引用数据加载器

from test_time_training_online.base_tta import BaseAdapter
from utils.init import init_random_and_cudnn, Recorder
from utils.file_utils import *
from models import *
from utils.metrics.dice import get_dice_assd_hd
from utils.tools import AverageMeter
from loss_functions.warm_up_bn_loss import warm_up_bn_loss
from models.PromptTTA.vptta import UNet_BN
from models.Official_VPTTA.vptta import *

class VPTTA(BaseAdapter):
    def __init__(self, args):
        super(VPTTA, self).__init__(args)
        print('device:', self.device, 'gpus', self.gpus)
        
        # 🔥 [Auto-Fix] 自动判断 Dataset Name
        csv_str = str(self.args.ts_csv)
        if "BMC" in csv_str: self.dataset_name = "BMC"
        elif "RUNMC" in csv_str: self.dataset_name = "RUNMC"
        elif "UCL" in csv_str: self.dataset_name = "UCL"
        elif "HK" in csv_str: self.dataset_name = "HK"
        elif "BIDMC" in csv_str: self.dataset_name = "BIDMC"
        elif "I2CVB" in csv_str: self.dataset_name = "I2CVB"
        else: self.dataset_name = "Prostate" # 默认兜底
        print(f"Detected Dataset Name: {self.dataset_name}")

        # Model
        self.backbone = args.backbone
        self.in_ch = args.in_ch
        self.out_ch = args.out_ch

        # Warm-up
        self.warm_n = args.warm_n

        # Prompt
        self.prompt_alpha = args.prompt_alpha
        self.iters = args.iters

        # Memory Bank
        self.neighbor = args.neighbor
        self.memory_bank = Memory(size=args.memory_size, dimension=self.prompt.data_prompt.numel())

    # 🔥 [Fix] 补上缺失的数据加载函数 (和之前修复的一样)
    def init_dataloader(self):
        csv_path = self.args.ts_csv[0] if isinstance(self.args.ts_csv, list) else self.args.ts_csv
        print(f"Loading data from CSV: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path, skipinitialspace=True)
            image_list = df['image'].tolist()
            # 尝试读取 Mask
            if'mask' in df.columns: mask_list = df['mask'].tolist()
            elif ' mask' in df.columns: mask_list = df[' mask'].tolist()
            else: mask_list = df.iloc[:, 1].tolist()

            print(f"Found {len(image_list)} images.")
            # 使用 batch_size=1 避免堆叠错误
            self.test_set = RIGA_labeled_set(self.args.root, image_list, mask_list, self.args.patch_size)
            self.ts_dataloader = DataLoader(self.test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            raise e
        return self.ts_dataloader

    def init_from_source_model(self):
        self.prompt = Prompt(prompt_alpha=self.args.prompt_alpha, image_size=self.args.patch_size[0]).to(self.device)
        self.pretrained_params = torch.load(self.args.pretrained_model)
        self.model = UNet_BN(pretrained_path=self.args.pretrained_model, patch_size=self.args.patch_size, resnet='resnet34', num_classes=self.args.out_ch).to(self.device)
        return self.model

    def setup_model_w_optimizer(self, base_model):
        if self.args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.prompt.parameters(), lr=self.args.lr, momentum=self.args.momentum, nesterov=True, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.prompt.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2), weight_decay=self.args.weight_decay)
        return base_model

    def print_prompt(self):
        num_params = sum(p.numel() for p in self.prompt.parameters())
        print("The number of total parameters: {}".format(num_params))

    def do_prostate_seg_tta_per_case(self, data):
        # 强制 batch_size=1
        batch_size = 1
        num_samples = data.shape[0]
        
        if self.args.model_episodic:
            base_model = self.init_from_source_model()
            self.model = self.setup_model_w_optimizer(base_model)
      
        train_loss_list = list()
        preds = []
        
        # 逐切片处理 (Slice by Slice or Batch by Batch)
        for i in range(0, num_samples, batch_size):
            batch_data = data[i:i+batch_size, :, :, :]
            
            self.model.eval()
            self.prompt.train()

            # Initialize Prompt
            if len(self.memory_bank.memory.keys()) >= self.neighbor:
                _, low_freq = self.prompt(batch_data)
                init_data, score = self.memory_bank.get_neighbours(keys=low_freq.cpu().numpy(), k=self.neighbor)
            else:
                init_data = torch.ones((1, 3, self.prompt.prompt_size, self.prompt.prompt_size)).to(self.device)
            self.prompt.update(init_data)

            # Train Prompt
            for tr_iter in range(self.iters):
                prompt_x, _ = self.prompt(batch_data)
                output, bn_f, _ = self.model(prompt_x, training=True)
                loss = warm_up_bn_loss(self.model, self.pretrained_params, bn_f=bn_f, index=i)
                train_loss_list.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Inference
            self.model.eval()
            self.prompt.eval()
            with torch.no_grad():
                prompt_x, low_freq = self.prompt(batch_data)
                pred_logit = self.model(prompt_x)
            preds.append(pred_logit) 

            # Update Memory Bank
            self.memory_bank.push(keys=low_freq.cpu().numpy(), logits=self.prompt.data_prompt.detach().cpu().numpy())

        # 打印平均 Loss
        if train_loss_list:
            self.logger.info('  loss:{:.4f}'.format(np.mean(train_loss_list)))
    
        # Final Inference (One pass)
        self.model.eval()
        self.prompt.eval()
        preds_after_one_case = [] 
        with torch.no_grad(): 
             for i in range(0, num_samples, batch_size):
                batch_data = data[i:i+batch_size, :, :, :]
                prompt_x, low_freq = self.prompt(batch_data)
                output = self.model(prompt_x)
                preds_after_one_case.append(output)
                
        full_pred = torch.cat(preds, dim=0)
        full_pred = torch.sigmoid(full_pred)
        full_pred = (full_pred > 0.5).float()

        full_pred_final = torch.cat(preds_after_one_case, dim=0)
        full_pred_final = torch.sigmoid(full_pred_final)
        full_pred_final = (full_pred_final > 0.5).float()
        return full_pred, full_pred_final

    def evaluate_prostate_seg(self):
        case_name_all = []
        Dice_Metrics = AverageMeter(save_all=True)
        HD_Metrics = AverageMeter(save_all=True)
        ASSD_Metrics = AverageMeter(save_all=True)
        
        Dice_Metrics_final = AverageMeter(save_all=True)
        HD_Metrics_final = AverageMeter(save_all=True)
        ASSD_Metrics_final = AverageMeter(save_all=True)

        for iter, batch in enumerate(self.ts_dataloader):
            data, seg, case_name = batch
            data = data[0].to(self.device)
            # 确保标签也是 Batch=1，并转为 numpy
            target_array = seg[0].squeeze().numpy()
             
            start = time.time()
            # 输入数据维度调整: (D, H, W) -> (D, 3, H, W) 适配网络输入
            pred_aggregated, pred_aggregated_final = self.do_prostate_seg_tta_per_case(data.unsqueeze(1).repeat(1, 3, 1, 1))
            end = time.time()
            
            case_name_all.append(case_name[0])
            self.logger.info('  Case {}  TTA time:{:.2f}s'.format(case_name[0], end-start))
            
            # 转换预测结果
            pred_array = pred_aggregated.detach().squeeze().cpu().numpy()
            pred_array_final = pred_aggregated_final.detach().squeeze().cpu().numpy()

            # 计算指标
            dc, assd, hd = get_dice_assd_hd(target_array, pred_array, self.logger)
            dc_final, assd_final, hd_final = get_dice_assd_hd(target_array, pred_array_final, self.logger)

            Dice_Metrics.update(dc)
            HD_Metrics.update(hd)
            ASSD_Metrics.update(assd)

            Dice_Metrics_final.update(dc_final)
            HD_Metrics_final.update(hd_final)
            ASSD_Metrics_final.update(assd_final)

            self.logger.info("Cur-patient {}  dice:{:.3f} final:{:.3f}".format(case_name[0], dc, dc_final))
                        
        avg_dc = Dice_Metrics.avg
        avg_dc_final = Dice_Metrics_final.avg
        
        self.logger.info(" Test Avg- dice:{:.4f}".format(avg_dc))
        self.logger.info(" Test Avg- dice_final:{:.4f}".format(avg_dc_final))

    def evaluate(self):
        # 统一入口
        self.evaluate_prostate_seg()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ... (参数定义省略，反正下面会被覆盖)
    parser.add_argument('--model', default="VPTTA")
    parser.add_argument('--gpu', type=int, nargs='+', default=[0])
    parser.add_argument('--log_folder', default='log_dir')
    parser.add_argument('--tag', default="VPTTA_Run")
    parser.add_argument('--root', default='data/MRI_prostate')
    parser.add_argument('--ts_csv', nargs='+', default=['data/MRI_prostate/BMC_test.csv'])
    parser.add_argument('--pretrained_model', default='log_dir/UNet_Source_Prostate/SiteA_RUNMC_batch_aug/checkpoints/model_final.model')
    
    # VPTTA 特定参数
    parser.add_argument('--prompt_alpha', type=float, default=0.01)
    parser.add_argument('--iters', type=int, default=10) # 默认跑10步
    parser.add_argument('--memory_size', type=int, default=40)
    parser.add_argument('--neighbor', type=int, default=16)
    parser.add_argument('--warm_n', type=int, default=5)
    
    # 通用参数
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[512, 512])
    parser.add_argument('--batch_size', type=int, default=1) # 强制为1
    parser.add_argument('--backbone', type=str, default='resnet34')
    parser.add_argument('--in_ch', type=int, default=3)
    parser.add_argument('--out_ch', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.00)
    parser.add_argument('--model_episodic', type=bool, default=False)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--momentum', type=float, default=0.99)

    args = parser.parse_args()
    
    # 强制覆盖为 1，防止报错
    args.batch_size = 1 
    
    adpater = VPTTA(args)
    adpater.evaluate()

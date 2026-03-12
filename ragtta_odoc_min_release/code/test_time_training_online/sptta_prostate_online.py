"""
sptta_prostate_online.py - V4
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from utils.init import init_random_and_cudnn, Recorder
from time import time
from torch.utils.tensorboard import SummaryWriter
from utils.file_utils import *
import argparse
from test_time_training_online.base_tta_myvida import BaseAdapter

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models import *
from utils.metrics.dice import get_dice_assd_hd
from utils.tools import AverageMeter
from models.PromptTTA.ours_ema import *
from torch.utils.data import DataLoader
from datasets.dataloaders.Prostate_dataloader import Prostate_labeled_set
from datasets.utils.convert_csv_to_list import convert_labeled_list


def entropy_loss_sigmoid(p):
    p = torch.sigmoid(p).clamp(1e-6, 1 - 1e-6)
    return (-p * torch.log(p) - (1 - p) * torch.log(1 - p)).mean()


class Outer_Adapter(BaseAdapter):
    def __init__(self, args):
        super(Outer_Adapter, self).__init__(args)
        print('device:', self.device, 'gpus', self.gpus)
        self.all_steps = 0
        # Prefer args (avoids hard-coded absolute paths)
        self.clip_prior_root_3d = getattr(self.args, "clip_prior_root_3d", "data/MRI_prostate/clip_priors_3d")
        self.clip_prior_root_2d = getattr(self.args, "clip_prior_root_2d", "data/MRI_prostate/clip_priors")

        csv_str = str(self.args.ts_csv)
        for name in ["BMC", "RUNMC", "UCL", "HK", "BIDMC", "I2CVB"]:
            if name in csv_str:
                self.dataset_name = name
                break
        else:
            self.dataset_name = "BMC"
        print(f"Detected Dataset Name: {self.dataset_name}")

    def init_dataloader(self):
        ts_img_list, ts_label_list = convert_labeled_list(self.args.ts_csv, r=-1)
        self.test_set = Prostate_labeled_set(self.args.root, ts_img_list, ts_label_list, 'test3d', self.args.patch_size, img_normalize=True)
        self.ts_dataloader = DataLoader(self.test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        return self.ts_dataloader

    def _resize_prior_2d(self, soft_map_2d, H, W):
        sm = torch.from_numpy(soft_map_2d).float().to(self.device)
        sm = F.interpolate(sm[None, None], (H, W), mode="bilinear", align_corners=False)[0]
        return sm.clamp(0, 1)

    def load_clip_prior(self, case_name, data_shape):
        """
        Loads CLIP prior.
        - If 3D prior exists: use per-slice soft_map.
        - If only 2D prior exists (mid-slice): DO NOT broadcast to all slices; apply only to a small window.
        """
        D, H, W = int(data_shape[0]), int(data_shape[-2]), int(data_shape[-1])
        clean_name = os.path.basename(str(case_name)).replace(".nii.gz", "").split(".")[0]

        # ---- 3D prior (preferred) ----
        # Support both:
        # - legacy pickled .npy (may be incompatible across numpy versions)
        # - robust .npz (recommended)
        path_3d_npy = os.path.join(self.clip_prior_root_3d, self.dataset_name, f"{clean_name}_clip.npy")
        path_3d_npz = os.path.join(self.clip_prior_root_3d, self.dataset_name, f"{clean_name}_clip.npz")
        for path_3d in [path_3d_npz, path_3d_npy]:
            if not os.path.exists(path_3d):
                continue
            try:
                if path_3d.endswith(".npz"):
                    data = np.load(path_3d)
                    soft_map = data.get("soft_map", None)
                else:
                    data = np.load(path_3d, allow_pickle=True).item()
                    soft_map = data.get("soft_map", None)
                if soft_map is not None and getattr(soft_map, "ndim", 0) == 3:
                    clip_volume = torch.zeros((D, 1, H, W), device=self.device)
                    d0 = min(D, soft_map.shape[0])
                    for si in range(d0):
                        sm = self._resize_prior_2d(soft_map[si], H, W)
                        clip_volume[si] = sm
                    clip_volume = clip_volume.clamp(0, 1)
                    self.logger.info(f"[CLIP] 3D prior loaded: {path_3d} | shape={tuple(soft_map.shape)}")
                    return clip_volume
            except Exception as e:
                self.logger.info(f"[CLIP] 3D prior load error: {path_3d} | {e}")

        # ---- 2D prior (mid-slice) ----
        path_2d_npy = os.path.join(self.clip_prior_root_2d, self.dataset_name, f"{clean_name}_clip.npy")
        path_2d_npz = os.path.join(self.clip_prior_root_2d, self.dataset_name, f"{clean_name}_clip.npz")
        for path_2d in [path_2d_npz, path_2d_npy]:
            if not os.path.exists(path_2d):
                continue
            try:
                if path_2d.endswith(".npz"):
                    data = np.load(path_2d)
                    soft_map = data.get("soft_map", None)
                    slice_idx = int(data.get("slice_idx", D // 2))
                else:
                    data = np.load(path_2d, allow_pickle=True).item()
                    soft_map = data.get("soft_map", None)
                    slice_idx = int(data.get("slice_idx", D // 2))

                if soft_map is not None and getattr(soft_map, "ndim", 0) == 2:
                    sm = self._resize_prior_2d(soft_map, H, W)  # [1,H,W]
                    clip_volume = torch.zeros((D, 1, H, W), device=self.device)
                    radius = int(getattr(self.args, "clip_slice_radius", 2))
                    s0 = max(0, slice_idx - radius)
                    s1 = min(D, slice_idx + radius + 1)
                    clip_volume[s0:s1] = sm
                    clip_volume = clip_volume.clamp(0, 1)
                    self.logger.info(f"[CLIP] 2D prior loaded: {path_2d} | slice_idx={slice_idx}, radius={radius}")
                    return clip_volume
            except Exception as e:
                self.logger.info(f"[CLIP] 2D prior load error: {path_2d} | {e}")

        self.logger.info(f"[CLIP] No prior found for {clean_name} (dataset={self.dataset_name})")
        return None

    def compute_clip_loss(self, pred_logits, clip_prior):
        if clip_prior is None:
            return torch.tensor(0.0, device=self.device)
        pred_prob = torch.sigmoid(pred_logits)
        high_q = float(getattr(self.args, "clip_high_quantile", 0.90))
        min_thresh = float(getattr(self.args, "clip_min_thresh", 0.3))
        high_thresh = max(min_thresh, float(clip_prior.quantile(high_q).item()))
        high_mask = (clip_prior > high_thresh).float()

        min_pixels = float(getattr(self.args, "clip_min_pixels", 50))
        if high_mask.sum() > min_pixels:
            focal_weight = (1 - pred_prob) ** 2
            return (focal_weight * high_mask * (1 - pred_prob)).sum() / (high_mask.sum() + 1e-6)
        return ((1 - pred_prob) * clip_prior).mean()

    def do_prostate_seg_tta_per_case(self, data, case_name):
        num_samples = data.shape[0]
        num_batches = (num_samples + self.args.batch_size - 1) // self.args.batch_size
        clip_volume = self.load_clip_prior(case_name, data.shape)

        if not self.args.model_episodic:
            self.model.reset_online_network()

        preds = []
        for step in range(self.args.optim_steps):
            self.model.online_network.train()
            step_preds, losses = [], []
            total_ent, total_clip = 0.0, 0.0

            for i in range(num_batches):
                bs, be = i * self.args.batch_size, min((i + 1) * self.args.batch_size, num_samples)
                batch_data = data[bs:be]
                output = self.model.online_network(batch_data)

                loss_ent = entropy_loss_sigmoid(output)
                loss_clip = self.compute_clip_loss(output, clip_volume[bs:be]) if clip_volume is not None else torch.tensor(0.0, device=self.device)

                total_ent += loss_ent.item()
                total_clip += loss_clip.item()

                clip_w = float(getattr(self.args, "clip_loss_w", 0.05))
                all_loss = loss_ent + clip_w * loss_clip
                self.optimizer.zero_grad()
                all_loss.backward()
                self.optimizer.step()
                self.model.update_target_network()

                self.source_model.eval()
                self.model.online_network.eval()
                with torch.no_grad():
                    out_src = self.source_model(batch_data)
                    out_ada = self.model.online_network(batch_data)
                step_preds.append(out_ada if entropy_loss_sigmoid(out_ada) <= entropy_loss_sigmoid(out_src) else out_src)
                losses.append(all_loss.item())
                self.model.online_network.train()

            self.all_steps += 1
            self.logger.info(f"Step {step}: Loss={np.mean(losses):.4f} (Ent={total_ent/num_batches:.4f}, Clip={total_clip/num_batches:.4f})")
            preds = step_preds

        self.model.online_network.eval()
        preds_final = []
        with torch.no_grad():
            for i in range(num_batches):
                bs, be = i * self.args.batch_size, min((i + 1) * self.args.batch_size, num_samples)
                preds_final.append(self.model.online_network(data[bs:be]))

        def to_mask(t):
            return (torch.sigmoid(torch.cat(t, 0)) > 0.5).float()
        return to_mask(preds), to_mask(preds_final)

    def evaluate_prostate_seg(self):
        Dice, ASSD, HD = AverageMeter(True), AverageMeter(True), AverageMeter(True)
        Dice_f, ASSD_f, HD_f = AverageMeter(True), AverageMeter(True), AverageMeter(True)

        for batch in self.ts_dataloader:
            data, seg, case_name = batch
            data, seg = data[0].to(self.device), seg[0].to(self.device)

            start = time()
            pred, pred_final = self.do_prostate_seg_tta_per_case(data.unsqueeze(1).repeat(1, 3, 1, 1), case_name[0])
            self.logger.info(f'  Case {case_name[0]}  TTA time:{time()-start:.2f}s')

            pred_np = pred.squeeze().cpu().numpy()
            pred_f_np = pred_final.squeeze().cpu().numpy()
            gt_np = seg.squeeze().cpu().numpy()

            min_d = min(gt_np.shape[0], pred_np.shape[0])
            gt_np, pred_np, pred_f_np = gt_np[:min_d], pred_np[:min_d], pred_f_np[:min_d]

            dc, assd, hd = get_dice_assd_hd(gt_np, pred_np, self.logger)
            dc_f, assd_f, hd_f = get_dice_assd_hd(gt_np, pred_f_np, self.logger)

            Dice.update(dc)
            ASSD.update(assd)
            HD.update(hd)
            Dice_f.update(dc_f)
            ASSD_f.update(assd_f)
            HD_f.update(hd_f)

            self.logger.info(f"  {case_name[0]} dice:{dc:.3f} assd:{assd:.3f} hd:{hd:.3f}")
            self.logger.info(f"  {case_name[0]} dice_final:{dc_f:.3f} assd_final:{assd_f:.3f} hd_final:{hd_f:.3f}")

        self.logger.info(f" Avg dice:{Dice.avg:.4f}±{np.std(Dice.all_data):.4f} hd:{HD.avg:.4f}±{np.std(HD.all_data):.4f} assd:{ASSD.avg:.4f}±{np.std(ASSD.all_data):.4f}")
        self.logger.info(f" Avg dice_final:{Dice_f.avg:.4f}±{np.std(Dice_f.all_data):.4f} hd_final:{HD_f.avg:.4f}±{np.std(HD_f.all_data):.4f} assd_final:{ASSD_f.avg:.4f}±{np.std(ASSD_f.all_data):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="Tent-EMA")
    parser.add_argument('--arch', default="unet_2d_sptta")
    parser.add_argument('--num_classes', default=1)
    parser.add_argument('--gpu', type=int, nargs='+', default=[0])
    parser.add_argument('--manualseed', type=int, default=100)
    parser.add_argument('--log_folder', default='log_dir')
    parser.add_argument('--only_bn_updated', default=True)
    parser.add_argument('--tag', default="Base2_test")
    parser.add_argument('--patch_size', type=int, nargs='+', default=[512, 512])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--initial_lr', type=float, default=1e-2)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('-c', '--continue_training', default=False, action='store_true')
    parser.add_argument('--num_threads', type=int, default=0)
    parser.add_argument('-r', '--root', default='data/MRI_prostate')
    parser.add_argument('--ts_csv', nargs='+', default=['data/MRI_prostate/BMC_test.csv'])
    parser.add_argument('--optim_steps', type=int, default=1)
    parser.add_argument('--pretrained_model', required=False)
    parser.add_argument('--model_episodic', type=bool, default=False)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--layers', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--save_pre', default=False)
    parser.add_argument('--ema_decay', type=float, default=0.6)
    parser.add_argument('--min_momentum_constant', type=float, default=0.005)
    # CLIP prior guidance (safer defaults)
    parser.add_argument('--clip_prior_root_2d', default='data/MRI_prostate/clip_priors')
    parser.add_argument('--clip_prior_root_3d', default='data/MRI_prostate/clip_priors_3d')
    parser.add_argument('--clip_slice_radius', type=int, default=2, help='When only 2D mid-slice prior exists, apply it to [slice_idx-radius, slice_idx+radius].')
    parser.add_argument('--clip_loss_w', type=float, default=0.05, help='Weight for CLIP-guided loss term.')
    parser.add_argument('--clip_high_quantile', type=float, default=0.90, help='Quantile to define confident CLIP region.')
    parser.add_argument('--clip_min_thresh', type=float, default=0.3, help='Minimum threshold for CLIP prior to be considered confident.')
    parser.add_argument('--clip_min_pixels', type=float, default=50, help='Minimum number of confident pixels to use focal-style loss.')

    args = parser.parse_args()
    args.tag = args.tag + '_sptta'
    Outer_Adapter(args).evaluate_prostate_seg()

"""TSViT_DINOv3 inference on one PASTIS window, with the same preprocessing as training (PASTIS_segmentation_transform)"""
import os
import numpy as np
import torch
from data.PASTIS24.data_transforms import PASTIS_segmentation_transform
from models.TSViT.TSViT_dinov3 import TSViT_DINOv3
from utils.config_files_utils import read_yaml

UNK_CLASS = 19      # PASTIS void label; inference has no labels, so a dummy all-background map is passed through the transforms


class Predictor:
    def __init__(self, model, transform, model_version, device='cpu'):
        self.model, self.transform = model.eval(), transform
        self.model_version, self.device = model_version, device

    @classmethod
    def from_env(cls):
        """model from MODEL_CONFIG + CHECKPOINT_PATH; DINOV3_REPO is the local facebookresearch/dinov3 snapshot (code only)"""
        cfg = read_yaml(os.environ.get('MODEL_CONFIG', 'configs/PASTIS24/TSViT_DINOv3_fold1.yaml'))['MODEL']
        threads = int(os.environ.get('TORCH_THREADS', 0))
        if threads:
            torch.set_num_threads(threads)
        # the checkpoint holds the frozen backbone too, so the DINOv3 architecture is built without its pretrained weights file
        backbone = torch.hub.load(os.environ.get('DINOV3_REPO', 'weights/dinov3'), 'dinov3_vits16', source='local', pretrained=False)
        model = TSViT_DINOv3(cfg, backbone=backbone)
        state = torch.load(os.environ.get('CHECKPOINT_PATH', 'weights/best.pth'), map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)       # strict: every backbone and head tensor must come from the checkpoint
        return cls(model, PASTIS_segmentation_transform(cfg, is_training=False),
                   os.environ.get('MODEL_VERSION', 'tsvit-dinov3-fold1-t9'))

    def predict(self, img, doy):
        """img (T, 10, 24, 24) reflectance, doy (T,) -> class map (24, 24), mean max-softmax confidence"""
        sample = {'img': np.asarray(img, np.float32), 'doy': np.asarray(doy),
                  'labels': np.zeros((1,) + np.shape(img)[-2:], np.uint8)}
        x = self.transform(sample)['inputs'].unsqueeze(0)       # (1, 60, 24, 24, 11)
        with torch.no_grad():
            prob = self.model(x).softmax(1)[0]                  # (19, 24, 24)
        conf, pred = prob.max(0)
        return pred.numpy(), float(conf.mean())

# ====================================
# 2023.07.04
# 
#    Object copy and paste module    
# 
# @author 
#   Yewon Lim (ga06033@yonsei.ac.kr)
# @env
#   Python      3.10.9
#   NumPy       1.23.5
#   pycocotools 2.0.6
#   PyTorch     2.0.0
# ====================================

import torch as th
from torchvision.transforms.v2 import Compose, ToTensor
import torchvision.transforms.functional as F
import argparse
import cv2 
import os 
import os.path as osp
import pandas as pd 
from saicinpainting.training.trainers import load_checkpoint

import yaml
from omegaconf import OmegaConf
from torchvision.utils import save_image
from torchvision.io import read_image
import torchvision.transforms as T
from diffusers import DDIMScheduler
import hashlib
from typing import Tuple
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import wrap_dataset_for_transforms_v2
import json 


def load_image(image_path, device, ismask=False, is768=False):
    image = read_image(image_path).cpu()
    if (image.shape[0] == 1) and not ismask:
        image = image.expand(3, -1, -1)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = T.Resize((512, 512))(image) if not is768 else T.Resize((768, 768))(image)
    image = image.to(device)
    return image


def parse_args():
    parser = argparse.ArgumentParser(description='Object copy and paste')
    parser.add_argument('--coco_root', type=str, help='coco dataset root dir')
    parser.add_argument('--outdir', type=str, help='save dir')
    parser.add_argument('--model_path', type=str, help='model path', default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument('--version', type=str, help='how to fill the blank region in lama', default='bbox')
    args = parser.parse_args()
    return args



@th.no_grad()
def recomposite_bbox(batch:th.Tensor, model:th.nn.Module, transform, segm_mask:th.Tensor):
    '''Parameters]
        batch: batch of images and masks for LaMa
            (dict)
        model: LaMa model
            (nn.Module)
        transform: Transformation applied to the target object.
            There should not be any randomness in transformation pipeline;

      Return]
        img: Recomposited image with the specified transformation. (numpy.ndarray)
    '''
    inpainted = model(batch)
    composited = inpainted.clone()

    subj = th.zeros_like(batch['image'])
    mask = segm_mask.to(th.bool)
    subj[..., mask] = batch['image'][..., mask]
    
    subj = transform(subj)
    tgt_mask = transform(mask.unsqueeze(0)).squeeze(0)
    
    composited[..., tgt_mask] = subj[..., tgt_mask]

    return inpainted, composited, mask, tgt_mask



class ScaleAndShift:
    def __init__(self, dx=0, dy=0, scale=1.0, rotate=0.0, center=None):
        self.dx=dx
        self.dy=dy
        self.rotate=rotate
        self.scale=scale
        self.center=center
    
    def __call__(self, x):
        img = F.affine(x, self.rotate, (self.dx, self.dy), self.scale, (0, 0), center=self.center)
        return img


def run(args):
    from torchvision.transforms import PILToTensor, Resize
    from torchvision.datasets import CocoDetection
    import os.path as osp
    import os 
    
    
    from tqdm.auto import tqdm

    

    out_dir = args.outdir
    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    
    dataset = CocoDetection(root=args.coco_root, annFile=osp.join(args.coco_root, "annotations", "instances_val2017.json"), transforms=Compose([ToTensor()]))
    dataset= wrap_dataset_for_transforms_v2(dataset, target_keys=['boxes', 'masks','image_id'])
    sampler = DistributedSampler(dataset) 
    loader = th.utils.data.DataLoader(dataset, 
                                      batch_size=1, 
                                      sampler=sampler,
                                      num_workers=args.num_workers//args.world_size,
                                      pin_memory=True,)
    train_config_path = osp.join(args.config_path)
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
        
    model = load_checkpoint(train_config, args.model_path, strict=False, map_location='cpu')
    model.freeze()
    device = th.device(f"cuda:{args.local_rank}") if th.cuda.is_available() else th.device("cpu")
    model.to(device)
    for i, (img, targets) in tqdm(enumerate(loader)):
        bbox, mask, img_id = targets['boxes'], targets['masks'], targets['image_id']
        img = img.to(device)
        bbox = bbox.to(device)
        mask = mask.to(device)
        
        """
        # CROP AND TRANSFORM
        """
        center = ((minx+maxx)/2, (miny+maxy)/2)
        w, h = mask.shape
        DX = th.randint(-minx, w-maxx, (1,)).item()
        DY = th.randint(-miny, h-maxy, (1,)).item()
        # SCALE = th.rand(1).item()*2+0.5
        SCALE = 1.
        ROTATE = th.randint(-30, 30, (1,)).item()
        transform = ScaleAndShift(DX, DY, SCALE, ROTATE, center=center)

        fname = f'{img}_{mask}_{DX}_{DY}_{SCALE}_{ROTATE}'
        m = hashlib.md5()
        m.update(fname.encode('utf-8'))    
        fname = m.hexdigest()[:8] + ".png"
        """
        # make batch 
        batch = None 
        transform = None

        if (args.version == 'noise'):
            raise NotImplementedError
        elif (args.version == 'bbox'):
            result, guide_mask = recomposite_bbox(batch, transform, model)
        elif (args.version == 'zero'):
            raise NotImplementedError
        elif (args.version == 'copy'):
            raise NotImplementedError


        else:
            raise AssertionError(f"Invalid method to fill the blank region; got {args.version}")
        
        
        for i, guide in enumerate(guide_mask):
            guide.save(osp.join(out_dir, f"guide_mask{i}", fname))
        
    

    


if __name__ == "__main__":
    import pandas as pd 

    args = parse_args()
    df = pd.read_csv(args.anns)
    run(args, (df, 0))
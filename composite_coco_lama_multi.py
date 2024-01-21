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
import torch.distributed as dist
from einops import repeat
import cv2 
import os 
import os.path as osp
import pandas as pd 
from saicinpainting.training.trainers import load_checkpoint
from coco_inpaint_dataset import CoCoInpaintDataset
import yaml
from omegaconf import OmegaConf
from torchvision.utils import save_image
from torchvision.io import read_image
import torchvision.transforms as T
#from diffusers import DDIMScheduler
import hashlib
from typing import Tuple
from torch.utils.data.distributed import DistributedSampler
#from torchvision.datasets import wrap_dataset_for_transforms_v2
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
    parser.add_argument('--batch', type=int, help='batch size', default=128)
    parser.add_argument('--coco_ann_root', type=str, help='coco annotation root dir')
    parser.add_argument('--outdir', type=str, help='save dir')
    parser.add_argument('--model_path', type=str, help='model path', required=True)
    parser.add_argument('--config_path', type=str, help='config path', required=True)
    parser.add_argument('--version', type=str, help='how to fill the blank region in lama', default='bbox')
    args = parser.parse_args()
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.num_workers = 4
    return args

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print



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
    inpainted = model(batch)['inpainted']
    composited = inpainted.clone()

    subj = th.zeros_like(batch['image'])
    mask = repeat(segm_mask.to(th.bool), 'b 1 h w -> b c h w', c=3)
    subj[mask] = batch['image'][mask]
    
    subj = transform(subj)
    tgt_mask = transform(mask)
    
    composited[..., tgt_mask] = subj[..., tgt_mask]

    return inpainted, composited, mask[:, 0, :], tgt_mask[:, 0, :]



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
    
    if 'train' in args.coco_root:
        dataset = CocoDetection(root=args.coco_root, annFile=osp.join(args.coco_ann_root,  "instances_train2017.json"), transforms=Compose([ToTensor()]))
    elif 'val' in args.coco_root:   
        dataset = CocoDetection(root=args.coco_root, annFile=osp.join(args.coco_ann_root,  "instances_val2017.json"), transforms=Compose([ToTensor()]))
    dataset = CoCoInpaintDataset(dataset)
    sampler = DistributedSampler(dataset) 
    loader = th.utils.data.DataLoader(dataset, 
                                      batch_size=args.batch, 
                                      sampler=sampler,
                                      num_workers=args.num_workers//args.world_size,
                                      pin_memory=True,)
    with open(args.config_path, "r") as f:
        predict_config = OmegaConf.create(yaml.safe_load(f))

    train_config_path = osp.join(predict_config.model.path, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    model = load_checkpoint(train_config, args.model_path, strict=False, map_location='cpu')
    model.freeze()
    device = th.device(f"cuda:{args.local_rank}") if th.cuda.is_available() else th.device("cpu")
    model.to(device)
    for i, (img, targets) in tqdm(enumerate(loader)):
        img = img.to(device)
        img_ids = targets['image_id'].tolist()
        mask = targets['masks'].to(device)
        bbox_mask = targets['bbox'].to(device)

        boxes = targets['boxes']
       
        # CROP AND TRANSFORM
        minx = boxes[:, 0].min().item()
        miny = boxes[:, 1].min().item()
        maxx = boxes[:, 2].max().item()
        maxy = boxes[:, 3].max().item()
        center = ((minx+maxx)/2, (miny+maxy)/2)
        w, h = mask.shape[2:]
        DX = th.randint(-w//4, w//4, (1,)).item()
        DY = th.randint(-h//4, h//4, (1,)).item()
        SCALE = 1.
        ROTATE = th.randint(-30, 30, (1,)).item()
        transform = ScaleAndShift(DX, DY, SCALE, ROTATE, center=center)

        fname = f'{"_".join(list(map(str, img_ids)))}_{DX}_{DY}_{SCALE}_{ROTATE}'
        m = hashlib.md5()
        m.update(fname.encode('utf-8'))    
        fname = m.hexdigest()[:8] + ".png"
        
        # make batch 
        batch = {'image':img, 'mask':bbox_mask}

        if (args.version == 'noise'):
            raise NotImplementedError
        elif (args.version == 'bbox'):
            inpainted, composited, mask, tgt_mask = recomposite_bbox(batch, transform=transform, model=model, segm_mask=mask)
        
        elif (args.version == 'zero'):
            raise NotImplementedError
        elif (args.version == 'copy'):
            raise NotImplementedError

        else:
            raise AssertionError(f"Invalid method to fill the blank region; got {args.version}")
        mask = mask.float()
        tgt_mask = tgt_mask.float()
        for b in range(img.shape[0]):
            imgname = str(img_ids[b])+ "_" + str(b)
            path = osp.join(out_dir, imgname)
            if not osp.exists(path):
                os.makedirs(path)
            
            save_image(inpainted[b], osp.join(path, "inpainted.png"), nrow=1)
            save_image(composited[b], osp.join(path, "composited.png"), nrow=1)
            save_image(mask[b], osp.join(path, "mask.png"), nrow=1)
            save_image(tgt_mask[b], osp.join(path, "target_mask.png"), nrow=1)
            save_image(img[b], osp.join(path, "image.png"), nrow=1)
            with open(osp.join(path, "transformation"), "w") as f:
                f.write(f"{DX} {DY} {SCALE} {ROTATE}")

if __name__ == "__main__":
    args = parse_args()
    dist.init_process_group(backend='nccl', init_method='env://')
    # rank = dist.get_rank()
    setup_for_distributed(args.local_rank==0)

    run(args)

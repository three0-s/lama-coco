import torch as th 
from torchvision.transforms.v2 import Resize
from torchvision.transforms.functional import crop
from torchvision.ops import masks_to_boxes

class CoCoInpaintDataset :
    def __init__(self, coco_dataset):
        self.coco_dataset = coco_dataset
        _ids = self.coco_dataset.ids
        self.img_idxs = []
        self.ann_idxs = []
        for i, img_id in enumerate(_ids):
            ann_num = len(self.coco_dataset.coco.getAnnIds(img_id))
            for j in range(ann_num):
                self.img_idxs.append(i)
                self.ann_idxs.append(j)
        
    
    def __len__(self):
        return len(self.img_idxs)
    
    def __getitem__(self, idx):
        img, ann_tot = self.coco_dataset[self.img_idxs[idx]]
        ann = ann_tot[self.ann_idxs[idx]]
        ann['mask'] = th.from_numpy(self.coco_dataset.coco.annToMask(ann))
        
        bbox = masks_to_boxes(ann['mask'].unsqueeze(0)).squeeze(0)
        center = bbox[::2].sum()//2, bbox[1::2].sum()//2
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        size = int(max(w, h) * 1.5)
        transform = lambda x, center, size: Resize((256, 256))(crop(x, int(center[1]-size//2), int(center[0]-size//2), size, size))
        
        scale = 256 / size
        cx, cy = size//2, size//2 
        
        img = transform(img, center, size)
        mask = transform(ann["mask"].unsqueeze(0), center, size).squeeze(0)

        cx, cy, w, h = list(map(lambda x: int(x * scale), [cx, cy, w, h]))
        bbox = [cx - w//2, cy - h//2, w, h]
        bbox_mask = th.zeros_like(mask)
        bbox_mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = 1

        return img, {"image_id":ann["image_id"], "bbox": bbox_mask, "masks":mask, "boxes":th.tensor([cx-w//2, cy-h//2, cx+w//2, cy+h//2]).unsqueeze(0)}
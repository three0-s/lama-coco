import torch as th 
from torchvision.transforms import Resize, Compose, Lambda
from torchvision.transforms.functional import crop
from torchvision.ops import masks_to_boxes

class CoCoInpaintDataset :
    def __init__(self, coco_dataset, res=256):
        self.coco_dataset = coco_dataset
        _ids = self.coco_dataset.ids
        self.img_idxs = []
        self.ann_idxs = []
        self.res = res
        for i, img_id in enumerate(_ids):
            # ann_num = len(self.coco_dataset.coco.getAnnIds(img_id))
            ann_ids = self.coco_dataset.coco.getAnnIds(img_id)
            for j, ann_id in enumerate(ann_ids):
                # check if ann mask is not too small
                ann = self.coco_dataset.coco.loadAnns(ann_id)[0]
                mask = th.from_numpy(self.coco_dataset.coco.annToMask(ann))
                if mask.sum() < 100:
                    continue
                self.img_idxs.append(i)
                self.ann_idxs.append(j)
        
    
    def __len__(self):
        return len(self.img_idxs)
    
    def __getitem__(self, idx):
        img, ann_tot = self.coco_dataset[self.img_idxs[idx]]
        ann = ann_tot[self.ann_idxs[idx]]
        mask = th.from_numpy(self.coco_dataset.coco.annToMask(ann))
        assert mask.shape == img.shape[1:], f"Mask shape {mask.shape} does not match image shape {img.shape[1:]}"
        bbox = masks_to_boxes(mask.unsqueeze(0)).squeeze(0)
        cx, cy = bbox[::2].sum()//2, bbox[1::2].sum()//2
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        size = int(max(w, h).item() * 2)
        x_r = int(size//2 - w//2)
        x_l = int(w//2 - size//2)
        y_r = int(size//2 - h//2)
        y_l = int(h//2 - size//2)
        dx, dy = th.randint(x_l, x_r, (1,)).item(), th.randint(y_l, y_r, (1,)).item()
        
        transform = Compose([
            Lambda(lambda x: crop(x, int(cy+dy)-size//2, int(cx+dx)-size//2, size, size)),
            Resize((self.res, self.res))

        ])
        scale = self.res/size
        img = transform(img)
        mask = transform(mask.unsqueeze(0)).squeeze(0)
        bbox = masks_to_boxes(mask.unsqueeze(0)).squeeze(0)
        bbox_mask = th.zeros_like(mask)
        bbox_mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 1

        return img, {"image_id":ann["image_id"], "ann_id":ann['id'], "bbox": bbox_mask.unsqueeze(0), "masks":mask.unsqueeze(0), "boxes":bbox, "translate":th.tensor([dx*scale, dy*scale])}
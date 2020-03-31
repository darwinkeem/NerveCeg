import torch
import torch.nn as nn
import torchvision

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from segmentation.dataset import RealTestNerveSegmentationDataset
from segmentation.transform import real_preprocessing
from classification.transform import pred
from segmentation.trainer import to_np


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated(device=device)
    ngpus_per_node = torch.cuda.device_count()
    target_folder = './data/'

    ds_realtest = RealTestNerveSegmentationDataset(root='./data/', transform=real_preprocessing)
    
    cls_model = torch.load('cls/final.pt')
    cls_model.eval()

    seg_model = torch.load('seg/final.pt')
    seg_model.eval()

    for i in range(1, 5509):
        X, y = ds_realtest.__getitem__(i)
        X = X.view(1, 3, 400, 400).cuda(device)
        values, indices = pred(cls_model(X))
        mask_exist = to_np(indices)[0]
        print("mask_exist_y_pred:", str(to_np(indices)[0]))
        # mask_exist = 1

        if mask_exist == 1:
            seg_pred = seg_model(X)
            torchvision.utils.save_image(seg_pred, './data/real_test/mask/' + str(i) + '.png')
        elif mask_exist == 0:
            seg_pred = torch.zeros_like(X)
            torchvision.utils.save_image(seg_pred, './data/real_test/mask/' + str(i) + '.png')

        print(str(i) + " image done.")

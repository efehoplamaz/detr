from generate_spectrogram import get_spectrogram_sampling_rate, display_spectrogram
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as F
import json
from matplotlib import pyplot as plt
from skimage import io, transform
import numpy as np


class BatAnnotationDataSet(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, audio_file, ann_file, transform=None, return_masks = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.bat_anns = json.load(open(ann_file))
        self.root_dir = audio_file
        self.transform = transform
        self.prepare = BatConvert(return_masks)
        #self.prepare = 

    def __len__(self):
        return len(self.bat_anns)

    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()

        wav_name = self.bat_anns[idx]['id']
        anns = self.bat_anns[idx]
        spec, sampling_rate = get_spectrogram_sampling_rate(self.root_dir + wav_name)
        anns_simplified = []
        for ann in anns['annotation']:
            d = {}
            x = ann['start_time']
            y = ann['high_freq']
            width = ann['end_time'] - ann['start_time']
            height = ann['high_freq'] - ann['low_freq']
            category_id = 0
            area = width * height
            d['bbox'] = [x, y, width, height]
            d['area'] = area
            d['category_id'] = category_id
            anns_simplified.append(d)
        target = {'annotations': anns_simplified, 'sampling_rate': sampling_rate}
        
        spec, target = self.prepare(spec, target)
        
        if self.transform:
            spec, target = self.transform(spec, target)

        return spec, target


class BatConvert(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        #w, h = image.size

        #image_id = target["image_id"]
        #image_id = torch.tensor([image_id])

        anno = target["annotations"]

        #anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        #boxes[:, 0::2].clamp_(min=0, max=w)
        #boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        #if self.return_masks:
        #    segmentations = [obj["segmentation"] for obj in anno]
        #    masks = convert_coco_poly_to_mask(segmentations, h, w)

#         keypoints = None
#         if anno and "keypoints" in anno[0]:
#             keypoints = [obj["keypoints"] for obj in anno]
#             keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
#             num_keypoints = keypoints.shape[0]
#             if num_keypoints:
#                 keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
#         if self.return_masks:
#             masks = masks[keep]
#         if keypoints is not None:
#             keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
#         if self.return_masks:
#             target["masks"] = masks
        #target["image_id"] = image_id
#         if keypoints is not None:
#             target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        #iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        #target["iscrowd"] = iscrowd[keep]

        #target["orig_size"] = torch.as_tensor([int(h), int(w)])
        #target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, spec, target):

        c, h, w = spec.shape

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        spec = transform.resize(spec[0].numpy(), (new_h, new_w))

        return torch.from_numpy(spec).unsqueeze(0), target


class ToTensor(object):
    def __call__(self, spec, target):
        return F.to_tensor(spec), target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, spec, target):
        for t in self.transforms:
            spec, target = t(spec, target)
        #print(spec.type(torch.DoubleTensor).dtype)
        return spec, target

def make_bat_transforms(image_set):
    # TODO: write transformations according to the image_set
    return Compose([ToTensor(),Rescale((256, 1718))])


def build(image_set, args):

    PATHS = {
        "train_val": ('C:\\Users\\ehopl\\Desktop\\Fourth Year\\Disseration\\bat_data_oct_2020_ug4\\annotations\\BritishBatCalls_MartynCooke_2018_1_sec_train_expert.json', 'C:\\Users\\ehopl\\Desktop\\Fourth Year\\Disseration\\bat_data_oct_2020_ug4\\audio\\mc_2018\\audio\\'),
        "test": ('C:\\Users\\ehopl\\Desktop\\Fourth Year\\Disseration\\bat_data_oct_2020_ug4\\annotations\\BritishBatCalls_MartynCooke_2019_1_sec_train_expert.json', 'C:\\Users\\ehopl\\Desktop\\Fourth Year\\Disseration\\bat_data_oct_2020_ug4\\audio\\mc_2019\\audio\\'),
    }

    if image_set == 'train_val':
        ann_file, audio_file = PATHS['train_val']
        dataset = BatAnnotationDataSet(ann_file = ann_file, audio_file= audio_file, transform=make_bat_transforms(image_set))
        train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
        return train_set, val_set

    elif image_set == 'test':
        ann_file, audio_file = PATHS['test']
        dataset = BatAnnotationDataSet(ann_file = ann_file, audio_file= audio_file, transform=make_bat_transforms(image_set))
        return dataset
    else:
        return None
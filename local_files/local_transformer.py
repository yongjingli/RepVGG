import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image, ImageOps

import cv2
import numpy as np
import os


class ResizeCenterCropPaddingShort(object):
    """Center crop the image.
    size = [224, -1]
    long size will resize to size[0]
    short size will padding to size[0] too
    """

    def __init__(self, size, interpolation=Image.BILINEAR, backend='torch'):
        assert isinstance(size, int) or (isinstance(size, tuple) and len(size) == 2)
        self.resize_w_long_side = False
        if isinstance(size, int):
            assert size > 0
            size = (size, size)
        else:
            assert size[0] > 0 and (size[1] > 0 or size[1] == -1)
            if size[1] == -1:
                self.resize_w_long_side = True
        # assert interpolation in (Image.NEAREST, Image.BILINEAR, Image.BICUBIC)
        assert backend in ['torch', 'cv2']
        self.size = size
        self.interpolation = interpolation
        self.backend = backend

    def _resize_img(self, img):
        ignore_resize = False
        if self.resize_w_long_side:
            w, h = img.size
            long_side = self.size[0]
            if (w >= h and w == long_side) or (h >= w and h == long_side):
                ignore_resize = True
            else:
                if w > h:
                    width = long_side
                    height = int(long_side * h / w)
                else:
                    height = long_side
                    width = int(long_side * w / h)
        else:
            height, width = self.size

        if not ignore_resize:
            if self.backend == 'torch':
                img_r = F.resize(img, (height, width), self.interpolation)
            elif self.backend == 'cv2':
                cv_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                img_r = cv2.resize(cv_img, (width, height), interpolation=self.interpolation)
                img_r = Image.fromarray(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB))
            return img_r
        else:
            return img

    def _center_crop_padding(self, img):
        pad_h = (self.size[0] - img.size[1]) // 2
        pad_w = (self.size[0] - img.size[0]) // 2

        pad_t = pad_h
        pad_d = pad_h

        pad_l = pad_w
        pad_r = pad_w

        if (self.size[0] - img.size[0]) % 2 != 0:
            pad_l = pad_l + 1
        if  (self.size[0] - img.size[1]) % 2 !=0:
            pad_t = pad_t + 1

        img_crop_padding = ImageOps.expand(img, border=(pad_l, pad_t, pad_r, pad_d), fill=0)
        return img_crop_padding

    def __call__(self, img):
        img_resize = self._resize_img(img)
        long_size = max(img_resize.size)
        assert long_size == self.size[0]
        img_crop_padding = self._center_crop_padding(img_resize)
        return img_crop_padding

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.size})'





class ResizeCenterCropPaddingShort2(object):
    """Center crop the image.
    size = [224, -1]
    long size will resize to size[0]
    short size will padding to size[0] too
    """

    def __init__(self, size, interpolation=Image.BILINEAR, backend='torch'):
        assert isinstance(size, int) or (isinstance(size, tuple) and len(size) == 2)
        self.resize_w_long_side = False
        if isinstance(size, int):
            assert size > 0
            size = (size, size)
        else:
            assert size[0] > 0 and (size[1] > 0 or size[1] == -1)
            if size[1] == -1:
                self.resize_w_long_side = True
        # assert interpolation in (Image.NEAREST, Image.BILINEAR, Image.BICUBIC)
        assert backend in ['torch', 'cv2']
        self.size = size
        self.interpolation = interpolation
        self.backend = backend


    def _resize_img(self, img):
        ignore_resize = False
        if self.resize_w_long_side:
            w, h = img.size
            long_side = self.size[0]
            if (w >= h and w == long_side) or (h >= w and h == long_side):
                ignore_resize = True
            else:
                if w > h:
                    width = long_side
                    height = int(long_side * h / w)
                else:
                    height = long_side
                    width = int(long_side * w / h)
        else:
            height, width = self.size

        if not ignore_resize:
            if self.backend == 'torch':
                img_r = F.resize(img, (height, width), self.interpolation)
            elif self.backend == 'cv2':
                cv_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                img_r = cv2.resize(cv_img, (width, height), interpolation=self.interpolation)
                img_r = Image.fromarray(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB))
            return img_r
        else:
            return img

    def _padding_short_size_times_32(self, img_resize):
        w, h = img_resize.size
        assert w > 0 and h > 0
        pad_w = 0
        pad_h = 0

        if w%32 != 0:
            pad_w = 32 - w % 32

        if h % 32 != 0:
            pad_h = 32 - h % 32

        pad_t = pad_h//2
        pad_d = pad_h//2

        pad_l = pad_w//2
        pad_r = pad_w//2

        if pad_w % 2 != 0:
            pad_l = pad_l + 1
        if pad_h % 2 != 0:
            pad_t = pad_t + 1

        img_crop_padding = ImageOps.expand(img_resize, border=(pad_l, pad_t, pad_r, pad_d), fill=0)
        return img_crop_padding

    def __call__(self, img):
        img_resize = self._resize_img(img)
        long_size = max(img_resize.size)
        assert long_size == self.size[0]
        img_crop_padding = self._padding_short_size_times_32(img_resize)
        return img_crop_padding

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.size})'





if __name__ == '__main__':
    img_path = '/home/liyongjing/Egolee_2021/programs/RepVGG-main/table.PNG'

    # random_resize = transforms.RandomResizedCrop(224)
    # transforms.ColorJitter
    #color_aug = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    # transforms.RandomErasing
    # transforms.RandomErasing(p=0.5,scale(0.05,0.05),ratio=(0.3,0.3),value=0,inplace=False)

    # RandomPerspective(p=0.5)

    random_erase = transforms.RandomErasing(p=0.5, scale=(0.001, 0.02), ratio=(0.2, 5), value=0, inplace=False)
    random_crop = transforms.RandomCrop((224, 224))
    random_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1, hue=0.1)
    center_crop_pad_short_size = ResizeCenterCropPaddingShort((268, -1))
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    data_transforms = transforms.Compose([random_jitter,to_tensor,random_erase,to_pil,
                                          center_crop_pad_short_size, random_crop, transforms.RandomHorizontalFlip()])

    center_crop_pad_short_size2 = ResizeCenterCropPaddingShort2((224, -1))

    input_dir = '/home/liyongjing/Egolee_2021/data/TrainData/train_fall_down/rep_vgg_format/train/fall_down'
    img_names = [x  for x in os.listdir(input_dir) if os.path.splitext(x)[-1] in ['.jpg', 'png']]
    for img_name in img_names:
        img_path = os.path.join(input_dir, img_name)
        # img_path = '/home/liyongjing/Egolee_2021/data/TrainData/train_fall_down/fall_down/open_image_e264330f827dddf5_fall_down_0.jpg'
        image = Image.open(img_path)
        # image_aug = data_transforms(image)

        image_aug = center_crop_pad_short_size2(image)
        print(image_aug.size)


        cv_image_aug = cv2.cvtColor(np.asarray(image_aug), cv2.COLOR_RGB2BGR)
        cv2.namedWindow('cv_image_aug', 0)
        cv2.imshow('cv_image_aug', cv_image_aug)

        cv_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        cv2.namedWindow('cv_image', 0)
        cv2.imshow('cv_image', cv_image)
        wait_key = cv2.waitKey(0)
        if wait_key == 27:
            break

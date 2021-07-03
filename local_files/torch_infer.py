import sys
sys.path.insert(0, '/home/liyongjing/Egolee_2021/programs/RepVGG-main')

import torch
import cv2
import os
import numpy as np
from repvgg import get_RepVGG_func_by_name
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import random


class RepVGGTorchInfer(object):
    def __init__(self, arch, weights):
        repvgg_build_func = get_RepVGG_func_by_name(arch)
        model = repvgg_build_func(deploy=True)

        if torch.cuda.is_available():
            model = model.cuda()

        self.model_w = weights
        checkpoint = torch.load(weights)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        ckpt = {k.replace('module.', ''):v for k,v in checkpoint.items()}   # strip the names
        model.load_state_dict(ckpt)

        self.model = model.eval()

        self.short_size = 256
        self.dst_w = 224
        self.dst_h = 224
        self.input_size = [self.dst_h, self.dst_w]

        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])
        self.std_inv = 1 / self.std
        self.img_t = None

        self.result = dict()

    def crop_img_short_size(self, cv_img):
        # resize the short size
        h, w, _ = cv_img.shape
        if h >= w:
            h = int(h * self.short_size / w)
            w = int(self.short_size)
        else:
            w = int(w * self.short_size / h)
            h = int(self.short_size)

        cv_img = cv2.resize(cv_img, (w, h), cv2.INTER_LINEAR)

        # center crop
        y1 = max(0, int(round((h - self.input_size[1]) / 2.)))
        x1 = max(0, int(round((w - self.input_size[0]) / 2.)))
        y2 = min(h-1, y1 + self.input_size[1])
        x2 = min(w-1, x1 + self.input_size[0])

        cv_img = cv_img[y1:y2, x1:x2, :]
        return cv_img

    def crop_img_long_size(self, cv_img):
        long_size = max(cv_img.shape[:2])

        pad_h = (long_size - cv_img.shape[0]) // 2
        pad_w = (long_size - cv_img.shape[1]) // 2
        img_input = np.ones((long_size, long_size, 3), dtype=np.uint8) * 0
        img_input[pad_h:cv_img.shape[0] + pad_h, pad_w:cv_img.shape[1] + pad_w, :] = cv_img
        img_input = cv2.resize(img_input, (self.input_size[1], self.input_size[0]), cv2.INTER_LINEAR)
        return img_input

    def crop_img_long_size2(self, cv_img):
        ignore_resize = False
        long_side = max(self.input_size)

        w, h, _ = cv_img.shape
        if (w >= h and w == long_side) or (h >= w and h == long_side):
            ignore_resize = True
        else:
            if w > h:
                width = long_side
                height = int(long_side * h / w)
            else:
                height = long_side
                width = int(long_side * w / h)

        if not ignore_resize:
            cv_img = cv2.resize(cv_img, (height, width), cv2.INTER_LINEAR)

        long_size = max(cv_img.shape[:2])
        pad_h = (long_size - cv_img.shape[1]) // 2
        pad_w = (long_size - cv_img.shape[0]) // 2

        pad_t = pad_h
        pad_d = pad_h

        pad_l = pad_w
        pad_r = pad_w

        if (long_size - cv_img.shape[0]) % 2 != 0:
            pad_l = pad_l + 1
        if (long_size - cv_img.shape[1]) % 2 != 0:
            pad_t = pad_t + 1

        img_crop_padding = cv2.copyMakeBorder(cv_img,  pad_l, pad_r, pad_t, pad_d, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return img_crop_padding

    def infer_cv_img(self, cv_img):
        # cv_img = self.crop_img_short_size(cv_img)
        cv_img = self.crop_img_long_size2(cv_img)
        assert list(cv_img.shape[:2]) == self.input_size

        # cv2.namedWindow("cv_img", 0)
        # cv2.imshow("cv_img", cv_img)

        # normalize
        cv_img = cv_img.copy().astype(np.float32)
        self.mean = np.float64(self.mean.reshape(1, -1))
        self.std_inv = 1 / np.float64(self.std.reshape(1, -1))
        if True:
            cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB, cv_img)  # inplace
        cv2.subtract(cv_img, self.mean, cv_img)  # inplace
        cv2.multiply(cv_img, self.std_inv, cv_img)  # inplace

        self.img_t = cv_img.transpose(2, 0, 1)  # to C, H, W
        self.img_t = np.ascontiguousarray(self.img_t)

        self.img_t = np.expand_dims(self.img_t, axis=0)
        self.img_t = torch.from_numpy(self.img_t)
        if torch.cuda.is_available():
            self.img_t = self.img_t.cuda()

        output = self.model(self.img_t)
        output = output.cpu().detach().numpy()

        # softmax
        tmp = np.max(output, axis=1)
        output -= tmp.reshape((output.shape[0],1))
        output = np.exp(output)
        tmp = np.sum(output, axis=1)
        output /= tmp.reshape((output.shape[0], 1))

        pred_score = np.max(output, axis=1)[0]
        pred_label = np.argmax(output, axis=1)[0]
        self.result.update({'pred_label': pred_label, 'pred_score': float(pred_score)})
        return self.result

    def infer_pil_img(self, pil_img):
        from local_files.local_transformer import ResizeCenterCropPaddingShort
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        center_crop_pad_short_size = ResizeCenterCropPaddingShort((224, -1), interpolation=cv2.INTER_LINEAR,
                                                                backend='cv2')
        # center_crop_pad_short_size = ResizeCenterCropPaddingShort((224, -1), interpolation=Image.BILINEAR,
        #                                                         backend='torch')
        transforms_compose = transforms.Compose([center_crop_pad_short_size, transforms.ToTensor(), normalize,])

        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        center_crop_pad_short_size = ResizeCenterCropPaddingShort((224, -1), interpolation=Image.BILINEAR,
                                                                  backend='torch')



        pil_img = transforms_compose(pil_img)

        pil_img = pil_img.unsqueeze(dim=0)

        if torch.cuda.is_available():
            pil_img = pil_img.cuda()

        output = self.model(pil_img)
        output = torch.nn.functional.softmax(output, dim=1)
        output = output.cpu().detach().numpy()

        pred_score = np.max(output, axis=1)[0]
        pred_label = np.argmax(output, axis=1)[0]
        self.result.update({'pred_label': pred_label, 'pred_score': float(pred_score)})
        return self.result

    def onnx_exprot(self):
        self.model = self.model.cpu()
        img_dry = torch.zeros((1, 3, self.dst_h, self.dst_w))
        with torch.no_grad():
            y = self.model(img_dry)  # forward

        try:
            import onnx
            print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
            f = self.model_w.replace('.pth', '.onnx')  # filename
            # print(model.t)
            torch.onnx.export(self.model, img_dry, f, verbose=False, opset_version=11, \
                              input_names=['images'], output_names=['output'])
            # Checks
            onnx_model = onnx.load(f)  # load onnx model
            onnx.checker.check_model(onnx_model)  # check onnx model
            print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model

            # simpily onnx
            from onnxsim import simplify
            model_simp, check = simplify(onnx_model)
            assert check, "Simplified ONNX model could not be validated"

            f2 = f.replace('.onnx', '_sim.onnx')  # filename
            onnx.save(model_simp, f2)
            print('====ONNX SIM export success, saved as %s' % f2)

            from onnx import shape_inference
            f3 = f2.replace('.onnx', '_shape.onnx')  # filename
            onnx.save(onnx.shape_inference.infer_shapes(onnx.load(f2)), f3)
            print('====ONNX shape inference export success, saved as %s' % f3)

            print('ONNX export success, saved as %s' % f)
        except Exception as e:
            print('ONNX export failure: %s' % e)

    def print_model_names(self):
        for name, paras, in self.model.named_modules():
            print(name)

    def diff_size_input_test(self, input_img):
        output = self.model(input_img)
        return output


def infer_images(images_dir, arch, weights):
    infer_model = RepVGGTorchInfer(arch, weights)
    img_names = [f for f in os.listdir(images_dir) if os.path.splitext(f)[-1] in ['.jpg']]
    for img_name in img_names:
        img_path = os.path.join(images_dir, img_name)
        # img = cv2.imread(img_path)
        # pred_result = infer_model.infer_cv_img(img)

        pil_image = Image.open(img_path)
        pred_result = infer_model.infer_pil_img(pil_image)
        img = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

        # print(pred_result)
        cv2.namedWindow('img')
        cv2.imshow('img', img)
        print(pred_result)
        if pred_result['pred_label'] != 0:
            cv2.waitKey(0)
        # exit(1)


def export_onnx(arch, weights):
    infer_model = RepVGGTorchInfer(arch, weights)
    infer_model.onnx_exprot()


def print_model_names(arch, weights):
    infer_model = RepVGGTorchInfer(arch, weights)
    infer_model.print_model_names()


def test_diff_size_input(arch, weights):
    infer_model = RepVGGTorchInfer(arch, weights)
    for i in range(20):
        w_random = random.randint(1, 20) * 32
        h_random = random.randint(1, 20) * 32
        input_t = torch.ones((1, 3, h_random, w_random))
        input_t = input_t.cuda()
        output = infer_model.diff_size_input_test(input_t)
        print('*'*20)
        print('input_t shape:{}'.format(input_t.shape))
        print('output shape:{}'.format(output.shape))


if __name__ == '__main__':
    input_dir = '/home/liyongjing/Egolee_2021/data/TrainData/train_fall_down/rep_vgg_format/val/fall_down'
    arch = 'RepVGG-B2'
    weights = '/home/liyongjing/Egolee_2021/programs/RepVGG-main/trained_model/RepVggB2-padding-short-size/RepVGG-B2-deploy.pth'

    infer_images(input_dir, arch, weights)
    # export_onnx(arch, weights)
    # print_model_names(arch, weights)
    # test_diff_size_input(arch, weights)

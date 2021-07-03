import cv2
import onnxruntime
import numpy as np
import os
import shutil


class RepVGGOnnx():
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        self.ort_sess = onnxruntime.InferenceSession(self.onnx_path)
        self.input_name = self.ort_sess.get_inputs()[0].name

        self.short_size = 256
        self.dst_w = 224
        self.dst_h = 224
        self.input_size = [self.dst_h, self.dst_w]

        self.long_size = 224

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

    def padding_short_size(self, cv_img):
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
        # cv_img = self.crop_img_long_size(cv_img)
        cv_img = self.padding_short_size(cv_img)
        assert list(cv_img.shape[:2]) == self.input_size

        cv2.namedWindow("cv_img", 0)
        cv2.imshow("cv_img", cv_img)

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

        output = self.ort_sess.run(None, {self.input_name: self.img_t})[0]

        pred_score = np.max(output, axis=1)[0]
        pred_label = np.argmax(output, axis=1)[0]
        self.result.update({'pred_label': pred_label, 'pred_score': float(pred_score)})
        return self.result


def repvgg_cls_images_infer(images_dir, onnx_path):
    check_dir = '/home/liyongjing/Egolee_2021/programs/RepVGG-main/check_data'

    mm_classifier_onnx = RepVGGOnnx(onnx_path)
    image_names = [f for f in os.listdir(images_dir) if os.path.splitext(f)[-1] in ['.jpg', 'png']]
    for image_name in image_names:
        img_path = os.path.join(images_dir, image_name)
        img = cv2.imread(img_path)
        cls_result = mm_classifier_onnx.infer_cv_img(img)
        print(cls_result)
        # print(image_net.CLASSES[cls_result['pred_label']])
        cv2.namedWindow("img", 0)
        cv2.imshow("img", img)
        if cls_result['pred_label'] != 0:
            # dst_img_path = img_path.replace(images_dir, check_dir)
            # shutil.copy(img_path, dst_img_path)
            wait_key = cv2.waitKey(0)
            if wait_key == 27:
                break


if __name__ == "__main__":
    print("Start Proc...")
    onnx_path = "/home/liyongjing/Egolee_2021/programs/RepVGG-main/trained_model/RepVggB1-padding-short-size/RepVGG-B1-deploy.onnx"
    images_dir = '/home/liyongjing/Egolee_2021/data/TrainData/train_fall_down/rep_vgg_format/val/fall_down'
    repvgg_cls_images_infer(images_dir, onnx_path)

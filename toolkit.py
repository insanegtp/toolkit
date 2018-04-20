import os, sys, cv2,re
import numpy as np
import random
import mxnet as mx
from mxnet.gluon.data import dataset
from mxnet.image import image
import xml.etree.ElementTree as ET
import pylab

def parse_class_names(num_class,class_names):
    """ parse # classes and class_names if applicable """
    num_class = num_class
    if len(class_names) > 0:
        if os.path.isfile(class_names):
                # try to open it to read class names
                with open(class_names, 'r') as f:
                    class_names = [l.strip().split('-')[1] for l in f.readlines()]
                    class_names+=['dumy']
        else:
            class_names = [c.strip() for c in class_names.split(',')]
            class_names += ['dumy']
        assert len(class_names) == num_class+1, str(len(class_names))
        for name in class_names:
            assert len(name) > 0
    else:
        class_names = None
    return class_names

def split_train_val(img_dirs,ratio=0.9):
    img_file_names=[]
    idx_train=[]
    idx_val=[]
    count=0
    last_count=0
    for root,_,files in os.walk(img_dirs):
        count=0
        for i,file in enumerate(files):
            if file[-3:]=='JPG' or file[-3:]=='jpg':
                img_file_names.append(os.path.join(root, file))
                count+=1
        if count!=0:
            if ratio < 1 and ratio > 0:
                idx_random = list(range(last_count,last_count+count))
                random.shuffle(idx_random)
                idx_train = idx_train+idx_random[:int(count * ratio) + 1]
                idx_val = idx_val+idx_random[int(count * ratio) + 1:]
            else:
                print('wrong ratio value!')
        last_count = last_count+count

    return img_file_names,idx_train,idx_val

class DetIter4sefl_defined_format(mx.io.DataIter):
    """
    Detection Iterator, which will feed data and label to network
    Optional data augmentation is performed when providing batch

    Parameters:
    ----------
    imdb : Imdb
        image database
    batch_size : int
        batch size
    data_shape : int or (int, int)
        image shape to be resized
    mean_pixels : float or float list
        [R, G, B], mean pixel values
    rand_samplers : list
        random cropping sampler list, if not specified, will
        use original image only
    rand_mirror : bool
        whether to randomly mirror input images, default False
    shuffle : bool
        whether to shuffle initial image list, default False
    rand_seed : int or None
        whether to use fixed random seed, default None
    max_crop_trial : bool
        if random crop is enabled, defines the maximum trial time
        if trial exceed this number, will give up cropping
    is_train : bool
        whether in training phase, default True, if False, labels might
        be ignored
    """
    def __init__(self, imdb, batch_size, data_shape, \
                 mean_pixels=[128, 128, 128], rand_samplers=[], \
                 rand_mirror=False, shuffle=False, rand_seed=None, \
                 is_train=True, max_crop_trial=50):
        super(DetIter4sefl_defined_format, self).__init__()

        self._imdb = imdb
        self.batch_size = batch_size
        if isinstance(data_shape, int):
            data_shape = (data_shape, data_shape)
        self._data_shape = data_shape
        self._mean_pixels = mx.nd.array(mean_pixels).reshape((3,1,1))
        if not rand_samplers:
            self._rand_samplers = []
        else:
            if not isinstance(rand_samplers, list):
                rand_samplers = [rand_samplers]
            assert isinstance(rand_samplers[0], RandSampler), "Invalid rand sampler"
            self._rand_samplers = rand_samplers
        self.is_train = is_train
        self._rand_mirror = rand_mirror
        self._shuffle = shuffle
        if rand_seed:
            np.random.seed(rand_seed) # fix random seed
        self._max_crop_trial = max_crop_trial

        self._current = 0
        self._size = imdb.num_images
        self._index = np.arange(self._size)

        self._data = None
        self._label = None
        self._get_batch()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in self._data.items()]

    @property
    def provide_label(self):
        if self.is_train:
            return [(k, v.shape) for k, v in self._label.items()]
        else:
            return [(k, v.shape) for k, v in self._label.items()]

    def reset(self):
        self._current = 0
        if self._shuffle:
            np.random.shuffle(self._index)

    def iter_next(self):
        return self._current < self._size

    def next(self):
        if self.iter_next():
            self._get_batch()
            data_batch = mx.io.DataBatch(data=list(self._data.values()),
                                   label=list(self._label.values()),
                                   pad=self.getpad(), index=self.getindex())
            self._current += self.batch_size
            return data_batch
        else:
            raise StopIteration

    def getindex(self):
        return self._current // self.batch_size

    def getpad(self):
        pad = self._current + self.batch_size - self._size
        return 0 if pad < 0 else pad

    def _get_batch(self):
        """
        Load data/label from dataset
        """
        batch_data = mx.nd.zeros((self.batch_size, 3, self._data_shape[0], self._data_shape[1]))
        batch_label = []
        for i in range(self.batch_size):
            if (self._current + i) >= self._size:
                if not self.is_train:
                    continue
                # use padding from middle in each epoch
                idx = (self._current + i + self._size // 2) % self._size
                index = self._index[idx]
            else:
                index = self._index[self._current + i]
            # index = self.debug_index
            im_path = self._imdb.image_path_from_index(index)
            with open(im_path, 'rb') as fp:
                img_content = fp.read()
            img = mx.img.imdecode(img_content)
            gt = self._imdb.label_from_index(index).copy() if self.is_train else None
            data, label = self._data_augmentation(img, gt)
            batch_data[i] = data
            if self.is_train:
                batch_label.append(label)
        self._data = {'data': batch_data}
        if self.is_train:
            self._label = {'yolo_output_label': mx.nd.array(np.array(batch_label))}
        else:
            self._label = {'yolo_output_label': mx.nd.zeros((1, 2, 5))}  # fake label

    def _data_augmentation(self, data, label):
        """
        perform data augmentations: crop, mirror, resize, sub mean, swap channels...
        """
        if self.is_train and self._rand_samplers:
            rand_crops = []
            for rs in self._rand_samplers:
                rand_crops += rs.sample(label)
            num_rand_crops = len(rand_crops)
            # randomly pick up one as input data
            if num_rand_crops > 0:
                index = int(np.random.uniform(0, 1) * num_rand_crops)
                width = data.shape[1]
                height = data.shape[0]
                crop = rand_crops[index][0]
                xmin = int(crop[0] * width)
                ymin = int(crop[1] * height)
                xmax = int(crop[2] * width)
                ymax = int(crop[3] * height)
                if xmin >= 0 and ymin >= 0 and xmax <= width and ymax <= height:
                    data = mx.img.fixed_crop(data, xmin, ymin, xmax-xmin, ymax-ymin)
                else:
                    # padding mode
                    new_width = xmax - xmin
                    new_height = ymax - ymin
                    offset_x = 0 - xmin
                    offset_y = 0 - ymin
                    data_bak = data
                    data = mx.nd.full((new_height, new_width, 3), 128, dtype='uint8')
                    data[offset_y:offset_y+height, offset_x:offset_x + width, :] = data_bak
                label = rand_crops[index][1]
        if self.is_train:
            interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, \
                              cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        else:
            interp_methods = [cv2.INTER_LINEAR]
        interp_method = interp_methods[int(np.random.uniform(0, 1) * len(interp_methods))]
        data = mx.img.imresize(data, self._data_shape[1], self._data_shape[0], interp_method)
        if self.is_train and self._rand_mirror:
            if np.random.uniform(0, 1) > 0.5:
                data = mx.nd.flip(data, axis=1)
                valid_mask = np.where(label[:, 0] > -1)[0]
                tmp = 1.0 - label[valid_mask, 1]
                label[valid_mask, 1] = 1.0 - label[valid_mask, 3]
                label[valid_mask, 3] = tmp
        data = mx.nd.transpose(data, (2,0,1))
        data = data.astype('float32')
        data = data - self._mean_pixels
        return data, label

class MultiBoxMetric(mx.metric.EvalMetric):
    """Calculate metrics for Multibox training """
    def __init__(self, thresh=0.5, eps=1e-8):
        super(MultiBoxMetric, self).__init__('MultiBox')
        self.eps = eps
        self.thresh = thresh
        self.num = 4
        self.name = ['Acc', 'IOU', 'BG-score', 'Obj-score']
        self.reset()

    def reset(self):
        """Override reset behavior"""
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num

    def update(self, labels, preds):
        """
        Implementation of updating metrics
        """
        # temp = preds[0].asnumpy()
        # print np.sum(temp[0, :, 0] > -1);
        # print temp[0, :, :]
        # raise RuntimeError
        # print np.reshape(temp, (-1, 32))[:3, :]
        # num_batch = preds[1].shape[0]
        # metric = mx.nd.slice_axis(preds[1].reshape((-1, num_batch)),
        #     axis=0, begin=0, end=3);
        # s = mx.nd.sum(metric, axis=1)
        # self.sum_metric[0] += s[0].asscalar()
        # self.num_inst[0] += 1
        # self.sum_metric[1] += s[1].asscalar()
        # self.num_inst[1] += 1
        # self.sum_metric[2] += s[2].asscalar()
        # self.num_inst[2] += 1
        # # assert(0 == 1)
        # # print "-----------------------"
        # # print preds[0].asnumpy()[0, :10, :]
        # # print preds[1].asnumpy()[0, :10, :]
        # return

        def calc_ious(b1, b2):
            assert b1.shape[1] == 4
            assert b2.shape[1] == 4
            num1 = b1.shape[0]
            num2 = b2.shape[0]
            b1 = np.repeat(b1.reshape(num1, 1, 4), num2, axis=1)
            b2 = np.repeat(b2.reshape(1, num2, 4), num1, axis=0)
            dw = np.maximum(0, np.minimum(b1[:, :, 2], b2[:, :, 2]) - \
                np.maximum(b1[:, :, 0], b2[:, :, 0]))
            dh = np.maximum(0, np.minimum(b1[:, :, 3], b2[:, :, 3]) - \
                np.maximum(b1[:, :, 1], b2[:, :, 1]))
            inter_area = dw * dh
            area1 = np.maximum(0, b1[:, :, 2] - b1[:, :, 0]) * \
                np.maximum(0, b1[:, :, 3] - b1[:, :, 1])
            area2 = np.maximum(0, b2[:, :, 2] - b2[:, :, 0]) * \
                np.maximum(0, b2[:, :, 3] - b2[:, :, 1])
            union_area = area1 + area2 - inter_area
            ious = inter_area / (union_area + self.eps)
            return ious

        def draw(boxes):
            import cv2
            w = 800
            h = 800
            canvas = np.ones((h, w, 3)) * 255
            for i in range(boxes.shape[0]):
                box = boxes[i, :]
                pt1 = (int(box[1] * h), int(box[0] * w))
                pt2 = (int(box[3] * h), int(box[2] * w))
                colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 255], [255, 255, 0]])
                color = colors[i % 5, :].astype(int)
                print(color)
                cv2.rectangle(canvas, pt1, pt2, color, 2)
            cv2.imshow('image', canvas)
            cv2.waitKey(0)

        out = preds.asnumpy() # list(preds)[0].asnumpy()
        out = out.reshape((out.shape[0], -1, 6))
        # draw(out[0, :, 2:])
        label = np.stack(labels)
        for i in range(out.shape[0]):
            valid_out_mask = np.where(out[i, :, 0] >= 0)[0]
            valid_out = out[i, valid_out_mask, :]
            valid_mask = np.where(label[i, :, 4] >= 0)[0]
            valid_label = label[i, valid_mask, 5:]
            ious = calc_ious(valid_out[:, 2:6], valid_label)
            # max_iou = np.amax(ious, axis=0)
            # self.sum_metric[0] += np.sum(max_iou > self.thresh)
            # self.num_inst[0] += max_iou.size
            # self.sum_metric[1] += np.sum(max_iou)
            # self.num_inst[1] += max_iou.size
            for j in range(valid_mask.size):
                gt_id = label[i, valid_mask[j], 4]
                # correct = np.intersect1d(np.where(ious[:, j] > self.thresh)[0],
                #     np.where(out[i, :, 0] == gt_id)[0])
                best_idx = np.argmax(ious[:, j])
                correct = valid_out[best_idx, 0] == gt_id
                if correct:
                    self.sum_metric[0] += 1.
                max_iou = np.amax(ious[:, j])
                self.sum_metric[1] += max_iou
                self.num_inst[1] += 1
                self.num_inst[0] += 1
            bg_mask = np.where(np.amax(ious, axis=1) < self.eps)[0]
            self.sum_metric[2] += np.sum(valid_out[bg_mask, 1])
            self.num_inst[2] += bg_mask.size
            obj_mask = np.where(np.amax(ious, axis=1) > self.thresh)[0]
            self.sum_metric[3] += np.sum(valid_out[obj_mask, 1])
            self.num_inst[3] += obj_mask.size

    def get(self):
        """Get the current evaluation result.
        Override the default behavior

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s'%(self.name[i]) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan') \
                for x, y in zip(self.sum_metric, self.num_inst)]
            return (names, values)


def cal_static_img(img_dir):
    imgs_std=[]
    imgs_mean=[]
    file_names = os.listdir(img_dir)
    file_num = int(len(file_names))
    for idx in range(file_num):
        file_name=file_names[idx]
        if not (file_name.endswith('JPG') or file_name.endswith('jpg')):
            continue
        img = cv2.imread(img_dir + file_name)
        # imgs.append(img)
        imgs_mean.append([img[...,i].ravel().mean() for i in range(img.shape[2])])
        imgs_std.append([img[..., i].ravel().std() for i in range(img.shape[2])])

    imgs_mean_seprate=np.stack(imgs_mean).mean(axis=0)
    imgs_std_seprate = np.stack(imgs_std).mean(axis=0)

    return imgs_mean_seprate, imgs_std_seprate

class ImageDetFolderDataset(dataset.Dataset):

    def __init__(self, dir_img, dir_txt,flag=1, transform=None):
        self._dir_img = os.path.expanduser(dir_img)
        self._dir_txt = os.path.expanduser(dir_txt)
        self._flag= flag
        self._transform = transform
        self._exts = ['.jpg', '.jpeg', '.png','JPG']
        self._list_images(self._dir_img,self._dir_txt)

    def _list_images(self, dir_img, dir_txt):
        self.synsets = []
        self.items = []
        miss_label=0
        miss_clsname=0
        repaired_clsname=0
        for img_root, dirs, files in os.walk(dir_img):
            for i, img_name in enumerate(files):
                if img_name[-3:] == 'JPG' or img_name[-3:] == 'jpg':
                    imgfilename=img_root+'/'+img_name
                    label_file_path=imgfilename[:-4]
                    if os.path.isfile(label_file_path):
                        label_file = label_file_path
                    elif os.path.isfile(label_file_path + '.xml'):
                        label_file = label_file_path + '.xml'
                    else:
                        miss_label += 1
                        # print('No label file exists!!!Corresponding to img: %s' % imgfilename)
                        # print('Number of missed label: %d' % miss_label)
                        continue
                    tree = ET.parse(label_file)
                    root = tree.getroot()
                    size = root.find('size')
                    width = float(size.find('width').text)
                    height = float(size.find('height').text)
                    label = []
                    cls_names = []
                    if not root.find('object'):
                        print('There is no object in the file!!!filename: %s' % label_file)
                        continue
                    for obj in root.iter('object'):
                        cls_name = obj.find('name').text
                        if not cls_name:
                            print('There is no class name in the file!!!')
                            continue
                        if cls_name not in class_names:
                            miss_clsname += 1
                            folder_info = root.find('folder').text
                            print('The class name is wrong!!!label_file is:%s, Folder info is : %s' % (label_file, folder_info))
                            # iffix=input('y or n ?')
                            # if iffix=='': iffix='y'
                            iffix = 'y'
                            if iffix == 'y':
                                true_clsname = label_file.split('/')[-2].split('-')[1]
                                obj.find('name').text = true_clsname
                                tree.write(label_file)
                                print('Repaired class name from %s to %s' % (cls_name, true_clsname))
                            repaired_clsname += 1
                            print(
                                'A label information of class name is repaired!!!Totally %d are repaired!!!' % repaired_clsname)
                            print('Class name is not in the list!!!(%s)' % cls_name)
                            print('Wrong class name file: %s, Totally %d are missed' % (label_file, miss_clsname))
                            continue
                        cls_id = class_names.index(cls_name)
                        xml_box = obj.find('bndbox')
                        xmin1 = float(xml_box.find('xmin').text) / width
                        ymin1 = float(xml_box.find('ymin').text) / height
                        xmax1 = float(xml_box.find('xmax').text) / width
                        ymax1 = float(xml_box.find('ymax').text) / height
                        label.append([4, 5, width, height, cls_id, xmin1, ymin1, xmax1, ymax1]) # only support one target this version!!!
                        cls_names.append(cls_name)  # only support one target this version!!!
                    if len(label)>1:
                        print('some label is more than one!!!filename:%s' % label_file)
                        pylab.figure()
                        img_tmp=cv2.imread(imgfilename)
                        pylab.imshow(img_tmp)
                        for ii,itm in enumerate(label):
                            color=['red','blue','yellow','green']
                            rect=pylab.Rectangle(
                                (itm[5]*itm[2], itm[6]*itm[3]), itm[7]*itm[2] - itm[5]*itm[2], itm[8]*itm[3] - itm[6]*itm[3],
                                fill=False, edgecolor=color[ii], linewidth=1)
                            pylab.gca().add_patch(rect)
                            # pylab.add(rect)
                        pylab.show()


                        continue

                    self.synsets.append(cls_names)
                    self.items.append((imgfilename, label))


    def __getitem__(self, idx):
        try:
            img = image.imread(self.items[idx][0], self._flag) # image.imread(self.items[idx][0], self._flag)
        except:
            img=cv2.imread(self.items[idx][0])
            cv2.imwrite(self.items[idx][0],img)
            img = image.imread(self.items[idx][0], self._flag)
            print('repaired one img format!!!')
        label = self.items[idx][1]#+[img.shape]+[self.items[idx][0]]
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def __len__(self):
        return len(self.items)
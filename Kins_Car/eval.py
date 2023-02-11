# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import pycocotools.mask as coco_mask
import numpy
import pickle as pkl
import os
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import cvbase as cvb
from tqdm import tqdm
import cv2


def get_kins_md5(kins_train_root):
    kins_train_md5 = {}
    for kins_imgfile in os.listdir(kins_train_root):
        if kins_imgfile.endswith(".png"):
            md5 = md5sum(os.path.join(kins_train_root, kins_imgfile))
            kins_train_md5[md5] = os.path.join(kins_train_root, kins_imgfile)

    return kins_train_md5


def md5sum(filename):
    fd = open(filename, "rb")
    fcont = fd.read()
    fd.close()
    fmd5 = hashlib.md5(fcont)
    return fmd5.hexdigest()


def get_IoU(pt_mask, gt_mask):
    # pred_mask  [..., Image_H, Image_W]
    # gt_mask   [..., Image_H, Image_W]
    pt_mask = pt_mask.astype(np.int32)
    gt_mask = gt_mask.astype(np.int32)
    SMOOTH = 1e-6
    intersection = (pt_mask & gt_mask).sum((-1, -2))  # [..., 1]
    union = (pt_mask | gt_mask).sum((-1, -2))  # [..., 1]

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # [..., 1]

    return iou


def get_crop_coord(binary_mask):
    x_min = np.min(np.where(binary_mask == 1)[0])
    x_max = np.max(np.where(binary_mask == 1)[0])

    y_min = np.min(np.where(binary_mask == 1)[1])
    y_max = np.max(np.where(binary_mask == 1)[1])

    return x_min, x_max, y_min, y_max


# In[2]:


def make_json_dict(imgs, anns):
    imgs_dict = {}
    anns_dict = {}
    for ann in anns:
        image_id = ann["image_id"]
        if not image_id in anns_dict:
            anns_dict[image_id] = []
            anns_dict[image_id].append(ann)
        else:
            anns_dict[image_id].append(ann)

    for img in imgs:
        image_id = img['id']
        imgs_dict[image_id] = img['file_name']

    return imgs_dict, anns_dict


# In[5]:


# match kins and kitti
kins_train_md5 = get_kins_md5("dataset/data/Kins/training/image_2")
print(len(kins_train_md5))


with open("dataset/data/car_data/data_config.pkl", "rb") as tmp_f:
    data_config = pkl.load(tmp_f)

test_path = data_config["test_video_path"]

test_Kitti_img_files = {}
for video_name in test_path:
    video_name = video_name.split("/")[-1]
    video_path = os.path.join(
        "dataset/data/Kitti/raw_video", "_".join(video_name.split("_")[:3]), video_name)

    for img_path in os.listdir(video_path):
        if img_path.endswith("png"):
            img_file = os.path.join(video_path, img_path)
            md5 = md5sum(img_file)
            if md5 in kins_train_md5:
                test_Kitti_img_files[img_file] = kins_train_md5[md5]


# In[8]:


# Kins
anns = cvb.load("dataset/data/Kins/instances_train.json")
imgs_info = anns['images']
anns_info = anns["annotations"]
imgs_dict, anns_dict = make_json_dict(imgs_info, anns_info)
to_ann_img_id_dict = {int(v[:-4]): k for k, v in imgs_dict.items()}


# In[37]:


all_convex_iou = []
all_convex_inv_iou = []
savos_iou = []
savos_inv_iou = []

for kitti_img, kins_img in tqdm(test_Kitti_img_files.items()):
    video_name = kitti_img.split("/")[-2]
    kitti_img_id = int(kitti_img.split("/")[-1][:-4])
    pred_res_file = os.path.join(
        "log_pred_res", video_name, "%d_pred_res.pkl" % kitti_img_id)
    if not os.path.isfile(pred_res_file):
        continue
    with open(pred_res_file, "rb") as tmp_f:
        pred_res = pkl.load(tmp_f)

    kins_img_id = int(kins_img.split("/")[-1][:-4])
    if not kins_img_id in to_ann_img_id_dict:
        continue

    gt_modal_res = anns_dict[to_ann_img_id_dict[kins_img_id]]

    img_ious = []
    img_inv_iou = []
    convex_ious = []
    convex_inv_ious = []

    all_vm = [val["vm"] for val in pred_res.values()]
    all_vm = sum(all_vm)

    for gt_obj in gt_modal_res:
        vm_code = gt_obj["inmodal_seg"]
        vm = coco_mask.decode(vm_code).astype(np.int32)
        height, width = gt_obj["inmodal_seg"]["size"]
        amodal_rle = coco_mask.frPyObjects(
            gt_obj['segmentation'], height, width)
        gt_fm = coco_mask.decode(amodal_rle)[:, :, 0].astype(np.int32)
        x_min, x_max, y_min, y_max = get_crop_coord(vm)

        match_candidate = []
        for k, pred_obj in pred_res.items():
            pred_vm = pred_obj["vm"].astype(np.int32)
            match_candidate.append([get_IoU(pred_vm, vm), k])
        match_candidate = sorted(match_candidate, key=lambda x: x[0])
        if match_candidate[-1][0] < 0.5:
            continue
        pred_obj = pred_res[match_candidate[-1][1]]
        pred_vm = pred_obj["vm"].astype(np.int32)

        # amodal pred
        other_visible_mask = all_vm - pred_vm
        pred_fm = (pred_obj["pred_fm"] > 0.5).astype(np.int32)
        # use the visible mask to fix the predict full mask
        pred_fm = pred_vm * (1-other_visible_mask) + \
            other_visible_mask * pred_fm
        img_ious.append(get_IoU(pred_fm.astype(np.int32), gt_fm))

        # amodal invisible
        img_inv_iou.append(
            get_IoU(pred_fm * other_visible_mask, gt_fm * other_visible_mask))

        # convex pred
        contours, hierarchy = cv2.findContours(pred_vm.astype(
            np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        convex_pred_fm = np.zeros(pred_vm.shape)
        contours = np.concatenate(contours, axis=0)
        hull = cv2.convexHull(contours)
        convex_pred_fm = cv2.fillPoly(np.zeros_like(convex_pred_fm), [hull], 1)
        convex_ious.append(get_IoU(convex_pred_fm, gt_fm))
        convex_inv_ious.append(
            get_IoU(convex_pred_fm * other_visible_mask, gt_fm * other_visible_mask))

    if len(img_ious) > 0:
        savos_iou.append(img_ious)
        savos_inv_iou.append(img_inv_iou)
        all_convex_iou.append(convex_ious)
        all_convex_inv_iou.append(convex_inv_ious)


# In[39]:


# calculate the mean iou of objects in each frame, and take average accross frames.
# we exclude the not occluded object.
def iou_metric1(ll):
    mIoU = []
    for val in ll:
        tmp = [iou for iou in val if iou != 1]
        if tmp:
            mIoU.append(np.mean(tmp))
    return np.mean(mIoU)

# calculate the iou of each objects in frames, and take average accross all objects.
# we exclude the not occluded object.


def iou_metric2(ll):
    mIoU = []
    for val in ll:
        mIoU.extend([iou for iou in val if iou != 1])
    return np.mean(mIoU)


# In[41]:


print("===mean iou accross frames (in paper)===")
print("Savos:", iou_metric1(savos_iou), iou_metric1(savos_inv_iou))
print("Convex:", iou_metric1(all_convex_iou), iou_metric1(all_convex_inv_iou))


# In[42]:


print("===mean iou accross objects===")
print("Savos:", iou_metric2(savos_iou), iou_metric2(savos_inv_iou))
print("Convex:", iou_metric2(all_convex_iou), iou_metric2(all_convex_inv_iou))

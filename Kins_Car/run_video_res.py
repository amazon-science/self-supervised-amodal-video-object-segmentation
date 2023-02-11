# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import sys
import pickle as pkl
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image
from moviepy.editor import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

sys.path.append("../")
from utils.utils import *


def get_color_dict(max_num):
    cNorm = colors.Normalize(vmin=0, vmax=max_num)
    jet = cm = plt.get_cmap('jet')
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    return scalarMap


def run_pred_res_video(video_name):

    pred_res_root = "log_pred_res"
    Kitti_video_root = "dataset/data/Kitti/raw_video"
    video_save_path = "log_video_pred_res"
    scalarMap = None
    rawvideo_root = os.path.join(Kitti_video_root, "_".join(
        video_name.split("_")[:3]), video_name)
    cur_video_save_path = os.path.join(video_save_path, video_name)
    if not os.path.exists(cur_video_save_path):
        os.makedirs(cur_video_save_path)

    for img_file in os.listdir(rawvideo_root):
        if img_file.find("png") == -1:
            continue
        img_id = int(img_file[:-4])
        base_image = np.asarray(Image.open(
            os.path.join(rawvideo_root, img_file)).convert("RGB"))
        base_image = base_image.copy()
        if os.path.isfile(os.path.join(pred_res_root, video_name, "%d_pred_res.pkl" % img_id)):

            pred_res = pkl.load(
                open(os.path.join(pred_res_root, video_name, "%d_pred_res.pkl" % img_id), "rb"))
            num_of_obj_in_video = list(pred_res.values())[
                0]["num_of_obj_in_video"]
            if scalarMap is None:
                scalarMap = get_color_dict(19)

            boundary_all = np.zeros_like(base_image)
            vm_all = np.zeros_like(base_image)

            all_vm = [val["vm"] for val in pred_res.values()]
            all_vm = sum(all_vm)
            all_vm = (all_vm > 0).astype(np.uint8)

            for obj_id in pred_res:
                colorVal = scalarMap.to_rgba(obj_id % 19)[:3]
                colorVal = (np.array(colorVal) * 255).astype(np.uint8)

                pred_fm = pred_res[obj_id]["pred_fm"]
                pred_fm = (pred_fm > 0.5).astype(np.uint8)
                ###
                vm = pred_res[obj_id]["vm"]
                vm = (vm > 0.5).astype(np.uint8)
                other_visible_mask = all_vm - vm
                pred_fm = vm * (1-other_visible_mask) + \
                    other_visible_mask * pred_fm
                pred_fm = pred_fm.astype(np.uint8)

                contours, hierarchy = cv2.findContours(
                    pred_fm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                # if len(contours) > 1:
                #     print("W")
                canvas = np.zeros(
                    (base_image.shape[0], base_image.shape[1])).astype(np.uint8)
                boundary = cv2.drawContours(canvas, contours, -1, [1, 1, 1], 2)
                # if np.sum(boundary) == 0:
                #     print("Warning! No boundary will be drawn")
                #     import pdb;pdb.set_trace()
                boundary_all[boundary > 0] = colorVal

                vm = pred_res[obj_id]["vm"]
                vm_cover = np.zeros_like(base_image)
                vm_cover[vm > 0] = colorVal
                vm_all += vm_cover

            # base_image = (base_image * 0.7 + vm_all * 0.3).astype(np.uint8)
            base_image = base_image.astype(np.uint8)
            base_image[boundary_all.sum(-1) >
                       0] = boundary_all[boundary_all.sum(-1) > 0]

        # import pdb;pdb.set_trace()
        new_img_path = os.path.join(
            cur_video_save_path, "%d_pred_res.png" % img_id)
        Image.fromarray(base_image).save(new_img_path)

    img2video(cur_video_save_path, video_name, "pred_fullmask")


def img2video(img_dir, video_name, mode):

    if not os.path.exists(os.path.join("log_video", video_name)):
        os.makedirs(os.path.join("log_video", video_name))

    filelist = [val for val in os.listdir(img_dir) if val.endswith(".png")]
    reverse = "reverse" in mode
    filelist = sorted(filelist, key=lambda x: int(
        x.split("_")[0]), reverse=reverse)
    fps = 5

    for item in filelist:
        img = cv2.imread(os.path.join(img_dir, item))
        break
    size = (img.shape[1], img.shape[0])

    video = cv2.VideoWriter(os.path.join("log_video", video_name, mode + ".avi"),
                            cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    for item in filelist:

        img = cv2.imread(os.path.join(img_dir, item))
        video.write(img)

    video.release()
    print("===video has been writed to %s" %
          os.path.join("log_video", video_name, mode + ".avi"))
    return


def multi_process_run():
    pred_res_root = "log_pred_res"

    tasks_num = len(os.listdir(pred_res_root)) * 3
    p = Pool(tasks_num)
    for video_name in os.listdir(pred_res_root):
        p.apply_async(run_pred_res_video, args=(video_name, ))

    p.close()
    p.join()
    print("==done==")


multi_process_run()

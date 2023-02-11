# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import pickle
import copy

import numpy as np
from numpy.core.records import array
import torch
import matplotlib.pyplot as plt
from skimage import transform
from pycocotools import mask as coco_mask


class Kitti(object):
    def __init__(self, data_dir, args, mode, subtest=None):

        self.mode = mode
        # self.mode = "train"
        self.device = "cpu"  # args.device
        self.dtype = args.dtype

        self.data_dir = data_dir
        self.data_summary = pickle.load(
            open(os.path.join(self.data_dir, self.mode+"_data.pkl"), "rb"))

        self.video_obj_lists = []
        for video_name in self.data_summary.keys():
            for obj_id in self.data_summary[video_name].keys():
                self.video_obj_lists.append(video_name + "-" + str(obj_id))

        self.img_path = "dataset/data/Kitti/raw_video"
        self.flow_path = "dataset/data/Kitti/Kitti_flow"
        self.flow_reverse_path = "dataset/data/Kitti/Kitti_reverse_flow"

        self.patch_h = args.patch_h
        self.patch_w = args.patch_w
        self.enlarge_coef = args.enlarge_coef
        self.train_seq_len = 30

        self.cur_video_name = None
        self.video_frames = None
        self.flows_frames = None
        self.flows_reverse_frames = None

        self.video_name_id_map = dict(
            zip(list(self.data_summary.keys()), range(len(self.data_summary))))

    def __len__(self):
        # print("dataset with num of data: %d" % self.num_objs)
        print(len(self.video_obj_lists))
        return len(self.video_obj_lists)

    def __getitem__(self, idx, specified_V_O_id=None):
        if specified_V_O_id is None:
            video_name, obj_id = self.video_obj_lists[idx].split("-")
        else:
            video_name, obj_id = specified_V_O_id.split("-")
        if video_name != self.cur_video_name:
            self.video_frames = self.getImg(video_name)
            self.flows_frames = self.getFlow(self.flow_path, video_name, 1)
            self.flows_reverse_frames = self.getFlow(
                self.flow_reverse_path, video_name, -1)
            self.cur_video_name = video_name
        video_frames = self.video_frames
        flows_frames = self.flows_frames
        flows_reverse_frames = self.flows_reverse_frames

        obj_patches = []
        obj_position = []
        counts = []
        vm_crop = []
        vm_no_crop = []
        flows_crop = []
        flows_no_crop = []
        flows_reverse_crop = []
        flows_reverse_nocrop = []
        loss_mask_weight = []

        # for evaluation
        video_ids = []
        object_ids = []
        frame_ids = []
        obj_dict = self.data_summary[video_name][int(obj_id)]
        timesteps = list(obj_dict.keys())
        assert np.all(np.diff(sorted(timesteps)) == 1)
        start_t, end_t = min(timesteps), max(timesteps)

        if self.mode != "test" and end_t - start_t > self.train_seq_len - 1:
            start_t = np.random.randint(start_t, end_t-(self.train_seq_len-2))
            end_t = start_t + self.train_seq_len - 1

        for t_step in range(start_t, end_t+1):
            Image_H, Image_W = obj_dict[t_step]["VM"]["size"]
            vm = coco_mask.decode(obj_dict[t_step]["VM"]).astype(bool)

            vx_min, vx_max, vy_min, vy_max = obj_dict[t_step]["VM_bbox"]
            # enlarge the bbox
            x_center = (vx_min + vx_max) // 2
            y_center = (vy_min + vy_max) // 2
            x_len = int((vx_max - vx_min) * self.enlarge_coef)
            y_len = int((vy_max - vy_min) * self.enlarge_coef)
            vx_min = max(0, x_center - x_len // 2)
            vx_max = min(Image_H, x_center + x_len // 2)
            vy_min = max(0, y_center - y_len // 2)
            vy_max = min(Image_W, y_center + y_len // 2)
            obj_position.append([vx_min, vx_max, vy_min, vy_max])

            # get visible mask
            vm_crop.append(vm[vx_min:vx_max+1, vy_min:vy_max+1])
            vm_no_crop.append(vm)

            # get loss mask
            loss_mask_weight.append(
                1 - coco_mask.decode(obj_dict[t_step]["loss_mask"]).astype(bool))

            # get patches and flow crop
            patch = video_frames[t_step][vx_min:vx_max+1, vy_min:vy_max+1]
            flow = flows_frames[t_step][vx_min:vx_max+1, vy_min:vy_max+1]
            flows_reverse = flows_reverse_frames[t_step][vx_min:vx_max +
                                                         1, vy_min:vy_max+1]

            obj_patches.append(patch)
            flows_crop.append(flow)
            flows_no_crop.append(flows_frames[t_step])
            flows_reverse_crop.append(flows_reverse)
            flows_reverse_nocrop.append(flows_reverse_frames[t_step])

            # for evaluation
            video_ids.append(self.video_name_id_map[video_name])
            object_ids.append(int(obj_id))
            frame_ids.append(t_step)
            counts.append(1)

        assert len(video_ids) >= 10, "maybe a bug in implementation"
        num_pad = max(self.train_seq_len - (end_t - start_t + 1), 0)
        for _ in range(num_pad):
            obj_position.append(copy.deepcopy(obj_position[-1]))
            vm_crop.append(copy.deepcopy(vm_crop[-1]))
            vm_no_crop.append(copy.deepcopy(vm_no_crop[-1]))

            loss_mask_weight.append(copy.deepcopy(loss_mask_weight[-1]))
            obj_patches.append(copy.deepcopy(obj_patches[-1]))

            flows_crop.append(copy.deepcopy(flows_crop[-1]))
            flows_no_crop.append(copy.deepcopy(flows_no_crop[-1]))
            flows_reverse_crop.append(copy.deepcopy(flows_reverse_crop[-1]))
            flows_reverse_nocrop.append(
                copy.deepcopy(flows_reverse_nocrop[-1]))

            video_ids.append(video_ids[-1])
            object_ids.append(object_ids[-1])
            frame_ids.append(frame_ids[-1] + 1)
            counts.append(0)

        obj_position = torch.from_numpy(
            np.array(obj_position)).to(self.dtype).to(self.device)
        counts = torch.from_numpy(np.array(counts)).to(
            self.dtype).to(self.device)
        loss_mask_weight = torch.from_numpy(
            np.array(loss_mask_weight)).to(self.dtype).to(self.device)

        # the patch, visible mask and flow crop of objects are in the same scale
        # while the full mask is not
        vm_term, obj_patches, flows_crop, flows_reverse_crop = self.rescale(
            vm_crop, obj_patches, flows_crop, flows_reverse_crop)
        for vt in vm_term:
            if vt.sum() == 0:
                assert False, "bug"
        # Seq_len * patch_h * patch_w * 3
        obj_temp = np.stack(obj_patches, axis=0)
        vm_temp = np.stack(vm_term, axis=0)  # Seq_len * patch_h * patch_w
        flows_crop = np.array(flows_crop)  # Seq_len * patch_h * patch_w * 2
        flows_reverse_crop = np.array(flows_reverse_crop)

        # add VM to be the 4th channel and the flow to be 5-6 th channel
        obj_patches = np.concatenate([obj_temp, np.expand_dims(
            vm_temp, axis=-1), flows_crop, flows_reverse_crop], axis=-1)
        obj_patches = torch.from_numpy(
            obj_patches).to(self.dtype).to(self.device)
        vm_no_crop = torch.from_numpy(np.array(vm_no_crop)).to(
            self.dtype).to(self.device)
        flows_no_crop = torch.from_numpy(
            np.array(flows_no_crop)).to(self.dtype).to(self.device)
        flows_reverse_no_crop = torch.from_numpy(
            np.array(flows_reverse_nocrop)).to(self.dtype).to(self.device)

        video_ids = torch.from_numpy(np.array(video_ids)).to(
            self.dtype).to(self.device)
        object_ids = torch.from_numpy(np.array(object_ids)).to(
            self.dtype).to(self.device)
        frame_ids = torch.from_numpy(np.array(frame_ids)).to(
            self.dtype).to(self.device)

        # one batch contains bz videos data
        # each video data contains num_obj fish object
        # each object is a dict with patches(6 channels),

        obj_data = {"input_obj_patches": obj_patches,
                    "vm_nocrop": vm_no_crop,
                    "obj_position": obj_position,
                    "flows_nocrop": flows_no_crop,
                    "flows_reverse_nocrop": flows_reverse_no_crop,
                    "loss_mask": loss_mask_weight,
                    "counts": counts,
                    "video_ids": video_ids,
                    "object_ids": object_ids,
                    "frame_ids": frame_ids,
                    }

        return obj_data

    def get_image_instances(self, video_name):
        objects_data = []
        for obj_id in self.data_summary[video_name].keys():
            specified_V_O_id = video_name + "-" + str(obj_id)
            obj_data = self.__getitem__(
                idx=None, specified_V_O_id=specified_V_O_id)
            timesteps = self.data_summary[video_name][obj_id].keys()
            obj_data["st"] = min(timesteps)
            obj_data["et"] = max(timesteps)
            obj_data["obj_id"] = obj_id
            objects_data.append(obj_data)
        return objects_data

    def fm_rescale(self, masks):

        for i, m in enumerate(masks):
            if m is None:
                continue
            h, w = masks[i].shape[:2]
            m = transform.rescale(m, (self.patch_h/h, self.patch_w/w))
            cur_h, cur_w = m.shape[:2]
            to_pad = ((0, max(self.patch_h-cur_h, 0)),
                      (0, max(self.patch_w-cur_w, 0)))
            m = np.pad(m, to_pad)[:self.patch_h, :self.patch_w]
            masks[i] = m
        return masks

    def rescale(self, masks, obj_patches=None, flows=None, flows_reverse=None):
        mask_count = [np.sum(m) if m is not None else 0 for m in masks]
        idx = np.argmax(mask_count)

        h, w = masks[idx].shape[:2]
        for i, m in enumerate(masks):
            if m is None:
                continue
            m = transform.rescale(m, (self.patch_h/h, self.patch_w/w))
            cur_h, cur_w = m.shape[:2]
            to_pad = ((0, max(self.patch_h-cur_h, 0)),
                      (0, max(self.patch_w-cur_w, 0)))
            m = np.pad(m, to_pad)[:self.patch_h, :self.patch_w]
            masks[i] = m

        for i, obj_pt in enumerate(obj_patches):
            obj_pt = transform.rescale(
                obj_pt, (self.patch_h/h, self.patch_w/w, 1))
            cur_h, cur_w = obj_pt.shape[:2]
            to_pad = ((0, max(self.patch_h-cur_h, 0)),
                      (0, max(self.patch_w-cur_w, 0)), (0, 0))
            obj_pt = np.pad(obj_pt, to_pad)[:self.patch_h, :self.patch_w, :3]
            obj_patches[i] = obj_pt

        for i, flow in enumerate(flows):
            flow = transform.rescale(flow, (self.patch_h/h, self.patch_w/w, 1))
            cur_h, cur_w = flow.shape[:2]
            to_pad = ((0, max(self.patch_h-cur_h, 0)),
                      (0, max(self.patch_w-cur_w, 0)), (0, 0))
            flow = np.pad(flow, to_pad)[:self.patch_h, :self.patch_w, :2]
            flows[i] = flow

        for i, flow_r in enumerate(flows_reverse):
            flow_r = transform.rescale(
                flow_r, (self.patch_h/h, self.patch_w/w, 1))
            cur_h, cur_w = flow_r.shape[:2]
            to_pad = ((0, max(self.patch_h-cur_h, 0)),
                      (0, max(self.patch_w-cur_w, 0)), (0, 0))
            flow_r = np.pad(flow_r, to_pad)[:self.patch_h, :self.patch_w, :2]
            flows_reverse[i] = flow_r

        return masks, obj_patches, flows, flows_reverse

    def getImg(self, v_id):
        imgs = []
        cur_img_path = os.path.join(
            self.img_path, "_".join(v_id.split("_")[:3]), v_id)
        imgs_list = os.listdir(cur_img_path)
        imgs_list = [val for val in imgs_list if val.find(".png") > -1]
        imgs_list = sorted(imgs_list, key=lambda x: int(x[:-4]))
        for sub_path in imgs_list:
            if sub_path.find("png") > -1:
                img_path = os.path.join(cur_img_path, sub_path)
                img_tmp = plt.imread(img_path)
                imgs.append(img_tmp)
        return imgs

    def getFlow(self, flow_path, v_id, shift):
        # \flow_t := F_{t->(t+1)}
        # copy \flow_{T-1} for \flow_T
        flows = []
        cur_flow_path = os.path.join(
            flow_path, "_".join(v_id.split("_")[:3]), v_id)
        flows_list = os.listdir(cur_flow_path)
        flows_list = sorted(flows_list, key=lambda x: int(x[:-4]))
        for sub_path in flows_list:
            if sub_path.find("flo") > -1:
                sub_flow_path = os.path.join(cur_flow_path, sub_path)
                f_tmp = self.readFlow(sub_flow_path)
                flows.append(f_tmp)
        if shift > 0:
            flows = flows + flows[-shift:]
        elif shift < 0:
            flows = flows[:-shift] + flows
        return flows

    def readFlow(self, fn):
        """ Read .flo file in Middlebury format"""
        # Code adapted from:
        # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

        # WARNING: this will work on little-endian architectures (eg Intel x86) only!
        # print 'fn = %s'%(fn)

        with open(fn, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                print('Magic number incorrect. Invalid .flo file')
                return None
            else:
                w = np.fromfile(f, np.int32, count=1)
                h = np.fromfile(f, np.int32, count=1)
                # print 'Reading %d x %d flo file\n' % (w, h)
                data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
                # Reshape data into 3D array (columns, rows, bands)
                # The reshape here is for visualization, the original code is (w,h,2)
                flow = np.resize(data, (int(h), int(w), 2))
                return flow

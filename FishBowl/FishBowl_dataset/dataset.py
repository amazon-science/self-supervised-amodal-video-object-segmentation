# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import pickle
import copy

import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage import transform
from pycocotools import _mask as coco_mask



class FishBowl(object):
    def __init__(self, data_dir, args, mode, subtest=None):
        # data_dir "dataset/data/val_data"
        self.datatype = data_dir.split("/")[-1].split("_")[0]
        self.img_path = os.path.join(data_dir, self.datatype+"_frames")
        self.flow_path = os.path.join(
            data_dir, self.datatype+"_flow/inference/run.epoch-0-flow-field")
        self.flow_reverse_path = os.path.join(
            data_dir, self.datatype+"_flow_reverse/inference/run.epoch-0-flow-field")

        self.mode = mode
        self.test_set = subtest

        self.data_summary = pickle.load(
            open(os.path.join(data_dir, self.datatype+"_data.pkl"), "rb"))
        self.obj_lists = list(self.data_summary.keys())
        self.device = "cpu"  # args.device
        self.dtype = args.dtype

        self.patch_h = args.patch_h
        self.patch_w = args.patch_w
        self.seq_len = 128 if self.mode == "test" else 40
        self.cur_vid = None
        self.video_frames = None
        self.flows_frames = None
        self.flows_reverse_frames = None

        self.enlarge_coef = args.enlarge_coef

    def decode2binarymask(self, masks):
        mask = coco_mask.decode(masks)
        binary_masks = mask.astype('bool')  # (Image_W,Image_H,128)
        binary_masks = binary_masks.transpose(
            2, 0, 1)  # (128, Image_W, Image_H)
        return binary_masks

    def __len__(self):
        # print("dataset with num of data: %d" % self.num_objs)
        print(len(self.obj_lists))
        return len(self.obj_lists) if self.mode != "valid" else len(self.obj_lists) // 2

    def __getitem__(self, idx):
        v_id, obj_id = self.obj_lists[idx].split("_")
        if v_id != self.cur_vid:
            # print("read video ", idx)
            self.video_frames = self.getImg(v_id)
            self.flows_frames = self.getFlow(self.flow_path, v_id, 1)
            self.flows_reverse_frames = self.getFlow(
                self.flow_reverse_path, v_id, -1)
            self.cur_vid = v_id
        video_frames = self.video_frames
        flows_frames = self.flows_frames
        flows_reverse_frames = self.flows_reverse_frames

        obj_patches = []
        fm_labels_crop = []
        fm_labels_no_crop = []
        obj_position = []
        counts = []
        fm_bx = []
        next_fm_bx = []
        occluded_ratio = []
        vm_crop = []
        vm_no_crop = []
        flows_crop = []
        flows_reverse_crop = []
        flows_no_crop = []
        flows_reverse_nocrop = []
        loss_mask_weight = []

        # for evaluation
        video_ids = []
        object_ids = []
        frame_ids = []

        obj_dict = self.data_summary[self.obj_lists[idx]]
        timesteps = list(obj_dict.keys())
        assert np.all(np.diff(sorted(timesteps)) == 1)
        start_t, end_t = min(timesteps), max(timesteps)
        if self.mode != "test" and end_t - start_t > self.seq_len - 1:
            start_t = np.random.randint(start_t, end_t-(self.seq_len-2))
            end_t = start_t + self.seq_len - 1

        for t_step in range(start_t, end_t+1):

            # get full mask label
            fm = self.decode2binarymask(obj_dict[t_step]["FM"])[0]  # 320, 480
            x_min, x_max, y_min, y_max = obj_dict[t_step]["FM_bx"]
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2
            x_len = int((x_max - x_min) * self.enlarge_coef)
            y_len = int((y_max - y_min) * self.enlarge_coef)
            x_min = max(0, x_center - x_len // 2)
            x_max = min(320, x_center + x_len // 2)
            y_min = max(0, y_center - y_len // 2)
            y_max = min(480, y_center + y_len // 2)

            fm_bx.append([x_min, x_max, y_min, y_max])
            fm_labels_crop.append(fm[x_min:x_max+1, y_min:y_max+1])
            fm_labels_no_crop.append(fm)

            # get visible mask
            vx_min, vx_max, vy_min, vy_max = obj_dict[t_step]["VM_bx"]
            # print(vx_min, vx_max, vy_min, vy_max)
            # enlarge the bbox
            x_center = (vx_min + vx_max) // 2
            y_center = (vy_min + vy_max) // 2
            x_len = int((vx_max - vx_min) * self.enlarge_coef)
            y_len = int((vy_max - vy_min) * self.enlarge_coef)
            vx_min = max(0, x_center - x_len // 2)
            vx_max = min(320, x_center + x_len // 2)
            vy_min = max(0, y_center - y_len // 2)
            vy_max = min(480, y_center + y_len // 2)

            obj_position.append([vx_min, vx_max, vy_min, vy_max])
            vm = self.decode2binarymask(obj_dict[t_step]["VM"])[0]
            vm_crop.append(vm[vx_min:vx_max+1, vy_min:vy_max+1])
            vm_no_crop.append(vm)

            # get loss mask
            loss_mask_weight.append(self.decode2binarymask(
                obj_dict[t_step]["loss_mask_weight"])[0])

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
            video_ids.append(int(v_id))
            object_ids.append(int(obj_id))
            frame_ids.append(t_step)
            counts.append(1)
        if True:
            # if self.mode != "test":
            num_pad = self.seq_len - (end_t - start_t + 1)
            for _ in range(num_pad):
                fm_bx.append(copy.deepcopy(fm_bx[-1]))
                fm_labels_crop.append(copy.deepcopy(fm_labels_crop[-1]))
                fm_labels_no_crop.append(copy.deepcopy(fm_labels_no_crop[-1]))

                obj_position.append(copy.deepcopy(obj_position[-1]))
                vm_crop.append(copy.deepcopy(vm_crop[-1]))
                vm_no_crop.append(copy.deepcopy(vm_no_crop[-1]))

                loss_mask_weight.append(copy.deepcopy(loss_mask_weight[-1]))
                obj_patches.append(copy.deepcopy(obj_patches[-1]))

                flows_crop.append(copy.deepcopy(flows_crop[-1]))
                flows_no_crop.append(copy.deepcopy(flows_no_crop[-1]))
                flows_reverse_crop.append(
                    copy.deepcopy(flows_reverse_crop[-1]))
                flows_reverse_nocrop.append(
                    copy.deepcopy(flows_reverse_nocrop[-1]))

                video_ids.append(video_ids[-1])
                object_ids.append(object_ids[-1])
                frame_ids.append(frame_ids[-1] + 1)
                counts.append(0)

        next_fm_bx = fm_bx[1:] + fm_bx[-1:]
        obj_position = torch.from_numpy(
            np.array(obj_position)).to(self.dtype).to(self.device)
        counts = torch.from_numpy(np.array(counts)).to(
            self.dtype).to(self.device)
        fm_bx = torch.from_numpy(np.array(fm_bx)).to(
            self.dtype).to(self.device)
        next_fm_bx = torch.from_numpy(np.array(next_fm_bx)).to(
            self.dtype).to(self.device)
        loss_mask_weight = torch.from_numpy(
            np.array(loss_mask_weight)).to(self.dtype).to(self.device)

        assert len(fm_labels_crop) > 0
        # the patch, visible mask and flow crop of objects are in the same scale
        # while the full mask is not
        vm_term, obj_patches, flows_crop, flows_reverse_crop = self.rescale(
            vm_crop, obj_patches, flows_crop, flows_reverse_crop)
        fm_labels_crop = self.fm_rescale(fm_labels_crop)

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
        fm_labels_no_crop = torch.from_numpy(
            np.array(fm_labels_no_crop)).to(self.dtype).to(self.device)
        flows_no_crop = torch.from_numpy(
            np.array(flows_no_crop)).to(self.dtype).to(self.device)
        flows_reverse_no_crop = torch.from_numpy(
            np.array(flows_reverse_nocrop)).to(self.dtype).to(self.device)
        fm_labels_crop = np.stack(fm_labels_crop, axis=0)
        fm_labels_crop = torch.from_numpy(
            np.array(fm_labels_crop)).to(self.dtype).to(self.device)

        video_ids = torch.from_numpy(np.array(video_ids)).to(
            self.dtype).to(self.device)
        object_ids = torch.from_numpy(np.array(object_ids)).to(
            self.dtype).to(self.device)
        frame_ids = torch.from_numpy(np.array(frame_ids)).to(
            self.dtype).to(self.device)

        # one batch contains bz videos data
        # each video data contains num_obj fish object
        # each object is a dict with patches(6 channels),
        # visible mask crop, flow crop, obj_position, fm_labels, occlued ratios
        # only variables be used in calculate the grad will be set to tensor.

        obj_data = {"input_obj_patches": obj_patches,
                    "vm_nocrop": vm_no_crop,
                    "obj_position": obj_position,
                    "fm_bx": fm_bx,
                    # "next_fm_bx": next_fm_bx,
                    "flows_nocrop": flows_no_crop,
                    "flows_reverse_nocrop": flows_reverse_no_crop,
                    "loss_mask": loss_mask_weight,
                    "fm_labels_crop": fm_labels_crop,
                    "full_mask": fm_labels_no_crop,
                    "counts": counts,
                    "video_ids": video_ids,
                    "object_ids": object_ids,
                    "frame_ids": frame_ids,
                    }

        return obj_data

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
        imgs_list = os.listdir(os.path.join(self.img_path, v_id))
        imgs_list.sort()
        for sub_path in imgs_list:
            img_path = os.path.join(self.img_path, v_id, sub_path)
            img_tmp = plt.imread(img_path)
            imgs.append(img_tmp)
        assert len(imgs) == 128
        return imgs

    def getFlow(self, flow_path, v_id, shift):
        # \flow_t := F_{t->(t+1)}
        # copy \flow_{T-1} for \flow_T
        flows = []
        flows_list = os.listdir(os.path.join(flow_path, v_id))
        flows_list.sort()
        for sub_path in flows_list:
            sub_flow_path = os.path.join(flow_path, v_id, sub_path)
            f_tmp = self.readFlow(sub_flow_path)  # Image_W Image_H 2
            flows.append(f_tmp)
        if shift > 0:
            flows = flows + flows[-shift:]
        elif shift < 0:
            flows = flows[:-shift] + flows
        assert len(flows) == 128
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
                flow = transform.rescale(
                    flow, (320/flow.shape[0], 480/flow.shape[1], 1), preserve_range=True)
                return flow

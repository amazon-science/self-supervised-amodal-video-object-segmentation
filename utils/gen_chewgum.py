# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import random
import cv2
import copy
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle
import multiprocessing
import argparse
from tqdm import tqdm

color = np.array([200,200,200]).astype(np.uint8)
canvas = np.zeros((56, 50,3)).astype(np.uint8)
canvas = cv2.circle(canvas, (28, 25), 8, (int(color[0]), int(color[1]), int(color[2])), 0)

pos = np.where(canvas[:,:,0]==200)
vertex_pool = np.array([val for val in zip(pos[0], pos[1])])
# rearrange
vertex_pool = [[17, 28], [18, 27], [18, 26], [18, 25], [19, 24], [19, 23], [20, 22],
               [21, 22], [22, 21], [23, 21], [24, 21], [25, 20], [26, 21], [27, 21],
               [28, 21], [29, 22], [30, 22], [31, 23], [31, 24], [32, 25], [32, 26],
               [32, 27], [33, 28], [32, 29], [32, 30], [32, 31], [31, 32], [31, 33],
               [30, 34], [29, 34], [28, 35], [27, 35], [26, 35], [25, 36], [24, 35],
               [23, 35], [22, 35], [21, 34], [20, 34], [19, 33], [19, 32], [18, 31],
               [18, 30], [18, 29]]
vertex_pool = np.array(vertex_pool)
vertex_pool = vertex_pool[:,::-1]
index_pool = list(range(len(vertex_pool)))

def draw_obj(np_canvas, st, obj_shape, color):
    # Fix bbox side as 16*16
    
    vertices = np.array(obj_shape.astype(np.int) - np.array([28, 25]).astype(np.int)+st+8, np.int32)
    pts = vertices.reshape((-1, 1, 2))
    np_canvas = cv2.fillPoly(np_canvas, [pts], color=(int(color[0]), int(color[1]), int(color[2])))
    
    return np_canvas

def draw_video(root, video_idx):
    shift1 = 8 * np.int(np.random.random()<0.5)
    shift2 = 8 * np.int(np.random.random()<0.5)
    num_objs = 2

    video_sidx = '%05d' % (int(video_idx))
    video_dir = os.path.join(root, video_sidx)
    os.makedirs(video_dir, exist_ok=True)

    # Draw Canvas
    np_canvas = np.zeros([50, 56, 3]).astype(np.uint8)
    canvas_color = np.random.randint(128, size=3)
    np_canvas[:,:,:] = canvas_color

    # Sample object shape and color
    obj_shapes = []
    obj_colors = []
    occlude_color = np.random.randint(256, size=3).astype(np.int)
    for i in range(num_objs):
        N = np.random.randint(7, 12)
        index_sample = np.random.choice(index_pool, size=N, replace=False)
        index_sample = sorted(index_sample)
        vertex_sample = vertex_pool[index_sample]
        obj_shapes.append(vertex_sample)
        obj_colors.append(np.random.randint(256, size=3).astype(np.int))

    # Sample which one is on top
    orders = np.arange(num_objs)
    np.random.shuffle(orders)

    # set the centers
    # 56 * 50 in cv2
    start_rotate_point = [np.random.choice([20, 36]), np.random.choice([17, 33])]
    obj_centers = np.array([[28, 25], start_rotate_point])

    # We set the frames to be 40, the last 20 is the same with the first 20.
    obj_direct = np.array([[0, 0], [-2, 0]])
    for f_idx in range(40):
        start_points = obj_centers - 8
        cur_canvas = copy.deepcopy(np_canvas)

        for ii, idx in enumerate(orders):
            cur_canvas = draw_obj(cur_canvas, start_points[idx], obj_shapes[idx], obj_colors[idx])

            # TODO: Here please double check whether the order is correct (may due to differnt settings !!!)
            if ii == 0:
                occ_shape = np.array([[28, 25], [20, 25], [20, 17], [28, 17]])
                st = [start_points[idx][0] + shift1,  start_points[idx][1] + shift2]
                cur_canvas = draw_obj(cur_canvas, st, occ_shape, occlude_color)

        # directly crop the boxes out
        img_objs = []
        for idx in range(num_objs):
            img_objs.append(cur_canvas[start_points[idx][1]:start_points[idx][1]+16, start_points[idx][0]:start_points[idx][0]+16, :])

        full_mask = []
        for idx in range(num_objs):
            mask = np.zeros((50, 56, 3)).astype(np.uint8)
            # TODO maybe wrong in three case
            mask = draw_obj(mask, start_points[idx], obj_shapes[idx], (10,10,10))
            mask = (mask[:,:,0] > 0).astype(np.int)
            full_mask.append(mask)
        full_mask = np.stack(full_mask, axis=0)

        obj_dict = {'order': orders, 'shape': obj_shapes,
                    'color': obj_colors, 'st': start_points,
                    'end': start_points+16, 'img_obj': img_objs,
                    'full_mask': full_mask}

        im = Image.fromarray(cur_canvas)
        filename = os.path.join(video_dir, '%05d.png' % (int(f_idx)))
        im.save(filename)

        pkl_filename = os.path.join(video_dir, '%05d.pkl' % (int(f_idx)))
        pickle.dump(obj_dict, open(pkl_filename, 'wb'))

        # for idx in range(num_objs):
        #     if obj_centers[idx][0] + obj_direct[idx][0] > 48 or obj_centers[idx][0] + obj_direct[idx][0] < 8:
        #         obj_direct[idx][0] *= -1
        #     if obj_centers[idx][1] + obj_direct[idx][1] > 42 or obj_centers[idx][1] + obj_direct[idx][1] < 8:
        #         obj_direct[idx][1] *= -1

        #     obj_centers[idx][0] += obj_direct[idx][0]
        #     obj_centers[idx][1] += obj_direct[idx][1]

        left_b, right_b =  20, 36
        down_b, up_b = 17, 33
        if obj_centers[1][0] + obj_direct[1][0] > right_b or obj_centers[1][0] + obj_direct[1][0] < left_b:
            obj_direct[1][0] = 0
            obj_direct[1][1] = 2 * (1 if 25 > obj_centers[1][1] else -1)
        
        if obj_centers[1][1] + obj_direct[1][1] > up_b or obj_centers[1][1] + obj_direct[1][1] < down_b:
            obj_direct[1][1] = 0
            obj_direct[1][0] = 2 * (1 if 28 > obj_centers[1][0] else -1)

        obj_centers[1][0] += obj_direct[1][0]
        obj_centers[1][1] += obj_direct[1][1]

      

def to_video(save_path, picture_path):
    # to generate a video from the temporary picture directory
    pictures = {}
    for pic in os.listdir(picture_path):
        if pic.endswith(".png"): # to ignore some .* files
            pictures[int(pic.split(".png")[0])] = pic
            sorted_pictures = [pictures[k] for k in sorted(pictures.keys())]

    fps = int(40 / 8)
    size = (56, 50)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(save_path, fourcc, fps, size)

    for item in sorted_pictures:
        item = os.path.join(picture_path, item)
        img = cv2.imread(item)
        video.write(img)

    video.release()
    return 1


if __name__ == "__main__":
    # e.g. python gen_amodal_2objs.py --dataset amodal_2objs_random_shape --train_val_test_num "10000 500 1000"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        type=str,
                        default="amodal_2objs_random_shape",
                        help="path to dataset")
    parser.add_argument("--train_val_test_num",
                        type=str,
                        default="10000 500 1000",
                        help="number of train, val and test data to generate")
    args = parser.parse_args()

    save_path = os.path.join("data/datasets", args.dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    num_train_data, num_val_data, num_test_data = [int(val) for val in args.train_val_test_num.split(" ")]

    print("====generate train samples====")
    train_data_path = os.path.join(save_path, "train")

    if not os.path.exists(os.path.join(save_path, "train_video_examples")):
        os.mkdir(os.path.join(save_path, "train_video_examples"))

    for i in tqdm(range(num_train_data)):
        # draw pictures
        draw_video(train_data_path, i)

        if i < 10:
            # to videos # Just be examples
            picture_path = os.path.join(train_data_path, "%05d" % i)
            video_path = os.path.join(save_path, "train_video_examples/{}.mp4".format(i))

            to_video(video_path, picture_path)
    print("done and saved to %s" % train_data_path)

    print("====generate val samples====")
    val_data_path = os.path.join(save_path, "val")

    for i in tqdm(range(num_val_data)):
        # draw pictures
        draw_video(val_data_path, i)
    print("done and saved to %s" % val_data_path)

    print("====generate test samples====")
    test_data_path = os.path.join(save_path, "test")

    if not os.path.exists(os.path.join(save_path, "test_video_examples")):
        os.mkdir(os.path.join(save_path, "test_video_examples"))
    
    for i in tqdm(range(num_test_data)):
        draw_video(test_data_path, i)

        if i < 10:
            # to videos # Just be examples
            picture_path = os.path.join(test_data_path, "%05d" % i)
            video_path = os.path.join(save_path, "test_video_examples/{}.mp4".format(i))
            os.mkdir(video_path) if not os.path.exists(video_path) else None
            to_video(video_path, picture_path)
    print("done and saved to %s" % test_data_path)
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import pandas as pd
import numpy as np


def print_res(cur_df, index, metric):
    test_df = pd.read_csv("test_files/test_subtest.csv")
    test_df = test_df[(test_df["OCC_R"] > 0.1) & (test_df["OCC_R"] < 0.7)]
    test_df = test_df.loc[:, ["v_id", "obj_id", "Type", "frame_id", "test"]]

    iou_tmp = cur_df.drop(["Unnamed: 0"], axis=1)
    test_tmp = pd.merge(test_df, iou_tmp, how="left", on=[
                        "v_id", "obj_id", "frame_id"])

    # if test_tmp[test_tmp[index].isnull()].shape[0]!=0:
    #     print("Error")
    if metric == "metric1":
        test_tmp = test_tmp[test_tmp["test"] == 1]
        test_tmp = test_tmp.groupby(["v_id", "frame_id"])[index].agg("mean")
        res = test_tmp.mean()

    elif metric == "metric2":
        res = test_tmp[test_tmp["test"] == 1][index].mean()

    print(index, np.round(res, 4))

    # return test_tmp[test_tmp["test"]==1][index].sum(), test_tmp[test_tmp["test"]==1].shape[0]

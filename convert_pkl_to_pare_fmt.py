import joblib
from glob import glob
# import torch
import numpy as np
import os

input_dir = "outputs/example_data/videos/sample_video_frames"
output_dir = "outputs/example_data/videos/sample_video_frames_pare_fmt_v02"

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

for _4dh_pkl_paths in glob(f"{input_dir}/*out.pkl"):
    pare_fmt = {}
    pkl_fn = "_".join(_4dh_pkl_paths.split("/")[-1].split(".")[:-1])
    _4dh_out = joblib.load(_4dh_pkl_paths)
    # _4dh_cam = joblib.load(_4dh_pkl_paths.replace("out.pkl", "cam.pkl"))
    
    concat_v1 = np.concatenate((_4dh_out['pred_smpl_params']['global_orient'].cpu().numpy(), _4dh_out['pred_smpl_params']['body_pose'].cpu().numpy()), axis=1)
    concat_v2 = np.concatenate((_4dh_out['pred_smpl_params']['body_pose'].cpu().numpy(), _4dh_out['pred_smpl_params']['global_orient'].cpu().numpy()), axis=1)
    pare_fmt['pred_pose'] = concat_v2
    pare_fmt['pred_cam'] = _4dh_out['pred_cam'].cpu().numpy()
    
    joblib.dump(pare_fmt, os.path.join(output_dir, pkl_fn + ".pkl"))

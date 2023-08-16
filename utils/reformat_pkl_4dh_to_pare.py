import os
import argparse
import joblib
import numpy as np
from glob import glob

def reformat_pkl_4dh_to_pare(_4dh_pkl):
    """ Change format of pkl from 4DHumans pkl to PARE"""
    pare_fmt_pkl = {}
    pare_fmt_pkl['pred_pose'] = np.concatenate((_4dh_pkl['pred_smpl_params']['body_pose'].cpu().numpy(), _4dh_pkl['pred_smpl_params']['global_orient'].cpu().numpy()), axis=1)
    pare_fmt_pkl['pred_cam'] = _4dh_pkl['pred_cam'].cpu().numpy()
    pare_fmt_pkl['pred_cam_t'] = _4dh_pkl['pred_cam_t'].cpu().numpy()

    return pare_fmt_pkl

if __name__ == "__main__":
    # input_dir = "outputs/example_data/videos/sample_video_frames"
    # output_dir = "outputs/example_data/videos/sample_video_frames_pare_fmt_v02"parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--args.input_dir')
    parser.add_argument('--args.output_dir')
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    for _4dh_pkl_paths in glob(f"{args.input_dir}/*out.pkl"):
        pkl_fn = "_".join(_4dh_pkl_paths.split("/")[-1].split(".")[:-1])
        _4dh_out = joblib.load(_4dh_pkl_paths)

        pare_fmt_pkl = reformat_pkl_4dh_to_pare(_4dh_out)
        joblib.dump(pare_fmt_pkl, os.path.join(args.output_dir, pkl_fn + ".pkl"))

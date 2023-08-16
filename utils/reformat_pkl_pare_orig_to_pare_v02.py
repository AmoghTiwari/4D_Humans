import os
import joblib

dataset_dir = "/data/groot/Datasets/3DPW_Dataset/imageFiles"
pkl_input_dir = "/data/amogh/projects/4D_Humans/outputs/3DPW/validation"
pkl_output_dir = "/data/amogh/projects/4D_Humans/outputs/3DPW/validation"

SKIPPED = []
for seq_name in sorted(os.listdir(pkl_input_dir)):
    # print(seq_name)
    for img_name in sorted(os.listdir(os.path.join(dataset_dir, seq_name))):
        pkls_fbn = os.path.join(pkl_input_dir, seq_name, "_".join(img_name.split(".")[:-1]))
        orig_out_fp = pkls_fbn+"_out.pkl"
        cam_fp = pkls_fbn+"_cam.pkl"
        pare_fmt_orig_fp = pkls_fbn+"_pare_fmt.pkl"
        try:
            pare_fmt_orig = joblib.load(pare_fmt_orig_fp)
            cam = joblib.load(cam_fp)
            orig_out = joblib.load(orig_out_fp)

            pare_fmt_new = {}
            pare_fmt_new['pred_pose'] = pare_fmt_orig['pred_pose']
            pare_fmt_new['pred_cam'] = pare_fmt_orig['pred_cam']
            pare_fmt_new['pred_cam_t'] = orig_out['pred_cam_t'].cpu().numpy()
            pare_fmt_new['orig_cam'] = cam
            print(pkls_fbn+"_pare_fmt_v02.pkl")
            joblib.dump(pare_fmt_new, pkls_fbn+"_pare_fmt_v02.pkl")
        except:
            SKIPPED.append(pare_fmt_orig_fp)
            SKIPPED.append(cam_fp)
            SKIPPED.append(orig_out_fp)
        # break

print()
print()
print("SKIPPED ---> ", SKIPPED)


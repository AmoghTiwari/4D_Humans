img_dir="/data/groot/Datasets/3DPW_Dataset/imageFiles"
export CUDA_VISIBLE_DEVICES=2

eval "$(conda shell.bash hook)"
which conda
conda activate 4D_humans_env
which python
cd ../

for seq_name in courtyard_rangeOfMotions_01 courtyard_basketball_01 courtyard_dancing_00 courtyard_drinking_00 courtyard_hug_00 courtyard_jumpBench_01 downtown_walkDownhill_00 outdoors_crosscountry_00 outdoors_freestyle_01 outdoors_golf_00 outdoors_parcours_00 outdoors_parcours_01

    do
       echo "##### Processing $img_dir/$seq_name #####"
       python demo_w_pkl.py --img_folder $img_dir/$seq_name --out_folder outputs/3DPW/validation/$seq_name --batch_size=48 --save_pkl
       echo
       echo
    done


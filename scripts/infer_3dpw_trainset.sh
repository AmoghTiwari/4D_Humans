img_dir="/data/groot/Datasets/3DPW_Dataset/imageFiles"

eval "$(conda shell.bash hook)"
which conda
conda activate 4D_humans_env
which python
cd ../

# for i in courtyard_bodyScannerMotions_00 courtyard_arguing_00 courtyard_backpack_00 courtyard_basketball_00 courtyard_box_00 courtyard_capoeira_00 courtyard_captureSelfies_00 courtyard_dancing_01 courtyard_giveDirections_00 courtyard_golf_00 courtyard_goodNews_00 courtyard_jacket_00 courtyard_laceShoe_00 courtyard_rangeOfMotions_00 courtyard_relaxOnBench_00 courtyard_relaxOnBench_01 courtyard_shakeHands_00 courtyard_warmWelcome_00 outdoors_climbing_00 outdoors_climbing_01 outdoors_climbing_02 outdoors_freestyle_00 outdoors_slalom_00 outdoors_slalom_01; do echo $i; done

for seq_name in courtyard_bodyScannerMotions_00; 
    do
       echo $img_dir/$seq_name
       python demo_w_pkl.py --img_folder $img_dir/$seq_name --out_folder outputs/3DPW/train/$seq_name --batch_size=48 --save_pkl
       echo
       echo
    done


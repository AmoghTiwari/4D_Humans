img_dir="/data/groot/Datasets/3DPW_Dataset/imageFiles"
export CUDA_VISIBLE_DEVICES=1

eval "$(conda shell.bash hook)"
which conda
conda activate 4D_humans_env
which python
cd ../

for seq_name in downtown_windowShopping_00 downtown_arguing_00 downtown_bar_00 downtown_bus_00 downtown_cafe_00 downtown_car_00 downtown_crossStreets_00 downtown_downstairs_00 downtown_enterShop_00 downtown_rampAndStairs_00 downtown_runForBus_00 downtown_runForBus_01 downtown_sitOnStairs_00 downtown_stairs_00 downtown_upstairs_00 downtown_walkBridge_01 downtown_walking_00 downtown_walkUphill_00 downtown_warmWelcome_00 downtown_weeklyMarket_00 flat_guitar_01 flat_packBags_00 office_phoneCall_00 outdoors_fencing_01 
    do
       echo "##### Processing $img_dir/$seq_name #####"
       python demo_w_pkl.py --img_folder $img_dir/$seq_name --out_folder outputs/3DPW/test/$seq_name --batch_size=48 --save_pkl
       echo
       echo
    done


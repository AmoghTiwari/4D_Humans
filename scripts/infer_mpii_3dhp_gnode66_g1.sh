#!/bin/bash

###### CHANGES BLOCK #####
# 1. Change data_base_dir variable
# 2. Change the CUDA_VISIBLE_DEVICES thing
# 3. Change the bash script file name
# 4. Change the log file name
# 5. Change the subject number and sequence number
# 6. For new machine, replace the dummy conda environment also
###### CHANGES BLOCK #####

data_base_dir="/home/cvit/Datasets/mpi_inf_3dhp"
export CUDA_VISIBLE_DEVICES=1

eval "$(conda shell.bash hook)"
conda activate 4D_humans_env
cd ../


run_start_dt=$(date)
echo "############################## ##############################" >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
echo "Present run started at: $run_start_dt" >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
echo "############################## ##############################" >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt

subjects=("S3")
seq_nums=("Seq2")
# vid_names=(video_2.avi video_4.avi)
vid_names=(video_2.avi)
echo "Subjects: ${subjects[@]}"  >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
echo "Sequences: ${seq_nums[@]}"  >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
echo "vid_names: ${vid_names[@]}"  >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt

for subject in ${subjects[@]}; do
    for seq_num in ${seq_nums[@]}; do
        data_dir=$data_base_dir/$subject/$seq_num/imageSequence
        # for vid_fp in $data_dir/*; do 
        for vid_name in ${vid_names[@]}; do
            #vid_fp=(See above)
            vid_fp=$data_base_dir/$subject/$seq_num/imageSequence/$vid_name
            echo "########## ##########" >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
            echo Processing: $vid_fp >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
            echo "########## ##########" >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
            echo >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
            vid_fbn=${vid_fp/.avi//}
            echo vid_fbn: $vid_fbn >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
            vid_bn=(${vid_fbn//\// })
            vid_bn=${vid_bn[-1]}
            echo vid_bn: $vid_bn >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
            echo  >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt

            ##### FFMPEG BLOCK #####
            mkdir $vid_fbn
            conda deactivate
            ffmpeg_start_dt=$(date)
            echo "Running ffmpeg, start time: $ffmpeg_start_dt" >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
            ffmpeg -i $vid_fp -r 30000/1001 -f image2 -v error $vid_fbn/%06d.jpg
            ffmpeg_end_dt=$(date)
            echo "Finished ffmpeg, end time: $ffmpeg_end_dt" >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
            num_frames=$(ls $vid_fbn | wc -l)
            echo "Processed $num_frames frames" >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
            echo >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
            conda activate 4D_humans_env
            ##### FFMPEG BLOCK #####
            
            inference_start_dt=$(date)
            echo "Running inference, start time: $inference_start_dt" >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
            python demo_w_pkl.py --img_folder $vid_fbn --out_folder outputs/mpi_inf_3dhp/$subject/$seq_num/$vid_bn --batch_size=48 --save_pkl
            inference_end_dt=$(date)
            echo "Finished inference, end time: $inference_end_dt" >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
            echo Processed $(ls $vid_fbn | wc -l) frames >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
            echo >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
            echo "Removing $vid_fbn" >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
            rm -r $vid_fbn

            echo "########## ##########" >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
            echo "STATISTICS" >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
            echo "########## ##########" >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
            echo "ffmpeg start time: $ffmpeg_start_dt" >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
            echo "ffmpeg end time: $ffmpeg_end_dt" >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
            echo "inference start time: $inference_start_dt" >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
            echo "inference end time: $inference_end_dt" >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
            echo Processed $num_frames frames >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
            echo >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
            echo >> scripts/log_infer_mpii_3dhp_gnode66_g1.txt
            # exit 0
        done
    done
done

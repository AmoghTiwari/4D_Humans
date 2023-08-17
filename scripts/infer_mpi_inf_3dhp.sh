data_base_dir="/data/groot/Datasets/mpi_inf_3dhp/"
export CUDA_VISIBLE_DEVICES=1

eval "$(conda shell.bash hook)"
conda activate 4D_humans_env
cd ../


run_start_dt=$(date)
echo "############################## ##############################" >> scripts/log_infer_mpi_inf_3dhp.txt
echo "Present run started at: $run_start_dt" >> scripts/log_infer_mpi_inf_3dhp.txt
echo "############################## ##############################" >> scripts/log_infer_mpi_inf_3dhp.txt

subjects=("S1")
seq_nums=("Seq1")
vid_names=(video_5.avi video_6.avi video_7.avi)
echo "Subjects: ${subjects[@]}"  >> scripts/log_infer_mpi_inf_3dhp.txt
echo "Sequences: ${seq_nums[@]}"  >> scripts/log_infer_mpi_inf_3dhp.txt
echo "vid_names: ${vid_names[@]}"  >> scripts/log_infer_mpi_inf_3dhp.txt

for subject in ${subjects[@]}; do
    for seq_num in ${seq_nums[@]}; do
        data_dir=$data_base_dir/$subject/$seq_num/imageSequence
        # for vid_fp in $data_dir/*; do 
        for vid_name in ${vid_names[@]}; do
            #vid_fp=(See above)
            vid_fp=$data_base_dir/$subject/$seq_num/imageSequence/$vid_name
            echo "########## ##########" >> scripts/log_infer_mpi_inf_3dhp.txt
            echo Processing: $vid_fp >> scripts/log_infer_mpi_inf_3dhp.txt
            echo "########## ##########" >> scripts/log_infer_mpi_inf_3dhp.txt
            echo >> scripts/log_infer_mpi_inf_3dhp.txt
            vid_fbn=${vid_fp/.avi//}
            echo vid_fbn: $vid_fbn >> scripts/log_infer_mpi_inf_3dhp.txt
            vid_bn=(${vid_fbn//\// })
            vid_bn=${vid_bn[-1]}
            echo vid_bn: $vid_bn >> scripts/log_infer_mpi_inf_3dhp.txt
            echo  >> scripts/log_infer_mpi_inf_3dhp.txt

            ##### FFMPEG BLOCK #####
            mkdir $vid_fbn
            conda activate mps_env
            ffmpeg_start_dt=$(date)
            echo "Running ffmpeg, start time: $ffmpeg_start_dt" >> scripts/log_infer_mpi_inf_3dhp.txt
            ffmpeg -i $vid_fp -r 30000/1001 -f image2 -v error $vid_fbn/%06d.jpg
            ffmpeg_end_dt=$(date)
            echo "Finished ffmpeg, end time: $ffmpeg_end_dt" >> scripts/log_infer_mpi_inf_3dhp.txt
            num_frames=$(ls $vid_fbn | wc -l)
            echo "Processed $num_frames frames" >> scripts/log_infer_mpi_inf_3dhp.txt
            echo >> scripts/log_infer_mpi_inf_3dhp.txt
            conda deactivate
            ##### FFMPEG BLOCK #####
            
            inference_start_dt=$(date)
            echo "Running inference, start time: $inference_start_dt" >> scripts/log_infer_mpi_inf_3dhp.txt
            python demo_w_pkl.py --img_folder $vid_fbn --out_folder outputs/mpi_inf_3dhp/$subject/$seq_num/$vid_bn --batch_size=48 --save_pkl
            inference_end_dt=$(date)
            echo "Finished inference, end time: $inference_end_dt" >> scripts/log_infer_mpi_inf_3dhp.txt
            echo Processed $(ls $vid_fbn | wc -l) frames >> scripts/log_infer_mpi_inf_3dhp.txt
            echo >> scripts/log_infer_mpi_inf_3dhp.txt
            echo "Removing $vid_fbn" >> scripts/log_infer_mpi_inf_3dhp.txt
            rm -r $vid_fbn

            echo "########## ##########" >> scripts/log_infer_mpi_inf_3dhp.txt
            echo "STATISTICS" >> scripts/log_infer_mpi_inf_3dhp.txt
            echo "########## ##########" >> scripts/log_infer_mpi_inf_3dhp.txt
            echo "ffmpeg start time: $ffmpeg_start_dt" >> scripts/log_infer_mpi_inf_3dhp.txt
            echo "ffmpeg end time: $ffmpeg_end_dt" >> scripts/log_infer_mpi_inf_3dhp.txt
            echo "inference start time: $inference_start_dt" >> scripts/log_infer_mpi_inf_3dhp.txt
            echo "inference end time: $inference_end_dt" >> scripts/log_infer_mpi_inf_3dhp.txt
            echo Processed $num_frames frames >> scripts/log_infer_mpi_inf_3dhp.txt
            echo >> scripts/log_infer_mpi_inf_3dhp.txt
            echo >> scripts/log_infer_mpi_inf_3dhp.txt
            # exit 0
        done
    done
done

data_base_dir="/data/groot/Datasets/mpi_inf_3dhp/"
export CUDA_VISIBLE_DEVICES=2

eval "$(conda shell.bash hook)"
conda activate 4D_humans_env
cd ../

vid_fp=$data_dir/video_1.avi

subjects=("S1")
seq_nums=("Seq1")
for subject in $subjects; do
    for seq_num in $seq_nums; do
        data_dir=$data_base_dir/$subject/$seq_num/imageSequence
        for vid_fp in $data_dir/*; do 
            #vid_fp=(See above)
            echo Processing: $vid_fp
            echo 
            vid_fbn=${vid_fp/.avi//}
            echo vid_fbn: $vid_fbn
            vid_bn=(${vid_fbn//\// })
            vid_bn=${vid_bn[-1]}
            echo vid_bn: $vid_bn
            echo 

            ##### FFMPEG BLOCK #####
            mkdir $vid_fbn
            conda activate mps_env
            ffmpeg_start_dt=$(date)
            echo "Running ffmpeg, start time: $ffmpeg_start_dt"
            ffmpeg -i $vid_fp -r 30000/1001 -f image2 -v error $vid_fbn/%06d.jpg
            ffmpeg_end_dt=$(date)
            echo "Finished ffmpeg, end time: $ffmpeg_end_dt"
            num_frames=$(ls $vid_fbn | wc -l)
            echo "Processed $num_frames frames"
            echo
            conda deactivate
            ##### FFMPEG BLOCK #####
            
            inference_start_dt=$(date)
            echo "Running inference, start time: $inference_start_dt"
            python demo_w_pkl.py --img_folder $vid_fbn --out_folder outputs/mpi_inf_3dhp/$subject/$seq_num/$vid_bn --batch_size=48 --save_pkl
            inference_end_dt=$(date)
            echo "Finished inference, start time: $inference_end_dt"
            echo Processed $(ls $vid_fbn | wc -l) frames
            echo
            echo "Removing $vid_fbn"
            rm -r $vid_fbn

            echo "########## ##########"
            echo "STATISTICS"
            echo "########## ##########"
            echo "ffmpeg start time: $ffmpeg_start_dt"
            echo "ffmpeg end time: $ffmpeg_end_dt"
            echo "inference start time: $inference_start_dt"
            echo "inference start time: $inference_end_dt"
            echo Processed $num_frames frames
            echo
            exit 0
        done
    done
done

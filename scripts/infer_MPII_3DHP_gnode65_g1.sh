#!/bin/bash

########## ########## CHANGES BLOCK ########## ##########
dataset_name="MPII_3DHP"
machine_name="gnode65"
gpu_nums=(1)
subjects=(S6)
seq_nums=(Seq1)
vid_names=(video_2.avi video_4.avi)

if [[ "$machine_name" == "groot" ]]; then
    data_base_dir=/data/groot/Datasets/$dataset_name/
    code_base_dir=/data/amogh/projects/4D_Humans
elif [[ "$machine_name" == "gnode65" ]]; then
    data_base_dir=/home/cvit/Datasets/$dataset_name/
    code_base_dir=/home/cvit/amogh/projects/4D_Humans
elif [[ "$machine_name" == "gnode66" ]]; then
    data_base_dir=/home/cvit/Datasets/$dataset_name/
    code_base_dir=/home/cvit/amogh/projects/4D_Humans
# elif [[ "$machine_name" == "ada" ]]; then
#     echo ada
else
    echo Unknown machine name specified. Exitting ... !!!
    exit
fi

output_base_dir=$code_base_dir/outputs/$dataset_name
log_base_dir=$code_base_dir/logs
log_fp=$log_base_dir/log_infer_$dataset_name\_$machine_name\_g$gpu_nums\.txt
cuda_visible_devices=${gpu_nums[@]}
########## ########## CHANGES BLOCK ########## ##########

########## LOGGER BLOCK ##########
mkdir $log_base_dir
echo "Logging details to $log_fp"

run_start_dt=$(date)
echo "############################## ##############################" >> $log_fp
echo "Present run started at: $run_start_dt" >> $log_fp
echo "############################## ##############################" >> $log_fp
echo >> $log_fp

echo "########## ##########" >> $log_fp
echo "RUN DETAILS" >> $log_fp
echo "########## ##########" >> $log_fp

echo "Machine: $machine_name" >> $log_fp
echo "Gpu Num: $gpu_nums" >> $log_fp
echo "Dataset: $dataset_name" >> $log_fp
echo "Subjects: ${subjects[@]}" >> $log_fp
echo "Sequences: ${seq_nums[@]}" >> $log_fp
echo "vid_names: ${vid_names[@]}" >> $log_fp
echo >> $log_fp

cd $code_base_dir
echo "Execution moved to $code_base_dir directory"  >> $log_fp

echo "Finding conda environments"  >> $log_fp
eval "$(conda shell.bash hook)"

conda activate 4D_humans_env
echo "Activated Conda Environment"  >> $log_fp

echo >> $log_fp
echo "Using GPU: $cuda_visible_devices"  >> $log_fp
export CUDA_VISIBLE_DEVICES=$cuda_visible_devices
echo >> $log_fp
########## LOGGER BLOCK ##########

for subject in ${subjects[@]}; do
    for seq_num in ${seq_nums[@]}; do
        data_dir=$data_base_dir/$subject/$seq_num/imageSequence
        for vid_name in ${vid_names[@]}; do
            vid_fp=$data_base_dir/$subject/$seq_num/imageSequence/$vid_name

            ########## LOGGER BLOCK ##########
            echo "########## ##########" >> $log_fp
            echo Processing: $vid_fp >> $log_fp
            echo "########## ##########" >> $log_fp
            echo >> $log_fp
            vid_fbn=${vid_fp/.avi//}
            echo vid_fbn: $vid_fbn >> $log_fp
            vid_bn=(${vid_fbn//\// })
            vid_bn=${vid_bn[-1]}
            echo vid_bn: $vid_bn >> $log_fp
            echo >> $log_fp
            ########## LOGGER BLOCK ##########

            ########## FFMPEG BLOCK ##########
            mkdir $vid_fbn
            echo "Created dir $vid_fbn">> $log_fp
            conda deactivate
            echo "Deactivated conda env for ffmpeg">> $log_fp
            ffmpeg_start_dt=$(date)
            echo "Running ffmpeg, start time: $ffmpeg_start_dt" >> $log_fp
            ffmpeg -i $vid_fp -r 30000/1001 -f image2 -v error $vid_fbn/%06d.jpg
            ffmpeg_end_dt=$(date)
            echo "Finished ffmpeg, end time: $ffmpeg_end_dt" >> $log_fp
            num_frames=$(ls $vid_fbn | wc -l)
            echo "Processed $num_frames frames" >> $log_fp
            conda activate 4D_humans_env
            echo "Reactivated 4DHumans env">> $log_fp
            echo >> $log_fp
            ########## FFMPEG BLOCK ##########
            
            ########## INFERENCE BLOCK ##########
            inference_start_dt=$(date)
            echo "Running inference, start time: $inference_start_dt" >> $log_fp
            python demo_w_pkl.py --img_folder $vid_fbn --out_folder $output_base_dir/$subject/$seq_num/$vid_bn --batch_size=48 --save_pkl
            inference_end_dt=$(date)
            echo "Finished inference, end time: $inference_end_dt" >> $log_fp
            echo Processed $(ls $vid_fbn | wc -l) frames >> $log_fp
            echo >> $log_fp
            echo "Removing $vid_fbn" >> $log_fp
            rm -r $vid_fbn
            ########## INFERENCE BLOCK ##########

            ########## LOGGER BLOCK ##########
            echo >> $log_fp
            echo "########## ##########" >> $log_fp
            echo "RUN STATISTICS" >> $log_fp
            echo "########## ##########" >> $log_fp
            echo "ffmpeg start time: $ffmpeg_start_dt" >> $log_fp
            echo "ffmpeg end time: $ffmpeg_end_dt" >> $log_fp
            echo "inference start time: $inference_start_dt" >> $log_fp
            echo "inference end time: $inference_end_dt" >> $log_fp
            echo Processed $num_frames frames >> $log_fp
            echo >> $log_fp
            echo >> $log_fp
            ########## LOGGER BLOCK ##########
        done
    done
done


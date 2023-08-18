MY_README

# Demo Run
## On Image Directory
`conda activate 4D_humans_env`
`python demo.py --img_folder example_data/images --out_folder outputs/example_data/images --batch_size=48 --side_view --save_mesh --full_frame`
`python demo_w_pkl.py --img_folder example_data/images --out_folder outputs2/example_data/images --batch_size=48 --save_pkl`

## Demo on videos
- Split images to frames using ffmpeg: `ffmpeg -i $vid_file -r 30000/1001 -f image2 -v error $img_folder/%06d.jpg` (Ensure that $img_folder exists before running the above command)
- Then run the inference code on the image directory using the command above.
- To combine the output frames into a video: 
```ffmpeg -framerate 30000/1001 -y -threads 16 -i $img_folder/%06d.jpg -profile:v baseline -level 3.0 -c:v libx264 -pix_fmt yuv420p -an -v error $output_vid_file
```

# New Files:
- `demo_yolo.py`: WIP file using yolo as MPT
- `demo_w_pkl.py`: Spin-off of `demo.py`. Has functionality for dumping pkl. And does NOT save a few other things to optimize storage

# Setup on other machines
- **ignored directories**: `data`, `outputs`, `outputs2`


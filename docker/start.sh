docker run \
            --gpus all -it \
            --rm \
            --ipc=host \
            -v /home/karmanov_aa/mmsegmentation/docker/add_req.sh:/workspace/add_req.sh \
            -v /home/adeshkin/projects/p_seg_dyn_map/kitti_360_track_0010_image_00:/workspace/Kitti \
            -v /home/adeshkin/projects/tools/Mapillary:/workspace/Mapillary \
            -v /home/karmanov_aa/mmsegmentation:/workspace/mmsegmentation \
            -v /home/karmanov_aa/best_models:/workspace/best_models \
            mmseg

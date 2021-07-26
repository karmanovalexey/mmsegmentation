docker run \
            --gpus all -it \
            --rm \
            --ipc=host \
            -v /home/karmanov_aa/mmsegmentation/docker/add_req.sh:/workspace/add_req.sh \
            -v /home/adeshkin/projects/tools/Mapillary:/workspace/Mapillary \
            -v /home/karmanov_aa/mmsegmentation:/workspace/mmsegmentation \
            -v /home/karmanov_aa/best_models:/workspace/best_models \
            mmseg

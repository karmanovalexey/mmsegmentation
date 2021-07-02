docker run \
            --gpus all -it \
            --rm \
            --ipc=host \
            -v /home/raaicv/mmsegmentation/docker/add_req.sh:/workspace/add_req.sh \
            -v /data/Mapillary:/workspace/Mapillary \
            -v /home/raaicv/mmsegmentation:/workspace/mmsegmentation \
            mmseg

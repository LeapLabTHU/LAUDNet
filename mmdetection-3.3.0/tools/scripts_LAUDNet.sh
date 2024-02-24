# DDQ-DETR
bash tools/dist_train.sh configs/ddq/ddq-detr-4scale_r101_channel_2222_0x5_8xb2-12e_coco.py 8 --work-dir work_dirs/ddq-detr-4scale_r101_channel_2222_0x5_8xb2-12e_coco

bash tools/dist_train.sh configs/ddq/ddq-detr-4scale_r101_layer_0x5_8xb2-12e_coco.py 8 --work-dir work_dirs/ddq-detr-4scale_r101_layer_0x5_8xb2-12e_coco



## Mask2Former

bash tools/dist_train.sh configs/mask2former/mask2former_r101_channel_2222_0x5_8xb2-lsj-50e_coco.py 8 --work-dir work_dirs/mask2former_r101_channel_2222_0x5_8xb2-lsj-50e_coco

bash tools/dist_train.sh configs/mask2former/mask2former_r101_layer_0x5_8xb2-lsj-50e_coco.py 8 --work-dir work_dirs/mask2former_r101_layer_0x5_8xb2-lsj-50e_coco
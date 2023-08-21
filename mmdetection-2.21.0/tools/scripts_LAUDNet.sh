# RetinaNet
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=11005 tools/train.py \
configs/retinanet/scale_backbone_lr/retinanet_ladmmdet_r101_fpn_1x_coco_r101_channel_2222_0x6_lrmult0x2.py \
--work-dir work_dirs/retinanet_lad_101_fpn_1x_coco_r101_channel_2222_0x6_lrmult0x2 \
--launcher pytorch;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=11005 tools/train.py \
configs/retinanet/scale_backbone_lr/retinanet_ladmmdet_r101_fpn_1x_coco_r101_layer_0x5_lrmult0x2.py \
--work-dir work_dirs/retinanet_ladmmdet_r101_fpn_1x_coco_r101_layer_0x5_lrmult0x2 \
--launcher pytorch;


# FasterRCNN
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=11005 tools/train.py \
configs/faster_rcnn/scale_backbone_lr/faster_rcnn_ladmmdet_r101_fpn_1x_coco_r101_channel_2222_0x8_lrmult0x5.py \
--work-dir work_dirs/faster_rcnn_ladmmdet_r101_fpn_1x_coco_r101_channel_2222_0x8_lrmult0x5 \
--launcher pytorch;


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=11005 tools/train.py \
configs/faster_rcnn/scale_backbone_lr/faster_rcnn_ladmmdet_r101_fpn_1x_coco_r101_layer_0x5_lrmult0x5.py \
--work-dir work_dirs/faster_rcnn_ladmmdet_r101_fpn_1x_coco_r101_layer_0x5_lrmult0x5 \
--launcher pytorch;


# MaskRCNN
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=11005 tools/train.py \
configs/mask_rcnn/scale_backbone_lr/mask_rcnn_ladmmdet_r101_fpn_1x_coco_r101_channel_2222_0x8_lrmult0x3.py \
--work-dir work_dirs/mask_rcnn_ladmmdet_r101_fpn_1x_coco_r101_channel_2222_0x8_lrmult0x3 \
--launcher pytorch;


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=11005 tools/train.py \
configs/mask_rcnn/scale_backbone_lr/mask_rcnn_ladmmdet_r101_fpn_1x_coco_r101_layer_0x5_lrmult0x3.py \
--work-dir work_dirs/mask_rcnn_ladmmdet_r101_fpn_1x_coco_r101_layer_0x5_lrmult0x3 \
--launcher pytorch;

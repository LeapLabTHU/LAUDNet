_base_ = '../retinanet_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        type='LAD_MMDet_ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        temperature_0=0.1,
        temperature_t=0.01,
        channel_dyn_granularity=[2,2,2,2],
        spatial_mask_channel_group=[1,1,1,1],
        mask_spatial_granularity=[1,1,1,1],
        dyn_mode=['channel','channel','channel','channel'],
        channel_masker=['MLP','MLP','MLP','MLP'],
        channel_masker_layers=[2,2,2,2],
        reduction_ratio=[16,16,16,16],
        init_cfg=dict(type='Pretrained', checkpoint='checkpoints/imagenet_pretrain/r101_channel_2222_0x6.pth.tar'),
        sparsity_target=0.6,
    )
)

optimizer = dict(
    type='SGD',
    lr=0.02,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.2, decay_mult=0.9)}
    ))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

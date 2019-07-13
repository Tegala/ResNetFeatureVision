from yacs.config import CfgNode as CN

cfg = CN()

cfg.MODEL = CN()

cfg.MODEL.INPUT_SIZE = (1000, 1000)

cfg.MODEL.RESNETS = CN()
cfg.MODEL.RESNETS.RETURN_C1_FEATURES = True
cfg.MODEL.RESNETS.STEM_FUNC = "StemWithFixedBatchNorm"
cfg.MODEL.RESNETS.TRANS_FUNC = "BottleneckWithFixedBatchNorm"
cfg.MODEL.RESNETS.NUM_GROUPS = 1
cfg.MODEL.RESNETS.WIDTH_PER_GROUP = 64
cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 64
cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
cfg.MODEL.RESNETS.STRIDE_IN_1X1 = True

cfg.MODEL.BACKBONE = CN()
cfg.MODEL.BACKBONE.CONV_BODY = "R-50-FPN" # "R-50-C4" "R-50-C5"
cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 2

cfg.OUTPUTS = CN()
cfg.OUTPUTS.PRETRAINED = 'checkpoints/r50body.pth'
cfg.OUTPUTS.RESULTS = 'images/resnet50_lung_features'



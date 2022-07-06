# run innoviz testing evaluation with dummy model, generating detection aroun GT data with noise.
python test.py --cfg_file cfgs/innoviz_models/dummy.yaml 

# run innoviz testing evaluation with with kitty pv_rcnn model as example. 
# results are expected to be mostly arbitrary since are based on kitty trained and configued model.
# python test.py --cfg_file cfgs/innoviz_models/pv_rcnn.yaml --ckpt pv_rcnn_8369.pth

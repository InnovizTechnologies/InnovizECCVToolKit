# run demo with dummy model, generating detection aroun GT data with noise.
python innoviz_demo.py --cfg_file cfgs/innoviz_models/dummy.yaml --data_path ../data/innoviz/testing

# run demo with kitty pv_rcnn model as example. 
# results are expected to be mostly arbitrary since are based on kitty trained and configued model.
# python innoviz_demo.py --cfg_file cfgs/innoviz_models/pv_rcnn.yaml --ckpt pv_rcnn_8369.pth --data_path ../data/innoviz/testing

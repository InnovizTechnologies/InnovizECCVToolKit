# run demo with dummy model.
# Dummy model - generating dummy detections by adding noise to GT data.
python innoviz_demo.py --cfg_file cfgs/innoviz_models/dummy.yaml --data_path ../data/innoviz/training

# demo for running an arbitrary model. results are expected to be arbitrary.
# python innoviz_demo.py --cfg_file cfgs/innoviz_models/pv_rcnn.yaml --ckpt pv_rcnn_8369.pth --data_path ../data/innoviz/testing

# OpenPCDet demo for running an arbitrary model. results are expected to be arbitrary.
# python demo.py --cfg_file cfgs/innoviz_models/pv_rcnn.yaml --ckpt pv_rcnn_8369.pth --data_path ../data/innoviz/testing/itwo
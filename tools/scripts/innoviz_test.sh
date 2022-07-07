# run innoviz testing evaluation with dummy model.
# Dummy model - generating dummy detections by adding noise to GT data.
python test.py --cfg_file cfgs/innoviz_models/dummy.yaml 

# demo for running an arbitrary model. results are expected to be arbitrary.
# python test.py --cfg_file cfgs/innoviz_models/pv_rcnn.yaml --ckpt pv_rcnn_8369.pth

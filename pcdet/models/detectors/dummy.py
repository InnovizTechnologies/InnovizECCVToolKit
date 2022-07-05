import os
import torch
import torch.nn as nn
import numpy as np
from ...utils.spconv_utils import find_all_spconv_keys
from ...ops.iou3d_nms import iou3d_nms_utils

class Dummy(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        return []

    def forward(self, batch_dict, **kwargs):
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts


    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []

        if 'gt_boxes' not in batch_dict:
            return pred_dicts, recall_dict

        gt_boxes = batch_dict['gt_boxes'].cpu().clone()
        for index in range(batch_size):

            pred_boxes = gt_boxes[index][:,:7]
            pred_labels = gt_boxes[index][:,7].cpu().numpy().astype(np.int64)
            pred_scores = np.ones_like(pred_labels, dtype=np.float32)

            # add noise to boxes translation
            pred_boxes[:,0:3] += np.random.uniform(0, post_process_cfg.TRANSLATION_NOISE_MAX, pred_boxes[:,0:3].size())

            # # add noise to boxes size
            pred_boxes[:,3:6] += np.random.uniform(0, post_process_cfg.SIZE_NOISE_MAX, pred_boxes[:,3:6].size())

            # # add noise to boxes rotation
            pred_boxes[:,6] += np.random.uniform(0, post_process_cfg.HEADING_NOISE_MAX, pred_boxes[:,6].size())

            pred_boxes = pred_boxes.to(batch_dict['gt_boxes'].device)

            record_dict = {
                'pred_boxes': pred_boxes,
                'pred_scores': torch.from_numpy(pred_scores),
                'pred_labels': torch.from_numpy(pred_labels),
            }
            pred_dicts.append(record_dict)

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )            

        return pred_dicts, recall_dict

    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k >= 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict

    def load_params_from_file(self, filename, logger, to_cpu=False):
        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        logger.info('==> Done')

    def get_training_loss(self):
        raise NotImplemented("The task is to implement training in an non dummy model.")

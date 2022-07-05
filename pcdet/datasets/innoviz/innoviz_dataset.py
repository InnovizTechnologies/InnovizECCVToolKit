
import glob

import numpy as np

from pcdet.datasets import DatasetTemplate

class InnovizDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, split=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        # TODO get folder names from configurations
        self._lidar_dir = 'itwo'
        self._lidar_ext = '.bin'

        self._gt_dir = 'gt_boxes'
        self._gt_boxes_ext = '.bin'

        self.split = split if split is not None else self.dataset_cfg.DATA_SPLIT[self.mode]
        self.lidar_path = self.root_path / self.split / self._lidar_dir
        self.gt_path = self.root_path / self.split / self._gt_dir
        
        data_file_list = glob.glob(str(self.lidar_path / f'*{self._lidar_ext}'))
        self.ids_list = [f.split('/')[-1][:-len(self._lidar_ext)] for f in data_file_list]
        self.ids_list.sort()

    def __len__(self):
        return len(self.ids_list)

    def __getitem__(self, index):
        file_name = self.ids_list[index]
        points = np.fromfile(self.lidar_path / f'{file_name}{self._lidar_ext}', dtype=np.float32).reshape(-1, 4)

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        gt_path = self.gt_path / f'{file_name}{self._gt_boxes_ext}' 
        if gt_path.exists():
            gt_boxes = np.fromfile(gt_path, dtype=np.float64).reshape(-1, 7)
            input_dict['gt_boxes'] = gt_boxes
            input_dict['gt_names'] = np.array(['Car'] * gt_boxes.shape[0])


        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 
                'boxes': np.zeros([num_samples, 7]),
                'score': np.zeros(num_samples), 
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['boxes'] = pred_boxes
            pred_dict['score'] = pred_scores

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)
            
        return annos       

    def evaluation(self, det_annos, class_names, **kwargs):        
        return 'evaluation is TBC', {}


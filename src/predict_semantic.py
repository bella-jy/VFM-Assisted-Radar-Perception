"""Export 3D detections for calibrated frames, without dataset-specific evaluation."""
import _init_paths
import json
from pathlib import Path
import numpy as np
import torch

from opts import opts
from dataset.dataset_factory import dataset_factory
from detector import Detector
from test import PrefetchDataset
from utils.ddd_utils import compute_box_3d, project_to_image


def main():
    option_parser = opts()
    opt = option_parser.parse()
    if not opt.semantic_fusion or not opt.load_model:
        raise ValueError('Use --semantic_fusion and a trained --load_model checkpoint.')
    dataset_class = dataset_factory[opt.test_dataset]
    opt = option_parser.update_dataset_info_and_set_heads(opt, dataset_class)
    dataset = dataset_class(opt, opt.val_split)
    detector = Detector(opt)
    loader = torch.utils.data.DataLoader(PrefetchDataset(opt, dataset, detector.pre_process),
                                         batch_size=1, shuffle=False, num_workers=0)
    frames = []
    for index, (image_id, inputs) in enumerate(loader):
        if opt.num_iters >= 0 and index >= opt.num_iters:
            break
        result = detector.run(inputs)['results']
        info = dataset.coco.loadImgs(ids=[int(image_id[0])])[0]
        calib = np.asarray(info['calib'], dtype=np.float32)
        for detection in result:
            corners = compute_box_3d(detection['dim'], detection['loc'], detection['rot_y'])
            detection['corners_3d_camera'] = corners
            detection['corners_2d'] = project_to_image(corners, calib) if np.all(corners[:, 2] > 0) else None
        frames.append({'image_id': int(image_id[0]), 'file_name': info['file_name'], 'detections': result})
    destination = Path(opt.save_dir) / 'semantic_detections_3d.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {'coordinate_system': 'camera: x right, y down, z forward; metres',
               'dimension_order': 'height,width,length', 'location': '3D box bottom center',
               'orientation': 'rot_y: radians around camera y',
               'checkpoint': str(Path(opt.load_model).resolve()), 'frames': frames}
    with destination.open('w', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False,
                  default=lambda value: value.tolist() if isinstance(value, (np.ndarray, np.generic)) else value)
    print('Saved 3D predictions to {}'.format(destination))


if __name__ == '__main__':
    main()

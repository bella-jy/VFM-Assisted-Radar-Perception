"""Synthetic contract tests; no trained accuracy claims and no weight downloads."""
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp' / 'matplotlib'))
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'src' / 'lib'),
               str(ROOT / 'segment-anything-main' / 'segment-anything-main')]
import cv2
import numpy as np
import pytest
import torch

torch.set_num_threads(2)
from utils.semantic_prior import radar_reliability, load_prior, warp_prior, feature_calibration
from utils.semantic_frustum import fill_frustum
from segment_anything.low_visibility import structural_losses, aggregate_masks
from segment_anything.modeling.image_encoder import ImageEncoderViT


def test_adapter_identity_and_gradient():
    torch.manual_seed(7)
    model = ImageEncoderViT(img_size=64, embed_dim=32, depth=12, num_heads=4,
                            out_chans=16, window_size=2).eval()
    image, reference = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)
    baseline = model(image).detach()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.enable_degradation_adaptation()
    output = model(image, reference=reference)
    torch.testing.assert_close(output, baseline, atol=0, rtol=0)
    output.square().mean().backward()
    adapter = model.degradation_adapter
    assert adapter.fusion.weight.grad.abs().sum() > 0
    assert adapter.global_maps[0].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='reference'):
        model(image)


def test_losses_reject_bad_reference_and_backpropagate():
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    target = torch.zeros_like(logits)
    target[:, :, 3:12, 5:10] = 1
    total, parts = structural_losses(logits, target, 1 - target, torch.zeros(2))
    assert parts['diff'] == 0 and parts['boundary'] == 0
    total.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    total, parts = structural_losses(logits, target, target, torch.ones(2))
    assert all(torch.isfinite(value) for value in parts.values())


def test_reliability_rejects_isolated_and_outside_and_empty():
    prior = np.zeros((64, 64), np.float32)
    prior[20:44, 20:44] = 1
    xyz = np.array([[0, 0, 10], [0.2, 0, 10], [0, 0, 30], [0, 0, -1]], np.float32)
    pixels = np.array([[32, 32], [33, 32], [32, 32], [32, 32]], np.float32)
    scores = radar_reliability(xyz, np.zeros(4), pixels, prior)
    assert np.all(scores[:2, 0] > 0.3)
    assert np.all(scores[2:, 0] == 0)
    np.testing.assert_allclose(scores[:, 0], scores[:, 1:].prod(1))
    assert radar_reliability(xyz, np.zeros(4), pixels, np.zeros_like(prior)).sum() == 0
    assert radar_reliability(np.empty((0, 3)), np.empty(0), np.empty((0, 2)), prior).shape == (0, 4)
    scores = radar_reliability(xyz[:2], np.array([0, 10]), pixels[:2], prior)
    assert scores[:, 0].sum() == 0


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_frustum_preserves_attributes_without_mutating_depth(backend):
    opt = SimpleNamespace(pc_feat_channels={'pc_dep': 0}, normalize_depth=True,
                          max_pc_dist=50, hm_to_box_ratio=0.3)
    radar = np.zeros((8, 16, 16), np.float32)
    attributes = np.array([10, 2, 3, 7, 0.4, 0.8, 0.9, 0.6], np.float32)
    radar[:, 8, 8] = attributes
    output = np.zeros_like(radar)
    if backend == 'torch':
        radar, output = torch.from_numpy(radar), torch.from_numpy(output)
    fill_frustum(output, radar, 10, [4, 4, 12, 12], 2, opt)
    actual = output.numpy() if backend == 'torch' else output
    expected = attributes.copy()
    expected[0] /= 50
    np.testing.assert_allclose(actual[:, 8, 8], expected)
    assert radar[0, 8, 8] == 10
    fill_frustum(output, radar, 10, [-100, -100, -10, -10], 2, opt)


def test_prior_geometry_and_missing_file(tmp_path):
    mask = np.zeros((32, 64), np.uint8)
    mask[4:12, 8:16] = 1
    np.savez(tmp_path / 'frame.jpg.npz', mask=mask)
    prior = load_prior(tmp_path, {'file_name': 'frame.jpg'}, (32, 64, 3))
    warped = warp_prior(prior, np.array([[1, 0, 0], [0, 1, 0]], np.float32), 64, 32, True)
    np.testing.assert_array_equal(warped[0], mask[:, ::-1])
    with pytest.raises(FileNotFoundError):
        load_prior(tmp_path, {'file_name': 'missing.jpg'}, (32, 64))
    with pytest.raises(ValueError, match='coordinates'):
        load_prior(tmp_path, {'file_name': 'frame.jpg'}, (16, 32))


def test_mask_aggregation_excludes_large_background():
    foreground = np.zeros((32, 32), bool)
    foreground[8:24, 8:24] = True
    records = [dict(segmentation=foreground, predicted_iou=0.95),
               dict(segmentation=np.ones((32, 32), bool), predicted_iou=0.99)]
    mask, response, accepted = aggregate_masks(records, (32, 32), min_area=4)
    assert len(accepted) == 1 and mask[16, 16] and not mask[0, 0]
    assert 0 <= response.min() <= response.max() <= 1


def test_automatic_sam_reference_crop_and_decoder_gradient():
    from segment_anything.modeling import Sam, PromptEncoder, MaskDecoder, TwoWayTransformer
    from segment_anything import SamAutomaticMaskGenerator
    from tools.low_visibility_sam import decode_prompt
    encoder = ImageEncoderViT(img_size=64, embed_dim=32, depth=12, num_heads=4,
                              out_chans=16, window_size=2)
    sam = Sam(encoder, PromptEncoder(embed_dim=16, image_embedding_size=(4, 4),
              input_image_size=(64, 64), mask_in_chans=16),
              MaskDecoder(transformer_dim=16, transformer=TwoWayTransformer(
                  depth=1, embedding_dim=16, num_heads=4, mlp_dim=32)),
              pixel_mean=[0, 0, 0], pixel_std=[255, 255, 255]).eval()
    for parameter in sam.parameters():
        parameter.requires_grad_(False)
    encoder.enable_degradation_adaptation()
    image = torch.rand(1, 3, 64, 64)
    embedding = encoder(image, reference=image * 0.8)
    logits, _ = decode_prompt(sam, embedding, np.array([20, 20], np.float32), (64, 64), (64, 64))
    logits.square().mean().backward()
    assert encoder.degradation_adapter.fusion.weight.grad.abs().sum() > 0
    rgb = np.full((40, 64, 3), 80, np.uint8)
    generator = SamAutomaticMaskGenerator(sam, points_per_side=2, points_per_batch=2,
                                          pred_iou_thresh=0, stability_score_thresh=0, crop_n_layers=1)
    records = generator.generate(rgb, reference_image=rgb + 20)
    assert isinstance(records, list)
    assert all(record['segmentation'].shape == (40, 64) for record in records)
    with pytest.raises(ValueError, match='dimensions'):
        generator.generate(rgb, reference_image=rgb[:20])


def test_feature_calibration_preserves_projection_after_flip_and_affine():
    camera = np.array([[80, 0, 40, 0], [0, 90, 30, 0], [0, 0, 1, 0]], np.float32)
    point = np.array([1, 0.5, 10, 1], np.float32)
    transform = np.array([[0.25, 0, 2], [0, 0.25, -1]], np.float32)
    original = camera @ point
    original /= original[2]
    expected = transform @ np.array([99 - original[0], original[1], 1])
    point[0] *= -1
    projected = feature_calibration(camera, transform, width=100, flipped=True) @ point
    np.testing.assert_allclose(projected[:2] / projected[2], expected)


def make_dataset(tmp_path, consistency=False):
    from opts import opts
    from dataset.datasets.custom_dataset import CustomDataset
    parser = opts()
    flags = ['ddd', '--dataset', 'custom', '--num_classes', '1', '--input_h', '64', '--input_w', '64',
             '--pointcloud', '--semantic_fusion', '--semantic_prior_dir', str(tmp_path),
             '--custom_dataset_img_path', str(tmp_path), '--custom_dataset_ann_path', str(tmp_path / '{split}.json'),
             '--gpus', '-1', '--dla_node', 'conv', '--no_pretrained_backbone', '--K', '8',
             '--velocity', '--not_rand_crop', '--shift', '0', '--scale', '0', '--flip', '0',
             '--aug_rot', '0', '--no_color_aug', '--head_conv', '16']
    if consistency:
        flags += ['--consistency_weight', '1', '--semantic_score_weight', '1']
    opt = parser.update_dataset_info_and_set_heads(parser.parse(flags), CustomDataset)
    image = np.zeros((64, 64, 3), np.uint8)
    cv2.imencode('.jpg', image)[1].tofile(str(tmp_path / 'frame.jpg'))
    mask = np.zeros((64, 64), np.uint8)
    mask[20:44, 20:44] = 1
    np.savez(tmp_path / 'frame.jpg.npz', mask=mask)
    radar = np.zeros((18, 3), np.float32)
    radar[0] = [0, 0.1, 0]
    radar[1] = [0.5, 0.5, 0.5]
    radar[2] = [10, 10, 35]
    radar[5] = 5
    calib = [[50, 0, 32, 0], [0, 50, 32, 0], [0, 0, 1, 0]]
    payload = dict(images=[dict(id=1, file_name='frame.jpg', width=64, height=64,
        calib=calib, camera_intrinsic=np.asarray(calib)[:, :3].tolist(), radar_pc=radar.tolist())],
        annotations=[dict(id=1, image_id=1, category_id=1, bbox=[20, 20, 24, 24], area=576,
            iscrowd=0, depth=10, dim=[1.5, 2, 4], alpha=0, amodel_center=[32, 32],
            velocity_cam=[0, 0, 0], radar_consistency=1)], categories=[dict(id=1, name='object')])
    for split in ('train', 'val'):
        (tmp_path / (split + '.json')).write_text(json.dumps(payload), encoding='utf-8')
    return opt, CustomDataset(opt, 'val')


def test_dataset_model_loss_decode_and_cpu_detector(tmp_path):
    from model.model import create_model, save_model
    from trainer import GenericLoss, ModelWithLoss
    from detector import Detector
    from test import PrefetchDataset
    opt, dataset = make_dataset(tmp_path, consistency=True)
    item = dataset[0]
    assert item['pc_dep'].shape == (8, 16, 16)
    assert item['pc_3d'].shape[0] == 22 and item['pc_N'] == 2
    assert item['pc_dep'][4].max() > 0
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt)
    combined = ModelWithLoss(model, GenericLoss(opt), opt)
    output, loss, _ = combined(batch, 'train')
    assert torch.isfinite(loss)
    loss.backward()
    assert model.semantic_fusion.gate.weight.grad is not None
    assert model.consistency[-1].weight.grad.abs().sum() > 0
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    previous = model.semantic_fusion.gate.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(previous, model.semantic_fusion.gate.weight)
    # Deterministic valid boxes make the postprocess assertions non-vacuous.
    with torch.no_grad():
        for name, bias in [('hm', 2), ('dim', 1.5), ('wh', 8), ('consistency', 1)]:
            getattr(model, name)[-1].weight.zero_()
            getattr(model, name)[-1].bias.fill_(bias)
    # Synthetic weights saved only in pytest's temporary directory.
    checkpoint = tmp_path / 'synthetic_untrained.pth'
    save_model(str(checkpoint), 0, model)
    opt.load_model = str(checkpoint)
    detector = Detector(opt)
    inputs = next(iter(torch.utils.data.DataLoader(PrefetchDataset(opt, dataset, detector.pre_process), batch_size=1)))
    result = detector.run(inputs[1])
    assert 'results' in result
    assert len(result['results']) > 0
    for detection in result['results']:
        assert all(key in detection for key in ('loc', 'dim', 'rot_y', 'consistency'))
        assert np.isfinite(detection['loc']).all()
    # The command-line exporter writes an actual schema, using synthetic weights only.
    import predict_semantic
    from unittest.mock import patch
    opt.save_dir = str(tmp_path / 'synthetic_output')
    with patch.object(predict_semantic.opts, 'parse', return_value=opt):
        # parse/update normally turns head_conv from an int into a dictionary.
        opt.head_conv = 16
        predict_semantic.main()
    payload = json.loads((Path(opt.save_dir) / 'semantic_detections_3d.json').read_text(encoding='utf-8'))
    assert payload['frames'][0]['detections'][0]['corners_3d_camera']


def test_inference_refuses_missing_semantic_weights(tmp_path):
    from model.model import create_model, load_model
    opt, _ = make_dataset(tmp_path)
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt)
    state = {key: value for key, value in model.state_dict().items() if not key.startswith('semantic_fusion.')}
    path = tmp_path / 'missing_semantic.pth'
    torch.save({'epoch': 0, 'state_dict': state}, path)
    opt.semantic_require_checkpoint = True
    with pytest.raises(ValueError, match='trained checkpoint'):
        load_model(model, str(path), opt)


def test_baseline_switch_and_missing_prior(tmp_path):
    from model.model import create_model
    opt, _ = make_dataset(tmp_path)
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt)
    with pytest.raises(ValueError, match='semantic_prior'):
        model(torch.randn(1, 3, 64, 64), pc_hm=torch.zeros(1, 8, 16, 16))
    opt.semantic_fusion = False
    opt.pc_feat_lvl = ['pc_dep', 'pc_vx', 'pc_vz']
    baseline = create_model(opt.arch, opt.heads, opt.head_conv, opt)
    output = baseline(torch.randn(1, 3, 64, 64), pc_hm=torch.zeros(1, 3, 16, 16))
    assert output[0]['dep_sec'].shape == (1, 1, 16, 16)
    assert not any(key.startswith('semantic_fusion') for key in baseline.state_dict())


def test_one_epoch_main_training_without_git_metadata(tmp_path):
    import main as training
    opt, _ = make_dataset(tmp_path, consistency=True)
    opt.head_conv = 16
    opt.num_epochs = opt.val_intervals = opt.batch_size = 1
    opt.chunk_sizes = [1]
    opt.num_workers = 0
    opt.save_dir = str(tmp_path / 'synthetic_training')
    opt.debug_dir = str(Path(opt.save_dir) / 'debug')
    training.main(opt)
    checkpoint = torch.load(Path(opt.save_dir) / 'model_last.pth', map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == 1 and 'optimizer' in checkpoint
    assert 'semantic_fusion.gate.weight' in checkpoint['state_dict']
    logs = list(Path(opt.save_dir).glob('logs_*/log.txt'))
    assert 'consistency' in logs[0].read_text(encoding='utf-8')

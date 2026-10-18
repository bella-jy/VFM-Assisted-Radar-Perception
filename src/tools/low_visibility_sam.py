"""SAM adaptation, reference preparation and cached prior generation.

Run from repository root: python src/tools/low_visibility_sam.py --help
"""
import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'segment-anything-main' / 'segment-anything-main'))
import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.low_visibility import build_adapted_sam, aggregate_masks, structural_losses
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.utils.amg import calculate_stability_score


def read_rgb(path):
    # imdecode supports Windows paths containing Chinese characters.
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot read image: {}'.format(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError('Could not encode {}'.format(path))
    encoded.tofile(str(path))


def image_paths(root):
    paths = sorted(p for p in Path(root).rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'))
    if not paths:
        raise ValueError('No images found under {}'.format(root))
    return paths


def tensor_image(sam, rgb):
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    resized = transform.apply_image(rgb)
    tensor = torch.as_tensor(resized.copy(), device=sam.device).permute(2, 0, 1)[None].float()
    return sam.preprocess(tensor), resized.shape[:2]


def decode_prompt(sam, embedding, point, input_size, original_size, multimask=False):
    coordinates = ResizeLongestSide(sam.image_encoder.img_size).apply_coords(point[None], original_size)
    points = torch.as_tensor(coordinates, device=sam.device, dtype=torch.float32)[None]
    labels = torch.ones((1, 1), device=sam.device, dtype=torch.int64)
    sparse, dense = sam.prompt_encoder(points=(points, labels), boxes=None, masks=None)
    logits, quality = sam.mask_decoder(image_embeddings=embedding,
        image_pe=sam.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense, multimask_output=multimask)
    return sam.postprocess_masks(logits, input_size, original_size), quality


def train(args):
    manifest_path = Path(args.manifest).resolve()
    records = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(records, list) or not records:
        raise ValueError('Training manifest must be a nonempty JSON list.')
    for record in records:
        if record.get('split') != 'train' or not record.get('sequence_id'):
            raise ValueError('Training entries must declare split=train and a sequence_id; partition sequences before training.')
        for key in ('image', 'reference', 'mask'):
            record[key] = str(manifest_path.parent / record[key])
            if not Path(record[key]).is_file():
                raise FileNotFoundError(record[key])
    sam = build_adapted_sam(args.checkpoint).to(args.device)
    teacher = sam_model_registry['vit_b'](checkpoint=args.checkpoint).to(args.device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    adapter = sam.image_encoder.degradation_adapter
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        random.shuffle(records)
        sam.eval()  # The frozen SAM remains deterministic; adapters have no dropout/BN.
        adapter.train()
        totals = []
        accepted = 0
        optimizer.zero_grad()
        for index, record in enumerate(records):
            image, reference = read_rgb(record['image']), read_rgb(record['reference'])
            mask = cv2.imdecode(np.fromfile(record['mask'], np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask is None or reference.shape != image.shape or mask.shape != image.shape[:2]:
                raise ValueError('Image, reference and instance mask must be aligned at original resolution.')
            target = torch.as_tensor(mask > 0, device=args.device).float()[None, None]
            positive_y, positive_x = np.nonzero(mask)
            if not len(positive_y):
                raise ValueError('Each SAM training entry must contain one nonempty object mask.')
            selection = np.random.randint(len(positive_y))
            point = np.array([positive_x[selection], positive_y[selection]], dtype=np.float32)
            low, input_size = tensor_image(sam, image)
            clear, _ = tensor_image(sam, reference)
            with torch.no_grad():
                ref_logits, quality = decode_prompt(teacher, teacher.image_encoder(clear), point,
                                                   input_size, image.shape[:2], multimask=True)
                best = int(quality[0].argmax())
                ref_logits = ref_logits[:, best:best+1]
                stability = calculate_stability_score(ref_logits, teacher.mask_threshold, 1.0)
                valid = (quality[:, best] >= 0.88) & (stability[:, 0] >= 0.95)
                reference_mask = (ref_logits > teacher.mask_threshold).float()
                accepted += int(valid.sum())
            embedding = sam.image_encoder(low, reference=clear)
            logits, _ = decode_prompt(sam, embedding, point, input_size, image.shape[:2])
            loss, _ = structural_losses(logits, target, reference_mask, valid, args.loss_weights)
            group_start = index // args.accumulate * args.accumulate
            group_size = min(args.accumulate, len(records) - group_start)
            (loss / group_size).backward()
            totals.append(float(loss.detach()))
            if (index + 1) % args.accumulate == 0 or index + 1 == len(records):
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        checkpoint = dict(adapter=adapter.state_dict(), model_type='vit_b', epoch=epoch,
                          base_checkpoint=str(Path(args.checkpoint).resolve()),
                          loss_weights=args.loss_weights, optimizer=optimizer.state_dict())
        torch.save(checkpoint, destination / 'adapter_last.pth')
        print(json.dumps(dict(epoch=epoch, loss=float(np.mean(totals)), accepted_references=accepted,
                              samples=len(records))), flush=True)


def generate(args):
    sam = build_adapted_sam(args.checkpoint, args.adapter).to(args.device).eval()
    generator = SamAutomaticMaskGenerator(sam, points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch, pred_iou_thresh=0.88, stability_score_thresh=0.95)
    source, references, destination = Path(args.images), Path(args.references), Path(args.output)
    for path in image_paths(source):
        relative = path.relative_to(source)
        image = read_rgb(path)
        reference = read_rgb(references / relative)
        records = generator.generate(image, reference_image=reference)
        mask, response, accepted = aggregate_masks(records, image.shape[:2], args.min_area,
                                                   args.max_area_ratio, args.threshold)
        output = destination / (str(relative) + '.npz')
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(output), mask=mask, response=response, schema_version=np.int32(1))
        write_image(str(output) + '.png', mask * 255)
        print('{}: {} retained masks -> {}'.format(relative, len(accepted), output), flush=True)


def references(args):
    """External restoration wrapper, NOT a substitute for trained diffusion weights.

    TorchScript contract: Bx3x256x256 in [-1,1] -> same shape/range. The wrapper
    must perform its complete conditioned reverse diffusion internally.
    """
    model = torch.jit.load(args.restorer, map_location=args.device).eval()
    source, destination = Path(args.images), Path(args.output)
    with torch.no_grad():
        for path in image_paths(source):
            rgb = read_rgb(path)
            tensor = torch.from_numpy(cv2.resize(rgb, (256, 256))).permute(2, 0, 1)[None].float()
            restored = model(tensor.to(args.device) / 127.5 - 1)
            if not torch.is_tensor(restored) or restored.shape != tensor.shape or not torch.isfinite(restored).all():
                raise ValueError('Restorer must return a finite Bx3x256x256 tensor.')
            restored = ((restored[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
            restored = cv2.resize(restored, (rgb.shape[1], rgb.shape[0]))
            write_image(destination / path.relative_to(source), cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    training = sub.add_parser('train', help='fine-tune adapters using GT instances and pseudo-clear references')
    training.add_argument('--manifest', required=True)
    training.add_argument('--checkpoint', required=True)
    training.add_argument('--epochs', type=int, default=20)
    training.add_argument('--lr', type=float, default=2e-4)
    training.add_argument('--accumulate', type=int, default=6)
    training.add_argument('--seed', type=int, default=317)
    training.add_argument('--loss-weights', type=float, nargs=4, default=[1, 0.5, 0.1, 1])
    generation = sub.add_parser('generate', help='cache automatic semantic priors from trained SAM')
    generation.add_argument('--checkpoint', required=True)
    generation.add_argument('--adapter', required=True)
    generation.add_argument('--images', required=True)
    generation.add_argument('--references', required=True)
    generation.add_argument('--points-per-side', type=int, default=32)
    generation.add_argument('--points-per-batch', type=int, default=16)
    generation.add_argument('--min-area', type=int, default=64)
    generation.add_argument('--max-area-ratio', type=float, default=0.8)
    generation.add_argument('--threshold', type=float, default=0.5)
    restoration = sub.add_parser('references', help='run an externally supplied trained diffusion-restoration wrapper')
    restoration.add_argument('--restorer', required=True)
    restoration.add_argument('--images', required=True)
    for command in (training, generation, restoration):
        command.add_argument('--output', required=True)
        command.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.command == 'train' and (args.accumulate < 1 or args.epochs < 1 or args.lr <= 0):
        parser.error('accumulate, epochs and lr must be positive')
    {'train': train, 'generate': generate, 'references': references}[args.command](args)


if __name__ == '__main__':
    main()

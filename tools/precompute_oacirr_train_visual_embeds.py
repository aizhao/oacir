import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from lavis.models import load_model_and_preprocess
from data_utils import OACIRRDataset, targetpad_transform, squarepad_transform
from utils import collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Union",
                        choices=["Union", "Fashion", "Car", "Product", "Landmark"])
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val"])
    parser.add_argument("--data-root", type=str, default="./Datasets/OACIRR")
    parser.add_argument("--blip-model-name", type=str, default="oacir_adafocal")
    parser.add_argument("--vit-backbone", type=str, default="pretrain")
    parser.add_argument("--transform", type=str, default="targetpad",
                        choices=["targetpad", "squarepad"])
    parser.add_argument("--target-ratio", type=float, default=1.25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {args.blip_model_name}, backbone={args.vit_backbone}")
    blip_model, _, _ = load_model_and_preprocess(
        name=args.blip_model_name,
        model_type=args.vit_backbone,
        is_eval=True,
        device=device,
    )
    blip_model.eval()
    feature_model = blip_model
    while hasattr(feature_model, "module"):
        feature_model = feature_model.module


    input_dim = 224
    if args.transform == "targetpad":
        preprocess = targetpad_transform(args.target_ratio, input_dim)
        print(f"Using targetpad transform, target_ratio={args.target_ratio}")
    else:
        preprocess = squarepad_transform(input_dim)
        print("Using squarepad transform")

    dataset = OACIRRDataset(
        data_root=args.data_root,
        variant=args.dataset,
        split=args.split,
        mode="classic",
        preprocess=preprocess,
        bounding_box_width=0,
        bounding_box_crop=False,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
    )

    all_names = []
    all_embeds = []

    print(f"Extracting frozen visual_encoder outputs for OACIRR [{args.dataset}] {args.split}...")
    for names, images in tqdm(loader, ncols=120, ascii=True):
        images = images.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            raw_embeds = feature_model.visual_encoder(images)

        raw_embeds = raw_embeds.detach().cpu().to(torch.float16)

        all_names.extend(list(names))
        all_embeds.append(raw_embeds)

    all_embeds = torch.cat(all_embeds, dim=0)

    # 额外保存原图尺寸，用于后续 reference bbox 坐标变换，避免训练时重复打开图片
    name_to_size = {}
    for name in tqdm(all_names, desc="Saving image sizes", ncols=120, ascii=True):
        image_path = dataset.img_root / dataset.name_to_relpath[name]
        with Image.open(image_path) as img:
            name_to_size[name] = img.size  # (width, height)

    cache = {
        "names": all_names,
        "embeds": all_embeds,
        "name_to_size": name_to_size,
        "meta": {
            "dataset": args.dataset,
            "split": args.split,
            "blip_model_name": args.blip_model_name,
            "vit_backbone": args.vit_backbone,
            "transform": args.transform,
            "target_ratio": args.target_ratio,
            "dtype": "float16",
            "note": "embeds are raw visual_encoder outputs before ln_vision",
        }
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)

    print(f"Saved cache to: {output}")
    print(f"num images: {len(all_names)}")
    print(f"embed shape: {tuple(all_embeds.shape)}")


if __name__ == "__main__":
    main()
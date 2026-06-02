
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import math
import json
import time
import random
import argparse
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    CLIPModel,
    CLIPProcessor,
    CLIPTokenizerFast,
    get_cosine_schedule_with_warmup,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_image_name(img_name: str) -> str:
    img_name = img_name.strip()
    img_name = os.path.basename(img_name)
    img_name = re.sub(r"(\.(jpg|jpeg|png|bmp|gif))\.\d+$", r"\1", img_name, flags=re.IGNORECASE)
    return img_name


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_flickr8k_pairs(caption_file):
    with open(caption_file, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    if len(lines) == 0:
        raise ValueError("captions file is empty.")

    first_line = lines[0].lower()

    if "," in lines[0] and ("image" in first_line and "caption" in first_line):
        pairs = []
        reader = csv.DictReader(lines)
        for row in reader:
            img = normalize_image_name(row["image"])
            cap = row["caption"].strip()
            if img and cap:
                pairs.append((img, cap))
        return pairs

    if "\t" in lines[0]:
        pairs = []
        for line in lines:
            img_part, cap = line.split("\t", 1)
            img = normalize_image_name(img_part.split("#")[0].strip())
            cap = cap.strip()
            if img and cap:
                pairs.append((img, cap))
        return pairs

    try:
        reader = csv.reader(lines)
        header = next(reader)
        header_lower = [x.lower() for x in header]
        if len(header) >= 2 and "image" in header_lower[0] and "caption" in header_lower[1]:
            pairs = []
            for row in reader:
                if len(row) >= 2:
                    img = normalize_image_name(row[0])
                    cap = ",".join(row[1:]).strip()
                    if img and cap:
                        pairs.append((img, cap))
            return pairs
    except Exception:
        pass

    raise ValueError("Unsupported captions file format.")


def try_load_split_file(data_root, split_dir=None):
    roots = []
    if split_dir is not None:
        roots.append(split_dir)
    roots.extend([
        data_root,
        os.path.join(data_root, "Flickr8k_text"),
        os.path.join(data_root, "flickr8k_text"),
        os.path.join(data_root, "texts"),
    ])

    candidates = [
        ("Flickr_8k.trainImages.txt", "Flickr_8k.devImages.txt", "Flickr_8k.testImages.txt"),
        ("train.txt", "val.txt", "test.txt"),
        ("train.txt", "valid.txt", "test.txt"),
    ]

    for base in roots:
        for train_name, val_name, test_name in candidates:
            train_path = os.path.join(base, train_name)
            val_path = os.path.join(base, val_name)
            test_path = os.path.join(base, test_name)

            if os.path.exists(train_path) and os.path.exists(val_path):
                def read_list(path):
                    with open(path, "r", encoding="utf-8") as f:
                        return [normalize_image_name(x) for x in f if x.strip()]
                train_imgs = read_list(train_path)
                val_imgs = read_list(val_path)
                test_imgs = read_list(test_path) if os.path.exists(test_path) else []
                return train_imgs, val_imgs, test_imgs
    return None


def build_splits_and_pairs(args):
    pairs = load_flickr8k_pairs(args.caption_file)

    clean_pairs = []
    missing_pairs = []
    for img, cap in pairs:
        img = normalize_image_name(img)
        img_path = os.path.join(args.image_dir, img)
        if os.path.exists(img_path):
            clean_pairs.append((img, cap))
        else:
            missing_pairs.append((img, cap))

    print(f"[Data] total pairs before cleaning: {len(pairs)}")
    print(f"[Data] total pairs after cleaning : {len(clean_pairs)}")
    print(f"[Data] missing pairs              : {len(missing_pairs)}")
    if missing_pairs:
        print("[Data] missing examples:", missing_pairs[:10])

    pairs = clean_pairs

    image_to_captions = defaultdict(list)
    for img, cap in pairs:
        image_to_captions[img].append(cap)

    all_images = sorted(image_to_captions.keys())
    print(f"[Data] unique valid images: {len(all_images)}")

    split_res = None if args.no_official_split else try_load_split_file(args.data_root, args.split_dir)

    if split_res is not None:
        train_images, val_images, test_images = split_res
        train_images = [x for x in train_images if x in image_to_captions]
        val_images = [x for x in val_images if x in image_to_captions]
        print("[Data] using detected official split files")
    else:
        train_images, val_images = train_test_split(
            all_images, test_size=args.val_ratio, random_state=args.seed, shuffle=True
        )
        train_images = sorted(train_images)
        val_images = sorted(val_images)
        print("[Data] official split not found, using random image-level split")

    train_pairs = []
    for img in train_images:
        for cap in image_to_captions[img]:
            train_pairs.append((img, cap))

    val_image_to_idx = {img: i for i, img in enumerate(val_images)}
    val_captions = []
    caption_to_image_idx = []
    for img in val_images:
        for cap in image_to_captions[img]:
            val_captions.append(cap)
            caption_to_image_idx.append(val_image_to_idx[img])

    info = {
        "num_total_pairs": len(pairs),
        "num_train_images": len(train_images),
        "num_train_pairs": len(train_pairs),
        "num_val_images": len(val_images),
        "num_val_captions": len(val_captions),
        "num_missing_pairs": len(missing_pairs),
    }
    print("[Data]", info)
    return train_pairs, val_images, val_captions, caption_to_image_idx, info


class FlickrTrainDataset(Dataset):
    def __init__(self, pairs, img_dir):
        self.pairs = pairs
        self.img_dir = img_dir

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, caption = self.pairs[idx]
        img_name = normalize_image_name(img_name)
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert("RGB")
        return image, caption


class FlickrImageDataset(Dataset):
    def __init__(self, image_names, img_dir):
        self.image_names = image_names
        self.img_dir = img_dir

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = normalize_image_name(self.image_names[idx])
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert("RGB")
        return image, img_name


class TextOnlyDataset(Dataset):
    def __init__(self, captions):
        self.captions = captions

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        return self.captions[idx]


class LayerwiseCLIP(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", k_dim=128, train_text=True):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.k_dim = k_dim
        self.num_layers = self.clip.text_model.config.num_hidden_layers
        self.embed_dim = self.clip.projection_dim

        for p in self.clip.parameters():
            p.requires_grad = False

        if train_text:
            for p in self.clip.text_model.parameters():
                p.requires_grad = True

            if isinstance(self.clip.text_projection, nn.Module):
                for p in self.clip.text_projection.parameters():
                    p.requires_grad = True
            elif isinstance(self.clip.text_projection, torch.nn.Parameter):
                self.clip.text_projection.requires_grad = True

            if hasattr(self.clip, "logit_scale") and isinstance(self.clip.logit_scale, torch.nn.Parameter):
                self.clip.logit_scale.requires_grad = True

    def _apply_text_projection(self, x):
        proj = self.clip.text_projection
        if isinstance(proj, nn.Linear):
            return proj(x)
        elif isinstance(proj, torch.nn.Parameter):
            return x @ proj
        elif torch.is_tensor(proj):
            return x @ proj
        raise TypeError(f"Unsupported text_projection type: {type(proj)}")

    def _build_causal_attention_mask(self, bsz, seq_len, dtype, device):
        mask = torch.full((seq_len, seq_len), fill_value=torch.finfo(dtype).min, dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, seq_len, seq_len)

    def _build_attention_mask(self, attention_mask, dtype):
        bsz, seq_len = attention_mask.shape
        expanded = attention_mask[:, None, None, :].expand(bsz, 1, seq_len, seq_len).to(dtype)
        return (1.0 - expanded) * torch.finfo(dtype).min

    @torch.no_grad()
    def encode_images(self, pixel_values):
        image_features = self.clip.get_image_features(pixel_values=pixel_values)
        if not torch.is_tensor(image_features):
            if hasattr(image_features, "image_embeds"):
                image_features = image_features.image_embeds
            elif hasattr(image_features, "pooler_output"):
                image_features = image_features.pooler_output
            else:
                raise TypeError(f"Unexpected image feature type: {type(image_features)}")
        return F.normalize(image_features, dim=-1)

    def encode_text_all_layers(self, input_ids, attention_mask):
        text_model = self.clip.text_model
        device = input_ids.device
        bsz, seq_len = input_ids.shape

        hidden_states = text_model.embeddings(input_ids=input_ids)

        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype, device)
        full_attention_mask = self._build_attention_mask(attention_mask, hidden_states.dtype)

        eos_pos = input_ids.argmax(dim=-1)
        batch_idx = torch.arange(bsz, device=device)

        layer_features = []
        for layer in text_model.encoder.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=full_attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=False,
            )

            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            elif hasattr(layer_outputs, "last_hidden_state"):
                hidden_states = layer_outputs.last_hidden_state
            elif hasattr(layer_outputs, "hidden_states"):
                hidden_states = layer_outputs.hidden_states
            else:
                hidden_states = layer_outputs

            hs = text_model.final_layer_norm(hidden_states)
            eos_hidden = hs[batch_idx, eos_pos]
            txt = self._apply_text_projection(eos_hidden)
            txt = F.normalize(txt, dim=-1)
            layer_features.append(txt)

        return layer_features


class ESECLIP(LayerwiseCLIP):
    def __init__(self, model_name="openai/clip-vit-base-patch32", k_dim=128):
        super().__init__(model_name=model_name, k_dim=k_dim, train_text=True)
        self.compressors = nn.ModuleList([
            nn.Linear(self.embed_dim, k_dim) for _ in range(self.num_layers)
        ])

    def forward(self, pixel_values, input_ids, attention_mask):
        image_features = self.encode_images(pixel_values)
        text_layer_features = self.encode_text_all_layers(input_ids, attention_mask)
        return image_features, text_layer_features


def bi_contrastive_loss(x, y, logit_scale):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    logits = logit_scale * (x @ y.t())
    labels = torch.arange(x.size(0), device=x.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def compute_layer_weights(
    num_layers,
    epoch,
    total_epochs,
    mode="curriculum_shallow_focus",
    shallow_alpha=1.8,
    shallow_power=1.5,
    late_alpha=0.6,
    final_layer_discount=0.85,
    curriculum_fraction=0.6,
):
    if mode == "uniform":
        weights = np.ones(num_layers, dtype=np.float32)

    elif mode == "log_decay":
        weights = []
        for idx in range(1, num_layers + 1):
            if idx == num_layers:
                w = 1.0
            else:
                i = num_layers - idx + 1
                w = 1.0 / (1.0 + math.log(i))
            weights.append(w)
        weights = np.asarray(weights, dtype=np.float32)

    elif mode == "shallow_focus":
        z = np.linspace(0.0, 1.0, num_layers, dtype=np.float32)
        weights = 1.0 + shallow_alpha * np.power(1.0 - z, shallow_power)
        weights[-1] *= final_layer_discount
        weights = weights.astype(np.float32)

    elif mode == "curriculum_shallow_focus":
        z = np.linspace(0.0, 1.0, num_layers, dtype=np.float32)
        early = 1.0 + shallow_alpha * np.power(1.0 - z, shallow_power)
        late = 1.0 + late_alpha * (1.0 - z)

        if total_epochs <= 1:
            lam = 0.0
        else:
            transition_epochs = max(1, int(round(total_epochs * curriculum_fraction)))
            progress = min(max((epoch - 1) / max(transition_epochs - 1, 1), 0.0), 1.0)
            lam = 1.0 - progress

        weights = lam * early + (1.0 - lam) * late
        weights[-1] *= final_layer_discount
        weights = weights.astype(np.float32)

    else:
        raise ValueError(f"Unknown weight mode: {mode}")

    weights = weights * (num_layers / weights.sum())
    return weights


def compute_express_loss(
    text_layer_features,
    image_features,
    k_dim,
    logit_scale,
    layer_weights,
):
    img_k = F.normalize(image_features[:, :k_dim], dim=-1)

    weights_tensor = torch.tensor(layer_weights, dtype=image_features.dtype, device=image_features.device)
    total = 0.0

    for idx, txt in enumerate(text_layer_features):
        txt_k = F.normalize(txt[:, :k_dim], dim=-1)
        total = total + weights_tensor[idx] * bi_contrastive_loss(txt_k, img_k, logit_scale)

    return total / weights_tensor.sum()


def compute_compression_loss(model, text_layer_features, temperature=1.0):
    total = 0.0
    for i, feat in enumerate(text_layer_features):
        prefix = feat[:, :model.k_dim]
        compressed = model.compressors[i](feat)

        mse = F.mse_loss(compressed, prefix)
        p_log = F.log_softmax(compressed / temperature, dim=-1)
        q = F.softmax(prefix / temperature, dim=-1)
        kl = F.kl_div(p_log, q, reduction="batchmean")
        total += (mse + kl)

    return total / len(text_layer_features)


def compute_total_loss(model, image_features, text_layer_features, layer_weights):
    logit_scale = model.clip.logit_scale.exp().clamp(max=100) if hasattr(model.clip, "logit_scale") else torch.tensor(1.0, device=image_features.device)

    cross_modal_loss = bi_contrastive_loss(text_layer_features[-1], image_features, logit_scale)
    express_loss = compute_express_loss(text_layer_features, image_features, model.k_dim, logit_scale, layer_weights)
    compression_loss = compute_compression_loss(model, text_layer_features)

    total_loss = cross_modal_loss + express_loss + compression_loss
    return total_loss, {
        "total_loss": total_loss.detach(),
        "cross_modal_loss": cross_modal_loss.detach(),
        "express_loss": express_loss.detach(),
        "compression_loss": compression_loss.detach(),
    }


@torch.no_grad()
def encode_all_images(model, image_loader, device, use_prefix=False, k_dim=128):
    model.eval()
    all_feats = []

    for batch in tqdm(image_loader, desc="Encoding images", leave=False):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        feats = model.encode_images(pixel_values)
        feats = F.normalize(feats[:, :k_dim], dim=-1) if use_prefix else F.normalize(feats, dim=-1)
        all_feats.append(feats.cpu())

    return torch.cat(all_feats, dim=0)


@torch.no_grad()
def encode_all_texts_by_layer(model, text_loader, device, use_prefix=False, k_dim=128):
    model.eval()
    all_layer_feats = None

    for batch in tqdm(text_loader, desc="Encoding texts", leave=False):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        layer_feats = model.encode_text_all_layers(input_ids, attention_mask)

        if all_layer_feats is None:
            all_layer_feats = [[] for _ in range(len(layer_feats))]

        for i, feat in enumerate(layer_feats):
            feat = F.normalize(feat[:, :k_dim], dim=-1) if use_prefix else F.normalize(feat, dim=-1)
            all_layer_feats[i].append(feat.cpu())

    return [torch.cat(parts, dim=0) for parts in all_layer_feats]


def compute_recall_metrics(image_feats, text_feats, caption_to_image_idx, topk_list=(1, 5)):
    image_feats = F.normalize(image_feats, dim=-1)
    text_feats = F.normalize(text_feats, dim=-1)

    sim_i2t = image_feats @ text_feats.t()
    sim_t2i = sim_i2t.t()

    num_images = image_feats.size(0)
    num_captions = text_feats.size(0)

    image_to_captions = [[] for _ in range(num_images)]
    for cap_idx, img_idx in enumerate(caption_to_image_idx):
        image_to_captions[img_idx].append(cap_idx)

    results = {}
    for k in topk_list:
        topk_caps = sim_i2t.topk(k, dim=1).indices.cpu().numpy()
        hits_i2t = 0
        for img_idx in range(num_images):
            gt_caps = set(image_to_captions[img_idx])
            pred_caps = set(topk_caps[img_idx].tolist())
            if len(gt_caps & pred_caps) > 0:
                hits_i2t += 1
        results[f"I2T_R@{k}"] = 100.0 * hits_i2t / num_images

        topk_imgs = sim_t2i.topk(k, dim=1).indices.cpu().numpy()
        hits_t2i = 0
        for cap_idx in range(num_captions):
            if caption_to_image_idx[cap_idx] in topk_imgs[cap_idx]:
                hits_t2i += 1
        results[f"T2I_R@{k}"] = 100.0 * hits_t2i / num_captions

    results["MeanR"] = np.mean(list(results.values()))
    return results


@torch.no_grad()
def evaluate_model_all_layers(model, image_loader, text_loader, caption_to_image_idx, device, use_prefix=False, k_dim=128):
    image_feats = encode_all_images(model, image_loader, device, use_prefix=use_prefix, k_dim=k_dim)
    text_layer_feats = encode_all_texts_by_layer(model, text_loader, device, use_prefix=use_prefix, k_dim=k_dim)

    rows = []
    for layer_idx, txt_feats in enumerate(text_layer_feats, start=1):
        metrics = compute_recall_metrics(image_feats, txt_feats, caption_to_image_idx, topk_list=(1, 5))
        row = {"Layer": layer_idx}
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)


def build_dataloaders(args, processor, tokenizer, train_pairs, val_images, val_captions):
    def train_collate_fn(batch):
        images, captions = zip(*batch)
        pixel_values = processor(images=list(images), return_tensors="pt")["pixel_values"]
        text_inputs = tokenizer(
            list(captions),
            padding=True,
            truncation=True,
            max_length=args.max_text_len,
            return_tensors="pt"
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
        }

    def image_collate_fn(batch):
        images, names = zip(*batch)
        pixel_values = processor(images=list(images), return_tensors="pt")["pixel_values"]
        return {"pixel_values": pixel_values, "image_names": list(names)}

    def text_collate_fn(batch):
        text_inputs = tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=args.max_text_len,
            return_tensors="pt"
        )
        return {"input_ids": text_inputs["input_ids"], "attention_mask": text_inputs["attention_mask"]}

    dl_kwargs = dict(num_workers=args.num_workers, pin_memory=args.pin_memory)
    if args.num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(
        FlickrTrainDataset(train_pairs, args.image_dir),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_collate_fn,
        **dl_kwargs,
    )

    val_image_loader = DataLoader(
        FlickrImageDataset(val_images, args.image_dir),
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=image_collate_fn,
        **dl_kwargs,
    )

    val_text_loader = DataLoader(
        TextOnlyDataset(val_captions),
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=text_collate_fn,
        **dl_kwargs,
    )

    return train_loader, val_image_loader, val_text_loader


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--caption_file", type=str, required=True)
    parser.add_argument("--split_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--k_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_text_len", type=int, default=77)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2)

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--no_pin_memory", action="store_false", dest="pin_memory")
    parser.set_defaults(pin_memory=True)

    parser.add_argument("--no_official_split", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.set_defaults(amp=True)

    parser.add_argument("--express_weight_mode", type=str, default="curriculum_shallow_focus",
                        choices=["log_decay", "uniform", "shallow_focus", "curriculum_shallow_focus"])
    parser.add_argument("--shallow_alpha", type=float, default=1.8)
    parser.add_argument("--shallow_power", type=float, default=1.5)
    parser.add_argument("--late_alpha", type=float, default=0.6)
    parser.add_argument("--final_layer_discount", type=float, default=0.85)
    parser.add_argument("--curriculum_fraction", type=float, default=0.6)

    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    set_seed(args.seed)

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = args.device if (torch.cuda.is_available() and args.device == "cuda") else "cpu"
    print(f"[System] device = {device}")

    if hasattr(torch.backends, "cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    save_json(vars(args), os.path.join(args.output_dir, "run_args.json"))

    train_pairs, val_images, val_captions, caption_to_image_idx, data_info = build_splits_and_pairs(args)
    save_json(data_info, os.path.join(args.output_dir, "data_info.json"))

    processor = CLIPProcessor.from_pretrained(args.model_name)
    tokenizer = CLIPTokenizerFast.from_pretrained(args.model_name)
    train_loader, val_image_loader, val_text_loader = build_dataloaders(args, processor, tokenizer, train_pairs, val_images, val_captions)

    print("[Check] quick text-layer extraction check...")
    tmp_model = LayerwiseCLIP(model_name=args.model_name, k_dim=args.k_dim, train_text=False).to(device)
    sample_batch = next(iter(val_text_loader))
    sample_input_ids = sample_batch["input_ids"][:4].to(device)
    sample_attention_mask = sample_batch["attention_mask"][:4].to(device)
    tmp_layers = tmp_model.encode_text_all_layers(sample_input_ids, sample_attention_mask)
    print(f"[Check] num layers = {len(tmp_layers)}")
    print(f"[Check] layer1 shape = {tuple(tmp_layers[0].shape)}")
    print(f"[Check] layer12 shape = {tuple(tmp_layers[-1].shape)}")
    del tmp_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    raw_clip = LayerwiseCLIP(model_name=args.model_name, k_dim=args.k_dim, train_text=False).to(device)

    print("\n========== Raw CLIP: full 512 ==========")
    raw_full_df = evaluate_model_all_layers(raw_clip, val_image_loader, val_text_loader, caption_to_image_idx, device, False, args.k_dim)
    raw_full_df["Model"] = "Raw-CLIP"
    raw_full_df["EvalMode"] = "full_512"
    print(raw_full_df)

    print(f"\n========== Raw CLIP: prefix_{args.k_dim} ==========")
    raw_prefix_df = evaluate_model_all_layers(raw_clip, val_image_loader, val_text_loader, caption_to_image_idx, device, True, args.k_dim)
    raw_prefix_df["Model"] = "Raw-CLIP"
    raw_prefix_df["EvalMode"] = f"prefix_{args.k_dim}"
    print(raw_prefix_df)

    raw_full_df.to_csv(os.path.join(args.output_dir, "raw_full_512.csv"), index=False)
    raw_prefix_df.to_csv(os.path.join(args.output_dir, f"raw_prefix_{args.k_dim}.csv"), index=False)

    ese_model = ESECLIP(model_name=args.model_name, k_dim=args.k_dim).to(device)
    if args.resume is not None and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        ese_model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Resume] loaded checkpoint from {args.resume}")

    if not args.eval_only:
        trainable_params = [p for p in ese_model.parameters() if p.requires_grad]
        print(f"[Train] trainable params: {sum(p.numel() for p in trainable_params)/1e6:.2f}M")

        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        num_training_steps = args.epochs * len(train_loader)
        num_warmup_steps = int(0.1 * num_training_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and args.amp))
        best_score = -1.0
        best_ckpt_path = os.path.join(args.output_dir, f"best_ese_clip_k{args.k_dim}.pt")
        history = []

        for epoch in range(1, args.epochs + 1):
            ese_model.train()
            epoch_start = time.time()

            current_weights = compute_layer_weights(
                num_layers=ese_model.num_layers,
                epoch=epoch,
                total_epochs=args.epochs,
                mode=args.express_weight_mode,
                shallow_alpha=args.shallow_alpha,
                shallow_power=args.shallow_power,
                late_alpha=args.late_alpha,
                final_layer_discount=args.final_layer_discount,
                curriculum_fraction=args.curriculum_fraction,
            )
            print(f"[Epoch {epoch}] express weights = {[round(float(x), 3) for x in current_weights]}")

            running_total = 0.0
            running_cross = 0.0
            running_expr = 0.0
            running_comp = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
            for step, batch in enumerate(pbar, start=1):
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                amp_ctx = torch.cuda.amp.autocast() if (device == "cuda" and args.amp) else nullcontext()

                with amp_ctx:
                    image_features, text_layer_features = ese_model(pixel_values, input_ids, attention_mask)
                    loss, loss_dict = compute_total_loss(ese_model, image_features, text_layer_features, current_weights)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                running_total += loss_dict["total_loss"].item()
                running_cross += loss_dict["cross_modal_loss"].item()
                running_expr += loss_dict["express_loss"].item()
                running_comp += loss_dict["compression_loss"].item()

                pbar.set_postfix({
                    "loss": f"{running_total/step:.4f}",
                    "cross": f"{running_cross/step:.4f}",
                    "expr": f"{running_expr/step:.4f}",
                    "comp": f"{running_comp/step:.4f}",
                })

            avg_total = running_total / len(train_loader)
            avg_cross = running_cross / len(train_loader)
            avg_expr = running_expr / len(train_loader)
            avg_comp = running_comp / len(train_loader)

            val_prefix_df = evaluate_model_all_layers(
                ese_model, val_image_loader, val_text_loader, caption_to_image_idx, device, True, args.k_dim
            )
            last_layer_row = val_prefix_df[val_prefix_df["Layer"] == ese_model.num_layers].iloc[0]
            val_score = float(last_layer_row["MeanR"])
            epoch_time = time.time() - epoch_start

            history.append({
                "epoch": epoch,
                "train_total_loss": avg_total,
                "train_cross_modal_loss": avg_cross,
                "train_express_loss": avg_expr,
                "train_compression_loss": avg_comp,
                "val_lastlayer_prefix_I2T_R@1": float(last_layer_row["I2T_R@1"]),
                "val_lastlayer_prefix_T2I_R@1": float(last_layer_row["T2I_R@1"]),
                "val_lastlayer_prefix_I2T_R@5": float(last_layer_row["I2T_R@5"]),
                "val_lastlayer_prefix_T2I_R@5": float(last_layer_row["T2I_R@5"]),
                "val_lastlayer_prefix_MeanR": val_score,
                "epoch_time_sec": epoch_time,
                "layer_weights": " ".join([f"{float(x):.4f}" for x in current_weights]),
            })

            print(f"\n[Epoch {epoch}]")
            print(f"train_total_loss       = {avg_total:.4f}")
            print(f"train_cross_modal_loss = {avg_cross:.4f}")
            print(f"train_express_loss     = {avg_expr:.4f}")
            print(f"train_compression_loss = {avg_comp:.4f}")
            print(f"val_lastlayer_prefix_MeanR = {val_score:.4f}")
            print(f"epoch_time_sec = {epoch_time:.2f}")

            if val_score > best_score:
                best_score = val_score
                torch.save(
                    {
                        "model_state_dict": ese_model.state_dict(),
                        "epoch": epoch,
                        "best_score": best_score,
                        "k_dim": args.k_dim,
                        "express_weight_mode": args.express_weight_mode,
                    },
                    best_ckpt_path,
                )
                print(f"[Checkpoint] saved best checkpoint to {best_ckpt_path}")

        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(args.output_dir, f"train_history_k{args.k_dim}.csv"), index=False)

        if os.path.exists(best_ckpt_path):
            ckpt = torch.load(best_ckpt_path, map_location=device)
            ese_model.load_state_dict(ckpt["model_state_dict"])
            print(f"[Checkpoint] loaded best checkpoint from {best_ckpt_path}")
    else:
        print("[Mode] eval_only=True, skip training")

    ese_model.eval()

    print("\n========== ESE-CLIP: full 512 ==========")
    ese_full_df = evaluate_model_all_layers(ese_model, val_image_loader, val_text_loader, caption_to_image_idx, device, False, args.k_dim)
    ese_full_df["Model"] = "ESE-CLIP"
    ese_full_df["EvalMode"] = "full_512"
    print(ese_full_df)

    print(f"\n========== ESE-CLIP: prefix_{args.k_dim} ==========")
    ese_prefix_df = evaluate_model_all_layers(ese_model, val_image_loader, val_text_loader, caption_to_image_idx, device, True, args.k_dim)
    ese_prefix_df["Model"] = "ESE-CLIP"
    ese_prefix_df["EvalMode"] = f"prefix_{args.k_dim}"
    print(ese_prefix_df)

    ese_full_df.to_csv(os.path.join(args.output_dir, "ese_full_512.csv"), index=False)
    ese_prefix_df.to_csv(os.path.join(args.output_dir, f"ese_prefix_{args.k_dim}.csv"), index=False)

    compare_df = pd.concat([raw_full_df, raw_prefix_df, ese_full_df, ese_prefix_df], ignore_index=True)[
        ["Model", "EvalMode", "Layer", "I2T_R@1", "I2T_R@5", "T2I_R@1", "T2I_R@5", "MeanR"]
    ].sort_values(["EvalMode", "Model", "Layer"]).reset_index(drop=True)

    prefix_summary_df = pd.concat([raw_prefix_df, ese_prefix_df], ignore_index=True)[
        ["Model", "EvalMode", "Layer", "I2T_R@1", "I2T_R@5", "T2I_R@1", "T2I_R@5", "MeanR"]
    ].sort_values(["Model", "Layer"]).reset_index(drop=True)

    compare_df.to_csv(os.path.join(args.output_dir, f"final_compare_k{args.k_dim}.csv"), index=False)
    prefix_summary_df.to_csv(os.path.join(args.output_dir, f"prefix_summary_compare_k{args.k_dim}.csv"), index=False)

    best_shallow_ese = ese_prefix_df[ese_prefix_df["Layer"] < ese_model.num_layers].sort_values("MeanR", ascending=False).iloc[0]
    final_ese = ese_prefix_df[ese_prefix_df["Layer"] == ese_model.num_layers].iloc[0]

    print("\n========== Key Result ==========")
    print(f"Best shallow ESE layer: L{int(best_shallow_ese['Layer'])}, MeanR={best_shallow_ese['MeanR']:.4f}")
    print(f"Final ESE layer:        L{int(final_ese['Layer'])}, MeanR={final_ese['MeanR']:.4f}")
    print(f"Gap(final - best shallow) = {final_ese['MeanR'] - best_shallow_ese['MeanR']:.4f}")
    print("\n[Done] outputs saved to:", args.output_dir)


if __name__ == "__main__":
    main()

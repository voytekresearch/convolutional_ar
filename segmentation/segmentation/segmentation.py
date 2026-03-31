import math
import random
import time
import os
import hashlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm
import json

# Repro + device
SEED = 1337
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

PATCH_SIZE = (16, 16, 16)

DATA_CFG = {

    # Per-group split config (group = tensors/{x}/...)
    "group_split_enabled": True,
    "group_split_root": "tensors",
    "split_ratios": (0.8, 0.1, 0.1),
    "split_seed": SEED,

    # Fallback global split if grouped split is disabled
    "val_fraction": 0.1,

    # Optional caps for debugging
    "max_train_samples": None,
    "max_val_samples": None,

    # If None, inferred from label_values/label scan below.
    "n_classes": None,

    # Optional explicit sparse label ids (e.g. [0, 2, 3, 4, 1001, ...]).
    # If empty and use_default_aparc_aseg_labels=True, use the canonical aparc+aseg ID set.
    # Otherwise scan a bounded number of y-files to build a sparse->dense LUT.
    "label_values": [],
    "use_default_aparc_aseg_labels": True,
    "label_scan_max_files": 64,

    # Keep only this many x/y files opened in-memory at once.
    "max_open_files": 1,

    # Preprocessed cache to remove CPU bottlenecks at train time.
    "cache_enabled": True,
    "cache_dir": ".tensor_cache_preproc",
    "cache_rebuild": False,
    "cache_apply_label_lut": True,
    "cache_zscore_x": True,
    "cache_x_dtype": "float16",
    "cache_y_dtype": "int16",
    # Behavior when source tensors cannot be loaded: "skip" or "raise".
    "cache_on_load_error": "skip",
    # Behavior when labels are not covered by current LUT during cache build: "skip" or "raise".
    "cache_on_label_miss": "skip",
    # If False, previously skipped files are not retried on every run.
    "cache_retry_skipped": False,
}

MODEL_CFG = {
    "in_channels": 1,
    "channels": (16, 24, 48, 72, 96, 128),
    "transformer_depth": 2,
    "n_heads": 4,
    "dropout": 0.1,
}

TRAIN_CFG = {
    "epochs": 200,
    "effective_batch_size": 2,
    "initial_micro_batch_size": 1,
    "patch_chunk_size": 64,

    # Use windows for VRAM safety. None => full volume.
    "spatial_window": (96, 96, 96),
    "spatial_stride": None,

    "lr": 3e-4,
    "min_lr": 1e-6,
    "weight_decay": 1e-4,
    "grad_clip_norm": 1.0,
    "target_max_vram_gb": 31.0,

    "num_workers": 4,
    "pin_memory": True,
    "prefetch_factor": 2,

    # Frequent validation reporting
    "quick_val_every_steps": 50,
    "quick_val_batches": 4,

    # Checkpoint / stopping
    "checkpoint_path": "checkpoints/transunet3d_best.pt",
    "early_stopping_patience": 20,

    # If an epoch runs cleanly (no OOM), try increasing micro-batch.
    "auto_increase_micro_batch": False,
}



def auto_discover_pairs(root=Path("tensors")):
    # TODO: move to utils
    root = Path(root)
    if not root.exists():
        return [], []

    x_files, y_files = [], []
    for sub in sorted([p for p in root.rglob("*") if p.is_dir()]):
        x = sub / "rawavg.pt"
        y = sub / "aparc+aseg.pt"
        if x.exists() and y.exists():
            x_files.append(str(x))
            y_files.append(str(y))
    return x_files, y_files


# Helpers + model
def gpu_mem_gb():
    # TODO: move to utils
    if DEVICE != "cuda":
        return 0.0, 0.0
    return (
        torch.cuda.memory_allocated() / (1024 ** 3),
        torch.cuda.max_memory_allocated() / (1024 ** 3),
    )


def ensure_divisible(shape, patch_size=PATCH_SIZE):
    # TODO: move to utils
    if any(s % p != 0 for s, p in zip(shape, patch_size)):
        raise ValueError(f"shape={shape} must be divisible by patch_size={patch_size}")


def patches_to_volume(patches, volume_shape, patch_size=PATCH_SIZE):
    # patches: [B, N, V] -> volume: [B, D, H, W]
    if patches.ndim != 3:
        raise ValueError(f"Expected [B, N, V], got {tuple(patches.shape)}")
    b, n, v = patches.shape
    pd, ph, pw = patch_size
    gd, gh, gw = (volume_shape[0] // pd, volume_shape[1] // ph, volume_shape[2] // pw)
    expected_n = gd * gh * gw
    expected_v = pd * ph * pw
    if (n, v) != (expected_n, expected_v):
        raise ValueError(f"Expected [B, {expected_n}, {expected_v}], got {tuple(patches.shape)}")
    x = patches.reshape(b, gd, gh, gw, pd, ph, pw)
    x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    return x.reshape(b, *volume_shape)


def volume_to_patches(volume, patch_size=PATCH_SIZE):
    # volume: [B, D, H, W] -> patches: [B, N, V]
    if volume.ndim != 4:
        raise ValueError(f"Expected [B, D, H, W], got {tuple(volume.shape)}")
    b, d, h, w = volume.shape
    ensure_divisible((d, h, w), patch_size)
    pd, ph, pw = patch_size
    gd, gh, gw = (d // pd, h // ph, w // pw)
    x = volume.reshape(b, gd, pd, gh, ph, gw, pw)
    x = x.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
    return x.reshape(b, gd * gh * gw, pd * ph * pw)


def _positions_1d(dim, win, stride):
    starts = list(range(0, dim - win + 1, stride))
    if starts[-1] != dim - win:
        starts.append(dim - win)
    return starts


def make_window_slices(spatial_shape, window_shape, stride_shape):
    d, h, w = spatial_shape
    wd, wh, ww = window_shape
    sd, sh, sw = stride_shape
    zs = _positions_1d(d, wd, sd)
    ys = _positions_1d(h, wh, sh)
    xs = _positions_1d(w, ww, sw)
    return [
        (slice(z0, z0 + wd), slice(y0, y0 + wh), slice(x0, x0 + ww))
        for z0 in zs for y0 in ys for x0 in xs
    ]


def fit_window_to_shape(spatial_shape, desired_window, patch_size=PATCH_SIZE):
    # Clamp requested window to fit current volume and keep divisibility by patch_size.
    if desired_window is None:
        return tuple(int(s) for s in spatial_shape)
    out = []
    for dim, req, p in zip(spatial_shape, desired_window, patch_size):
        w = min(int(req), int(dim))
        w = max(int(p), (w // int(p)) * int(p))
        out.append(w)
    return tuple(out)


def halve_window_shape(window_shape, patch_size=PATCH_SIZE):
    out = []
    for s, p in zip(window_shape, patch_size):
        half = max(int(p), int(s) // 2)
        half = max(int(p), (half // int(p)) * int(p))
        out.append(half)
    return tuple(out)


def _group_norm(ch):
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _group_norm(out_ch),
            nn.GELU(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _group_norm(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class DownBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.down = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.block = ConvBlock3D(out_ch, out_ch, dropout=dropout)

    def forward(self, x):
        return self.block(self.down(x))


class UpBlock3D(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.reduce = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        self.block = ConvBlock3D(out_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class TransUNet3D(nn.Module):
    def __init__(
        self,
        n_classes,
        in_channels=1,
        base_shape=(208, 240, 192),
        patch_size=PATCH_SIZE,
        channels=(16, 24, 48, 72, 96, 128),
        transformer_depth=2,
        n_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        if len(channels) != 6:
            raise ValueError("channels must have length 6")

        self.n_classes = int(n_classes)
        self.base_shape = tuple(int(v) for v in base_shape)
        self.patch_size = tuple(int(v) for v in patch_size)

        c0, c1, c2, c3, c4, c5 = [int(c) for c in channels]

        self.stem = ConvBlock3D(in_channels, c0, dropout=0.0)
        self.down1 = DownBlock3D(c0, c1, dropout=0.0)
        self.down2 = DownBlock3D(c1, c2, dropout=0.0)
        self.down3 = DownBlock3D(c2, c3, dropout=dropout)
        self.down4 = DownBlock3D(c3, c4, dropout=dropout)
        self.down5 = DownBlock3D(c4, c5, dropout=dropout)
        self.bottleneck_conv = ConvBlock3D(c5, c5, dropout=dropout)

        g = list(self.base_shape)
        for _ in range(5):
            g = [(x + 1) // 2 for x in g]
        self.base_bottleneck_grid = tuple(g)
        self.pos_embed = nn.Parameter(torch.zeros(1, int(math.prod(g)), c5))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=c5,
            nhead=n_heads,
            dim_feedforward=4 * c5,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_depth)

        self.up5 = UpBlock3D(c5, c4, c4, dropout=dropout)
        self.up4 = UpBlock3D(c4, c3, c3, dropout=dropout)
        self.up3 = UpBlock3D(c3, c2, c2, dropout=dropout)
        self.up2 = UpBlock3D(c2, c1, c1, dropout=0.0)
        self.up1 = UpBlock3D(c1, c0, c0, dropout=0.0)

        self.token_norm = nn.LayerNorm(c0)
        self.classifier = nn.Linear(c0, self.n_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _token_positional_embedding(self, btm):
        seq_len = btm.shape[2] * btm.shape[3] * btm.shape[4]
        if seq_len == self.pos_embed.shape[1]:
            return self.pos_embed
        pos3 = self.pos_embed.transpose(1, 2).reshape(1, btm.size(1), *self.base_bottleneck_grid)
        pos3 = F.interpolate(pos3, size=btm.shape[2:], mode="trilinear", align_corners=False)
        return pos3.flatten(2).transpose(1, 2)

    def forward_features(self, x):
        # x: [B, 1, D, H, W]
        if x.ndim != 5 or x.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, D, H, W], got {tuple(x.shape)}")

        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        btm = self.bottleneck_conv(x5)
        seq = btm.flatten(2).transpose(1, 2)
        seq = self.transformer(seq + self._token_positional_embedding(btm))
        btm = seq.transpose(1, 2).reshape_as(btm)

        d4 = self.up5(btm, x4)
        d3 = self.up4(d4, x3)
        d2 = self.up3(d3, x2)
        d1 = self.up2(d2, x1)
        d0 = self.up1(d1, x0)
        return d0

    def features_to_patch_features(self, feat):
        # feat: [B, C, D, H, W] -> [B, N, V, C]
        if feat.ndim != 5:
            raise ValueError(f"Expected [B, C, D, H, W], got {tuple(feat.shape)}")
        b, c, d, h, w = feat.shape
        ensure_divisible((d, h, w), self.patch_size)
        pd, ph, pw = self.patch_size
        gd, gh, gw = d // pd, h // ph, w // pw
        x = feat.reshape(b, c, gd, pd, gh, ph, gw, pw)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        x = x.reshape(b, gd * gh * gw, pd * ph * pw, c)
        return self.token_norm(x)

    def _iter_patch_logits(self, patch_feat, patch_chunk_size=96):
        n = patch_feat.shape[1]
        for s in range(0, n, int(patch_chunk_size)):
            e = min(s + int(patch_chunk_size), n)
            chunk_feat = patch_feat[:, s:e]
            chunk_logits = self.classifier(chunk_feat)
            yield s, e, chunk_logits

    def loss_from_features(self, feat, y_volume, patch_chunk_size=96):
        # y_volume: [B, D, H, W]
        y_patch = volume_to_patches(y_volume, self.patch_size).long()
        patch_feat = self.features_to_patch_features(feat)

        total_num = 0.0
        total_den = 0
        for s, e, logit in self._iter_patch_logits(patch_feat, patch_chunk_size=patch_chunk_size):
            tgt = y_patch[:, s:e]
            ce = F.cross_entropy(logit.permute(0, 3, 1, 2), tgt)
            n = int(tgt.numel())
            total_num += ce * n
            total_den += n
        return total_num / max(1, total_den)

    @torch.no_grad()
    def predict_volume(self, x, patch_chunk_size=96):
        feat = self.forward_features(x)
        patch_feat = self.features_to_patch_features(feat)
        pred_patch = []
        for _, _, logit in self._iter_patch_logits(patch_feat, patch_chunk_size=patch_chunk_size):
            pred_patch.append(logit.argmax(dim=-1).cpu())
        pred_patch = torch.cat(pred_patch, dim=1)
        return patches_to_volume(pred_patch, volume_shape=tuple(x.shape[2:]), patch_size=self.patch_size)


# Streaming dataset + DataLoader (plain torch.load, no conversion)
def resolve_paths(items):
    if items is None:
        return []
    if isinstance(items, (str, Path)):
        items = [items]

    paths = []
    for item in items:
        p = Path(item).expanduser()
        if p.suffix.lower() in {".txt", ".lst"}:
            lines_ = [ln.strip() for ln in p.read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]
            paths.extend(lines_)
        elif p.suffix.lower() == ".json":
            payload = json.loads(p.read_text())
            if not isinstance(payload, list):
                raise ValueError(f"Expected list in {p}, got {type(payload)}")
            paths.extend([str(x) for x in payload])
        else:
            paths.append(str(p))

    return [str(Path(x).expanduser()) for x in paths]


def _to_batched_3d(t, name):
    # Normalize to [N, D, H, W]
    if t.ndim == 3:
        t = t.unsqueeze(0)
    elif t.ndim == 4:
        pass
    elif t.ndim == 5 and t.shape[1] == 1:
        t = t[:, 0]
    else:
        raise ValueError(f"{name}: expected [D,H,W], [N,D,H,W], or [N,1,D,H,W], got {tuple(t.shape)}")
    return t


def load_tensor_pt(path):
    # Requested simple loader path.
    t = torch.load(path)
    if not torch.is_tensor(t):
        raise TypeError(f"{path} did not load as a Tensor. Got: {type(t)}")
    return _to_batched_3d(t, path)


def _group_key_from_path(x_path, root):
    p = Path(x_path).resolve()
    root = Path(root).resolve()
    try:
        rel = p.relative_to(root)
        return rel.parts[0] if len(rel.parts) > 0 else p.parent.name
    except Exception:
        return p.parent.as_posix()


def _group_split_counts(n, ratios):
    tr, va, te = ratios
    if n <= 0:
        return 0, 0, 0

    n_val = int(round(n * va))
    n_test = int(round(n * te))

    # For reasonably sized groups, enforce non-empty val/test.
    if n >= 10:
        n_val = max(1, n_val)
        n_test = max(1, n_test)

    # Ensure at least 1 train sample.
    if n_val + n_test >= n:
        overflow = n_val + n_test - (n - 1)
        while overflow > 0 and (n_val > 0 or n_test > 0):
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
            overflow -= 1

    n_train = n - n_val - n_test
    return n_train, n_val, n_test


def split_pairs_by_group(x_files, y_files, root, ratios=(0.8, 0.1, 0.1), seed=1337):
    if len(x_files) != len(y_files):
        raise ValueError("x_files and y_files must have same length")

    grouped = {}
    for x, y in zip(x_files, y_files):
        k = _group_key_from_path(x, root)
        grouped.setdefault(k, []).append((x, y))

    x_tr, y_tr, x_va, y_va, x_te, y_te = [], [], [], [], [], []

    for gk in sorted(grouped.keys()):
        pairs = grouped[gk][:]
        rng = random.Random(int(seed) + sum(ord(c) for c in gk))
        rng.shuffle(pairs)

        n_train, n_val, n_test = _group_split_counts(len(pairs), ratios)

        train_pairs = pairs[:n_train]
        val_pairs = pairs[n_train:n_train + n_val]
        test_pairs = pairs[n_train + n_val:n_train + n_val + n_test]

        for x, y in train_pairs:
            x_tr.append(x); y_tr.append(y)
        for x, y in val_pairs:
            x_va.append(x); y_va.append(y)
        for x, y in test_pairs:
            x_te.append(x); y_te.append(y)

    return x_tr, y_tr, x_va, y_va, x_te, y_te


def summarize_group_splits(x_train_files, x_val_files, x_test_files, root):
    train_counts = {}
    val_counts = {}
    test_counts = {}

    for p in x_train_files:
        k = _group_key_from_path(p, root)
        train_counts[k] = train_counts.get(k, 0) + 1
    for p in x_val_files:
        k = _group_key_from_path(p, root)
        val_counts[k] = val_counts.get(k, 0) + 1
    for p in x_test_files:
        k = _group_key_from_path(p, root)
        test_counts[k] = test_counts.get(k, 0) + 1

    keys = sorted(set(train_counts.keys()) | set(val_counts.keys()) | set(test_counts.keys()))
    if not keys:
        return

    print("per-group file split (train/val/test):")
    for k in keys:
        tr = train_counts.get(k, 0)
        va = val_counts.get(k, 0)
        te = test_counts.get(k, 0)
        n = tr + va + te
        print(f"  {k}: {tr}/{va}/{te} (n={n})")


def build_label_lut(label_values):
    values = torch.as_tensor(label_values, dtype=torch.int64)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("label_values must be a non-empty 1D sequence")
    values = torch.unique(values).sort().values
    if int(values[0].item()) < 0:
        raise ValueError("label_values must be non-negative")
    lut = torch.full((int(values[-1].item()) + 1,), -1, dtype=torch.int64)
    lut[values] = torch.arange(values.numel(), dtype=torch.int64)
    return values, lut


def infer_label_values(y_files, max_files=64, seed=SEED):
    if not y_files:
        raise ValueError("Cannot infer labels from an empty list")

    files = list(y_files)
    if max_files is not None and int(max_files) > 0 and len(files) > int(max_files):
        rng = random.Random(int(seed))
        rng.shuffle(files)
        files = files[: int(max_files)]

    values = set()
    for yp in files:
        y = load_tensor_pt(yp).to(torch.int64)
        values.update(int(v) for v in torch.unique(y).tolist())

    return sorted(values), len(files)


def default_aparc_aseg_label_values():
    # Common FreeSurfer aparc+aseg IDs mapped to a compact contiguous class space.
    vals = [
        0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 30,
        31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62, 63, 72,
        77, 80, 85, 251, 252, 253, 254, 255,
    ]
    vals.extend(range(1000, 1036))
    vals.extend(range(2000, 2036))
    return vals


def dtype_from_name(name):
    n = str(name).lower()
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
    }
    if n not in table:
        raise ValueError(f"Unsupported dtype name: {name}")
    return table[n]


def _cache_paths_for_pair(x_path, y_path, cache_dir, cache_tag):
    digest = hashlib.sha1(f"{Path(x_path).resolve()}|{Path(y_path).resolve()}|{cache_tag}".encode("utf-8")).hexdigest()[:16]
    x_name = Path(x_path).stem
    y_name = Path(y_path).stem
    x_cache = cache_dir / f"{x_name}.{digest}.x.pt"
    y_cache = cache_dir / f"{y_name}.{digest}.y.pt"
    return x_cache, y_cache


def _save_tensor_atomic(t, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(t, tmp)
    os.replace(tmp, out_path)

## TODO: refactor to data.py
def build_cache_for_pairs(x_files, y_files, cache_dir, label_lut, rebuild=False):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    skipped_db_path = cache_dir / "_skipped.json"
    if skipped_db_path.exists():
        try:
            skipped_db = json.loads(skipped_db_path.read_text())
            if not isinstance(skipped_db, dict):
                skipped_db = {}
        except Exception:
            skipped_db = {}
    else:
        skipped_db = {}

    cache_x_dtype = dtype_from_name(DATA_CFG.get("cache_x_dtype", "float16"))
    cache_y_dtype = dtype_from_name(DATA_CFG.get("cache_y_dtype", "int16"))
    cache_zscore_x = bool(DATA_CFG.get("cache_zscore_x", True))
    cache_apply_label_lut = bool(DATA_CFG.get("cache_apply_label_lut", True))
    cache_on_load_error = str(DATA_CFG.get("cache_on_load_error", "skip")).strip().lower()
    if cache_on_load_error not in {"skip", "raise"}:
        raise ValueError("DATA_CFG['cache_on_load_error'] must be 'skip' or 'raise'")
    cache_on_label_miss = str(DATA_CFG.get("cache_on_label_miss", "skip")).strip().lower()
    if cache_on_label_miss not in {"skip", "raise"}:
        raise ValueError("DATA_CFG['cache_on_label_miss'] must be 'skip' or 'raise'")
    cache_retry_skipped = bool(DATA_CFG.get("cache_retry_skipped", False))

    lut_sig = "nolut"
    if label_lut is not None:
        lut_sig = str(int((label_lut >= 0).sum().item()))

    cache_tag = (
        f"xd={DATA_CFG.get('cache_x_dtype','float16')}"
        f"|yd={DATA_CFG.get('cache_y_dtype','int16')}"
        f"|zs={int(cache_zscore_x)}"
        f"|map={int(cache_apply_label_lut)}"
        f"|lut={lut_sig}"
    )

    out_x, out_y = [], []
    built = 0
    reused = 0
    skipped = 0

    iterator = zip(x_files, y_files)
    iterator = tqdm(list(iterator), desc="cache build/check", leave=False)

    for xf, yf in iterator:
        x_cache, y_cache = _cache_paths_for_pair(xf, yf, cache_dir=cache_dir, cache_tag=cache_tag)
        pair_key = hashlib.sha1(f"{Path(xf).resolve()}|{Path(yf).resolve()}".encode("utf-8")).hexdigest()

        if (not rebuild) and x_cache.exists() and y_cache.exists():
            out_x.append(str(x_cache))
            out_y.append(str(y_cache))
            reused += 1
            if pair_key in skipped_db:
                skipped_db.pop(pair_key, None)
            continue

        if (not rebuild) and (not cache_retry_skipped) and pair_key in skipped_db:
            skipped += 1
            continue

        try:
            x = load_tensor_pt(xf)
            y = load_tensor_pt(yf)
        except Exception as e:
            msg = f"Failed to load source tensors: {type(e).__name__}: {e}"
            if cache_on_load_error == "skip":
                skipped_db[pair_key] = {"x": str(xf), "y": str(yf), "reason": msg}
                skipped += 1
                continue
            raise RuntimeError(f"{msg} | x={xf} | y={yf}") from e

        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Sample count mismatch: {xf} has {x.shape[0]}, {yf} has {y.shape[0]}")
        if tuple(x.shape[1:]) != tuple(y.shape[1:]):
            raise ValueError(f"Spatial mismatch: {xf} {tuple(x.shape[1:])} vs {yf} {tuple(y.shape[1:])}")
        if y.dtype.is_floating_point:
            raise TypeError(f"{yf}: labels must be integer class IDs, got float dtype {y.dtype}")
        if x.shape[0] != 1:
            raise ValueError(
                f"{xf}: expected one sample per file ([D,H,W] or [1,D,H,W]), got {tuple(x.shape)}. "
                "Split batched tensors into one sample per file to use this cache pipeline."
            )

        x3 = x[0].to(torch.float32)
        y3 = y[0].to(torch.int64)

        if cache_apply_label_lut and label_lut is not None:
            y_max = int(y3.max().item())
            if y_max >= label_lut.numel():
                msg = (
                    f"Label id {y_max} exceeds current LUT range {label_lut.numel()-1}. "
                    "Set DATA_CFG['label_values'] explicitly or set cache_on_label_miss='skip'."
                )
                if cache_on_label_miss == "skip":
                    skipped_db[pair_key] = {"x": str(xf), "y": str(yf), "reason": msg}
                    skipped += 1
                    continue
                raise ValueError(msg)
            y_map = label_lut[y3]
            if bool((y_map < 0).any()):
                msg = (
                    "Found labels not present in LUT during cache build. "
                    "Set DATA_CFG['label_values'] explicitly or set cache_on_label_miss='skip'."
                )
                if cache_on_label_miss == "skip":
                    skipped_db[pair_key] = {"x": str(xf), "y": str(yf), "reason": msg}
                    skipped += 1
                    continue
                raise ValueError(msg)
            y3 = y_map

        if cache_zscore_x:
            x3 = (x3 - x3.mean()) / x3.std().clamp_min(1e-6)

        x3 = x3.to(cache_x_dtype).contiguous()
        y3 = y3.to(cache_y_dtype).contiguous()

        _save_tensor_atomic(x3, x_cache)
        _save_tensor_atomic(y3, y_cache)
        out_x.append(str(x_cache))
        out_y.append(str(y_cache))
        skipped_db.pop(pair_key, None)
        built += 1

    skipped_db_path.write_text(json.dumps(skipped_db, indent=1))
    return out_x, out_y, built, reused, skipped

## TODO: refactor to data.py
class LazyTensorPairDataset(Dataset):
    # Strictly on-the-fly: no full-dataset loads during __init__.

    def __init__(self, x_files, y_files, zscore_x=True, max_open_files=1, label_lut=None):
        if len(x_files) != len(y_files):
            raise ValueError("x_files and y_files must have same length")

        self.x_files = [str(Path(p)) for p in x_files]
        self.y_files = [str(Path(p)) for p in y_files]
        self.zscore_x = bool(zscore_x)
        self.max_open_files = max(1, int(max_open_files))
        self.length = len(self.x_files)
        if self.length == 0:
            raise ValueError("Dataset is empty")

        self.label_lut = label_lut
        self._cache = {}
        self._cache_order = []

    def __len__(self):
        return self.length

    def _open_pair(self, file_idx):
        if file_idx in self._cache:
            return self._cache[file_idx]

        while len(self._cache_order) >= self.max_open_files:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)

        xf = self.x_files[file_idx]
        yf = self.y_files[file_idx]
        x = load_tensor_pt(xf)
        y = load_tensor_pt(yf)

        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Sample count mismatch: {xf} has {x.shape[0]}, {yf} has {y.shape[0]}")
        if tuple(x.shape[1:]) != tuple(y.shape[1:]):
            raise ValueError(f"Spatial mismatch: {xf} {tuple(x.shape[1:])} vs {yf} {tuple(y.shape[1:])}")
        if y.dtype.is_floating_point:
            raise TypeError(f"{yf}: labels must be integer class IDs, got float dtype {y.dtype}")

        if x.shape[0] != 1:
            raise ValueError(
                f"{xf}: expected one sample per file ([D,H,W] or [1,D,H,W]), got {tuple(x.shape)}. "
                "Split batched tensors into one sample per file to keep loading fully on-the-fly."
            )

        x3 = x[0]
        y3 = y[0]
        self._cache[file_idx] = (x3, y3)
        self._cache_order.append(file_idx)
        return x3, y3

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.length

        x, y = self._open_pair(int(idx))
        x = x.to(torch.float32)
        y = y.to(torch.int64)

        if self.label_lut is not None:
            y_max = int(y.max().item())
            if y_max >= self.label_lut.numel():
                raise ValueError(
                    f"Label id {y_max} exceeds current LUT range {self.label_lut.numel()-1}. "
                    "Increase DATA_CFG['label_scan_max_files'] or set DATA_CFG['label_values'] explicitly."
                )
            y = self.label_lut[y]
            if bool((y < 0).any()):
                raise ValueError(
                    "Found labels not present in LUT. "
                    "Increase DATA_CFG['label_scan_max_files'] or set DATA_CFG['label_values'] explicitly."
                )

        if self.zscore_x:
            x = (x - x.mean()) / x.std().clamp_min(1e-6)

        return x.contiguous(), y.contiguous()

## TODO: refactor to data.py
def collate_pad_to_patch(batch, patch_size=PATCH_SIZE):
    xs, ys = zip(*batch)
    max_d = max(int(x.shape[0]) for x in xs)
    max_h = max(int(x.shape[1]) for x in xs)
    max_w = max(int(x.shape[2]) for x in xs)

    tgt_d = ((max_d + patch_size[0] - 1) // patch_size[0]) * patch_size[0]
    tgt_h = ((max_h + patch_size[1] - 1) // patch_size[1]) * patch_size[1]
    tgt_w = ((max_w + patch_size[2] - 1) // patch_size[2]) * patch_size[2]

    x_out, y_out = [], []
    for x, y in zip(xs, ys):
        d_pad = tgt_d - int(x.shape[0])
        h_pad = tgt_h - int(x.shape[1])
        w_pad = tgt_w - int(x.shape[2])
        pads = (0, w_pad, 0, h_pad, 0, d_pad)
        x_out.append(F.pad(x, pads, mode="constant", value=0.0))
        y_out.append(F.pad(y, pads, mode="constant", value=0))

    return torch.stack(x_out, dim=0), torch.stack(y_out, dim=0)

## TODO: refactor to data.py
def limit_dataset(ds, max_samples, seed=SEED):
    if max_samples is None or int(max_samples) >= len(ds):
        return ds
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[: int(max_samples)].tolist()
    return Subset(ds, idx)

## TODO: refactor to data.py
def make_loader(ds, batch_size, shuffle, num_workers=TRAIN_CFG["num_workers"], pin_memory=TRAIN_CFG["pin_memory"], prefetch_factor=TRAIN_CFG["prefetch_factor"]):
    num_workers = int(num_workers)
    pin_memory = bool(pin_memory and DEVICE == "cuda")
    kwargs = {
        "dataset": ds,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "collate_fn": collate_pad_to_patch,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**kwargs)

## TODO: refactor to data.py
def build_dataset(x_files, y_files, max_open_files, zscore_x, label_lut):
    return LazyTensorPairDataset(
        x_files,
        y_files,
        zscore_x=zscore_x,
        max_open_files=max_open_files,
        label_lut=label_lut,
    )

## TODO: refactor to train.py
def batch_loss_on_windows(xb, yb, patch_chunk_size, requested_window, requested_stride):
    # xb: [B, D, H, W], yb: [B, D, H, W]
    x5 = xb.unsqueeze(1).float()  # [B,1,D,H,W]
    spatial_shape = tuple(int(v) for v in x5.shape[2:])
    ensure_divisible(spatial_shape, PATCH_SIZE)

    win = fit_window_to_shape(spatial_shape, requested_window, PATCH_SIZE)
    stride_in = requested_stride if requested_stride is not None else win
    stride = fit_window_to_shape(spatial_shape, stride_in, PATCH_SIZE)
    stride = tuple(min(s, w) for s, w in zip(stride, win))

    slices = make_window_slices(spatial_shape, win, stride)
    total = 0.0
    for zs, ys, xs in slices:
        xw = x5[:, :, zs, ys, xs]
        yw = yb[:, zs, ys, xs]
        feat = model.forward_features(xw)
        loss_w = model.loss_from_features(feat, yw, patch_chunk_size=patch_chunk_size)
        total = total + loss_w
    return total / max(1, len(slices))

## TODO: refactor to train.py
def train_step_adaptive(x_cpu, y_cpu):
    # Returns: (loss_num, loss_den, hit_oom)
    global runtime_micro_bs, runtime_patch_chunk, runtime_window
    hit_oom = False

    total_voxels = int(y_cpu.numel())
    while True:
        optimizer.zero_grad(set_to_none=True)
        batch_num = 0.0

        try:
            for s in range(0, x_cpu.size(0), runtime_micro_bs):
                xb = x_cpu[s : s + runtime_micro_bs].to(DEVICE, non_blocking=True)
                yb = y_cpu[s : s + runtime_micro_bs].to(DEVICE, non_blocking=True)

                weight = float(yb.numel()) / float(total_voxels)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    loss_micro = batch_loss_on_windows(
                        xb,
                        yb,
                        patch_chunk_size=runtime_patch_chunk,
                        requested_window=runtime_window,
                        requested_stride=runtime_stride,
                    )
                    loss_scaled = loss_micro * weight

                if scaler.is_enabled():
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

                batch_num += float(loss_micro.detach().item()) * int(yb.numel())

            clip_norm = float(TRAIN_CFG["grad_clip_norm"])
            if clip_norm > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            return batch_num, total_voxels, hit_oom

        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise

            hit_oom = True
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            if runtime_micro_bs > 1:
                runtime_micro_bs = max(1, runtime_micro_bs // 2)
                print(f"[OOM] micro_batch_size -> {runtime_micro_bs}")
            elif runtime_patch_chunk > 16:
                runtime_patch_chunk = max(16, runtime_patch_chunk // 2)
                print(f"[OOM] patch_chunk_size -> {runtime_patch_chunk}")
            else:
                if runtime_window is None:
                    runtime_window = fit_window_to_shape(tuple(int(v) for v in x_cpu.shape[1:]), x_cpu.shape[1:], PATCH_SIZE)
                new_window = halve_window_shape(runtime_window, PATCH_SIZE)
                if new_window == runtime_window:
                    raise RuntimeError("OOM at minimum micro-batch/patch-chunk/window settings") from e
                runtime_window = new_window
                print(f"[OOM] spatial_window -> {runtime_window}")

## TODO: refactor to train.py
@torch.no_grad()
def evaluate(model, loader, max_batches=None, desc="val"):
    model.eval()
    total_num = 0.0
    total_den = 0

    limit = len(loader) if max_batches is None else min(len(loader), int(max_batches))
    pbar = tqdm(loader, total=limit, desc=desc, leave=False)

    for b_idx, (x_cpu, y_cpu) in enumerate(pbar):
        if max_batches is not None and b_idx >= int(max_batches):
            break

        batch_num = 0.0
        for s in range(0, x_cpu.size(0), runtime_micro_bs):
            xb = x_cpu[s : s + runtime_micro_bs].to(DEVICE, non_blocking=True)
            yb = y_cpu[s : s + runtime_micro_bs].to(DEVICE, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                loss_micro = batch_loss_on_windows(
                    xb,
                    yb,
                    patch_chunk_size=runtime_patch_chunk,
                    requested_window=runtime_window,
                    requested_stride=runtime_stride,
                )
            batch_num += float(loss_micro.item()) * int(yb.numel())

        total_num += batch_num
        total_den += int(y_cpu.numel())

        alloc_gb, peak_gb = gpu_mem_gb()
        pbar.set_postfix(
            loss=f"{(total_num / max(1, total_den)):.4f}",
            mb=runtime_micro_bs,
            pc=runtime_patch_chunk,
            vram=f"{alloc_gb:.1f}/{peak_gb:.1f}G",
        )

    pbar.close()
    model.train()
    return total_num / max(1, total_den)
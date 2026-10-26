# %% [md]
# # 010 — Data Audit NWPU-RESISC45
#
# **Mục tiêu:** audit đầy đủ dataset NWPU-RESISC45 gắn vào kernel và sinh **split manifest cố định** dùng chung cho mọi model của project.
#
# **Nguồn data:** Kaggle dataset `punfake/resisc45` (đã xác minh 31.500 ảnh = 45 lớp × 700; cấu trúc `NWPU-RESISC45/<class>/<class>_NNN.jpg`).
#
# **Nhiệm vụ:**
# 1. Tìm root dataset (fail-fast), đếm ảnh/lớp.
# 2. Kiểm tra ảnh decode lỗi + kích thước thật.
# 3. Exact duplicate (MD5), near-duplicate (dHash, ngưỡng Hamming ≤ 6) — kể cả trùng khác nhãn.
# 4. Gom nhóm trùng (union-find) → chia split 75/10/15 stratified, **mỗi nhóm ở đúng 1 split** (chống leakage).
# 5. Xuất artifacts: `audit_report.json`, `split_manifest.csv`, `class_order.json`, `sha256sums.txt`.
#
# **Lưu ý:** chạy ở chế độ SMOKE (local) chỉ dùng subset để kiểm logic; artifact khi đó có hậu tố `.smoke.*` — **không** dùng cho thí nghiệm thật.

# %%
# ---- CONFIG (tập trung) ----
import os

RUN_ID = "010-data-audit-v1"
DATASET_SLUG = "punfake/resisc45"
SPLIT_SEED = 42                 # seed tạo split — cố định toàn project
SPLIT_RATIOS = (0.75, 0.10, 0.15)
NEARDUP_HAMMING_MAX = 6         # ngưỡng dHash 64-bit; ứng viên được sample kiểm tra thủ công
NEARDUP_MAX_PAIRS_OUT = 60      # số cặp ứng viên xuất sample để review
# SMOKE: bật khi chạy local (nếu không có /kaggle) — chỉ test logic
SMOKE = os.environ.get("AUDIT_SMOKE", "0" if os.path.isdir("/kaggle") else "1") == "1"
SMOKE_CLASSES = 5
SMOKE_PER_CLASS = 60
SUFFIX = ".smoke" if SMOKE else ""

# %%
# ---- ENV + FAIL-FAST: tìm root dataset ----
import json
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image

t_start = time.time()
print(f"run_id: {RUN_ID} | smoke: {SMOKE}")
print("python:", sys.version.split()[0], "| pandas:", pd.__version__, "| numpy:", np.__version__)
print("PIL:", Image.__version__ if hasattr(Image, "__version__") else "PIL")

CANDIDATES = [
    "/kaggle/input/resisc45/NWPU-RESISC45",
    "/kaggle/input/nwpu-resisc45/NWPU-RESISC45",
    "data/raw/punfake---resisc45/NWPU-RESISC45",              # local, cwd = project root
    "../../data/raw/punfake---resisc45/NWPU-RESISC45",        # local, cwd = notebook dir (jupyter execute)
]
DATA_ROOT = None
for c in CANDIDATES:
    if os.path.isdir(c):
        classes = sorted(d for d in os.listdir(c) if os.path.isdir(os.path.join(c, d)))
        if len(classes) >= 40:
            DATA_ROOT = c
            break
if DATA_ROOT is None:
    # fallback: quét /kaggle/input sâu 2 tầng
    for base in ("/kaggle/input", "."):
        if not os.path.isdir(base):
            continue
        for l1 in sorted(os.listdir(base)):
            p1 = os.path.join(base, l1)
            if not os.path.isdir(p1):
                continue
            for l2 in sorted(os.listdir(p1)):
                p2 = os.path.join(p1, l2)
                if os.path.isdir(p2) and len([d for d in os.listdir(p2) if os.path.isdir(os.path.join(p2, d))]) >= 40:
                    DATA_ROOT = p2
                    break
            if DATA_ROOT:
                break
        if DATA_ROOT:
            break
assert DATA_ROOT, "FAIL-FAST: khong tim thay dataset root co >=40 class dirs"
CLASSES = sorted(d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d)))
assert len(CLASSES) == 45, f"FAIL-FAST: expect 45 classes, got {len(CLASSES)}"
CLASS_ORDER = {c: i for i, c in enumerate(CLASSES)}
print("DATA_ROOT:", DATA_ROOT)
print("45 classes OK | ví dụ:", CLASSES[:3])

# %%
# ---- 1) File manifest: đếm ảnh/lớp ----
rows = []
for c in CLASSES:
    cdir = os.path.join(DATA_ROOT, c)
    for f in sorted(os.listdir(cdir)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            rows.append({"image_id": f"{c}/{f}", "relpath": f"NWPU-RESISC45/{c}/{f}", "class_name": c, "class_index": CLASS_ORDER[c]})
df = pd.DataFrame(rows)
per_class = df.groupby("class_name").size()
print("tổng ảnh:", len(df), "| lớp:", per_class.size, "| min/max ảnh mỗi lớp:", int(per_class.min()), int(per_class.max()))
assert (per_class == 700).all() or SMOKE, "FAIL-FAST: so anh/lop khac 700 — xem per_class"
if SMOKE:
    keep = df["class_name"].isin(CLASSES[:SMOKE_CLASSES])
    df = df[keep].copy()
    df = df.groupby("class_name", group_keys=False).head(SMOKE_PER_CLASS).reset_index(drop=True)
    print(f"SMOKE: giữ {len(df)} ảnh / {df.class_name.nunique()} lớp")
df = df.reset_index(drop=True)
df.head(3)

# %%
# ---- 2) Decode + kích thước ----
sizes, bad = [], []
for r in df.itertuples():
    try:
        with Image.open(os.path.join(DATA_ROOT, r.class_name, os.path.basename(r.image_id))) as im:
            im.verify()  # verify header nhanh
            sizes.append(im.size)
    except Exception as e:  # noqa: BLE001 — audit phải thu mọi lỗi decode
        bad.append({"image_id": r.image_id, "error": repr(e)})
size_arr = np.array([s for s in sizes if s])
sizes_df = pd.DataFrame(size_arr, columns=["w", "h"])
size_stats = {
    "decoded_ok": int(len(sizes)),
    "decode_errors": len(bad),
    "w_min": int(sizes_df.w.min()), "w_max": int(sizes_df.w.max()),
    "h_min": int(sizes_df.h.min()), "h_max": int(sizes_df.h.max()),
    "n_non_256x256": int(((sizes_df.w != 256) | (sizes_df.h != 256)).sum()),
}
print(size_stats)
print("bad images:", bad[:10])
df = df.iloc[[i for i, s in enumerate(sizes) if s is not None]].reset_index(drop=True)

# %%
# ---- 3a) Exact duplicate theo MD5 ----
import hashlib

md5s = []
for r in df.itertuples():
    with open(os.path.join(DATA_ROOT, r.class_name, os.path.basename(r.image_id)), "rb") as f:
        md5s.append(hashlib.md5(f.read()).hexdigest())
df["md5"] = md5s
dup_md5 = df[df.duplicated("md5", keep=False)].sort_values("md5")
n_dup_files = int(len(dup_md5))
n_dup_groups_md5 = int(dup_md5.md5.nunique()) if n_dup_files else 0
cross_class_md5 = int(dup_md5.groupby("md5")["class_name"].nunique().gt(1).sum()) if n_dup_files else 0
print(f"exact-dup: {n_dup_files} file / {n_dup_groups_md5} nhóm / {cross_class_md5} nhóm đa lớp")
dup_md5.head(6)

# %%
# ---- 3b) Near-duplicate bằng dHash (vector hoá) ----
def dhash_batch(paths: list[str], hash_size: int = 8) -> np.ndarray:
    out = np.zeros((len(paths), hash_size * hash_size), dtype=bool)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            g = im.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
        a = np.asarray(g, dtype=np.int16)
        out[i] = (a[:, 1:] > a[:, :-1]).reshape(-1)
    return out

t0 = time.time()
paths = [os.path.join(DATA_ROOT, r.class_name, os.path.basename(r.image_id)) for r in df.itertuples()]
hashes = dhash_batch(paths)
print(f"dHash xong {len(hashes)} ảnh trong {time.time()-t0:.1f}s")

# pairwise Hamming: XOR trên bitpack + popcount LUT (unpackbits sẽ tốn ~8GB ở full 31.500 ảnh)
t0 = time.time()
n = len(hashes)
pairs: list[tuple[int, int, int]] = []  # (i, j, dist)
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
CH = 512
packed = np.packbits(hashes, axis=1)  # (n, 8) uint8
for s in range(0, n, CH):
    e = min(s + CH, n)
    xor = np.bitwise_xor(packed[s:e, None, :], packed[None, :, :])
    dist = POPCOUNT[xor].sum(axis=2).astype(np.int16)
    ii, jj = np.where((dist <= NEARDUP_HAMMING_MAX) & (dist > 0))
    for i, j in zip(ii, jj):  # noqa: B905 — chỉ giữ nửa trên (i<j) tránh double-count
        if s + i < j:
            pairs.append((int(s + i), int(j), int(dist[i, j])))
print(f"pairwise xong trong {time.time()-t0:.1f}s | {len(pairs)} cặp ứng viên (hamming<= {NEARDUP_HAMMING_MAX})")
pairs_df = pd.DataFrame(pairs, columns=["i", "j", "hamming"])
pairs_df["cls_i"] = df.class_index.values[pairs_df.i]
pairs_df["cls_j"] = df.class_index.values[pairs_df.j]
pairs_df["same_class"] = pairs_df.cls_i == pairs_df.cls_j
print("cặp cùng lớp:", int(pairs_df.same_class.sum()), "| khác lớp:", int((~pairs_df.same_class).sum()))

# %%
# ---- 3c) Sample ứng viên near-dup để kiểm tra thủ công (grid PNG + CSV) ----
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def grid_pairs(pairs_subset: pd.DataFrame, fname: str, k: int = 12):
    k = min(k, len(pairs_subset))
    if k == 0:
        return
    fig, axes = plt.subplots(k, 2, figsize=(4, 2 * k))
    if k == 1:
        axes = axes[None, :]
    for row, (_, p) in enumerate(pairs_subset.head(k).iterrows()):
        for col, idx in enumerate((p.i, p.j)):
            with Image.open(paths[idx]) as im:
                axes[row, col].imshow(im)
            axes[row, col].set_title(f"{df.image_id[idx]} (h={p.hamming})", fontsize=7)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(fname, dpi=90)
    plt.close(fig)

if len(pairs_df):
    grid_pairs(pairs_df[~pairs_df.same_class].head(20), f"near_dup_cross_class{SUFFIX}.png")
    grid_pairs(pairs_df[pairs_df.same_class].head(20), f"near_dup_same_class{SUFFIX}.png")
    pairs_df_out = pairs_df.copy()
    pairs_df_out["image_i"] = df.image_id.values[pairs_df_out.i]
    pairs_df_out["image_j"] = df.image_id.values[pairs_df_out.j]
    pairs_df_out.to_csv(f"near_dup_pairs{SUFFIX}.csv", index=False)
print("Đã xuất sample cặp near-dup (PNG + CSV) — CẦN KIỂM TRA THỦ CÔNG trước khi chốt groups")

# %%
# ---- 4) Union-find gom nhóm trùng (exact + near-dup) ----
parent = list(range(len(df)))

def find(x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
        if parent[x] != parent[parent[x]]:
            x = parent[x]
    return x

def union(a: int, b: int) -> None:
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)

for md5, g in df.groupby("md5").groups.items():  # exact dup
    idxs = list(g)
    for x in idxs[1:]:
        union(idxs[0], x)
for p in pairs_df.itertuples():  # near-dup ứng viên
    union(int(p.i), int(p.j))

df["dup_group"] = [find(i) for i in range(len(df))]
grp_sizes = df.groupby("dup_group").size()
grp_classes = df.groupby("dup_group")["class_name"].nunique()
print(f"nhóm dup: {int((grp_sizes > 1).sum())} nhóm >1 ảnh | nhóm lớn nhất: {int(grp_sizes.max())} ảnh | nhóm đa lớp: {int(grp_classes.gt(1).sum())}")

# %%
# ---- 5) Split 75/10/15 group-aware, stratified theo lớp ----
rng = np.random.default_rng(SPLIT_SEED)
unit = df[["image_id", "class_name", "class_index", "dup_group"]].copy()
unit["grp_size"] = unit.groupby("dup_group")["dup_group"].transform("size")

# cảnh báo: nhóm quá lớn so với 10% lớp → không thể giữ nguyên mà vẫn tách val/test
oversize = unit[unit.grp_size > 0.10 * unit.groupby("class_name")["image_id"].transform("count")]
if len(oversize):
    print("CẢNH BÁO nhóm oversize (giữ nguyên, chấp nhận lệch tỉ lệ):")
    print(oversize.drop_duplicates("dup_group")[["class_name", "grp_size"]])

split_of: dict[int, str] = {}
counts = {}
for cls, gdf in unit.groupby("class_name"):
    grps = sorted(gdf.dup_group.unique())
    rng.shuffle(grps)
    total = len(gdf)
    n_train, n_val = int(round(total * SPLIT_RATIOS[0])), int(round(total * SPLIT_RATIOS[1]))
    c_train = c_val = c_test = 0
    for g in grps:
        gs = int(unit[unit.dup_group == g].grp_size.iloc[0])
        # gán vào split còn thiếu nhiều nhất (theo tỉ lệ đã normalize về số ảnh)
        need = [(n_train - c_train, "train"), (n_val - c_val, "val"), (total * SPLIT_RATIOS[2] - c_test, "test")]
        need.sort(key=lambda t: -t[0])
        split_of[g] = need[0][1]
        if need[0][1] == "train":
            c_train += gs
        elif need[0][1] == "val":
            c_val += gs
        else:
            c_test += gs
    counts[cls] = (c_train, c_val, c_test)

unit["split"] = unit.dup_group.map(split_of)
counts_df = pd.DataFrame(counts, index=["train", "val", "test"]).T
print("số ảnh mỗi lớp × split (mẫu 5 lớp đầu):")
print(counts_df.head())
print("tổng:", counts_df.sum().to_dict())

# KIỂM TRA leakage: không dup_group nào xuất hiện ở >1 split
leak = unit.groupby("dup_group")["split"].nunique()
assert (leak == 1).all(), "FAIL-FAST: dup group trải nhiều split"
print("leakage check: PASS — mọi dup group ở đúng 1 split")

# %%
# ---- 6) Xuất artifacts ----
out_manifest = unit[["image_id", "class_name", "class_index", "dup_group", "split"]].sort_values("image_id")
out_manifest.to_csv(f"split_manifest{SUFFIX}.csv", index=False)
with open(f"class_order{SUFFIX}.json", "w") as f:
    json.dump(CLASS_ORDER, f, indent=1)

split_hash = hashlib.sha256(out_manifest.to_csv(index=False).encode()).hexdigest()

audit_report = {
    "run_id": RUN_ID, "smoke": SMOKE, "dataset": DATASET_SLUG,
    "data_root_seen": DATA_ROOT, "n_images": int(len(df)), "n_classes": len(CLASSES),
    "sizes": size_stats, "decode_errors": bad,
    "exact_dup_files": n_dup_files, "exact_dup_groups": n_dup_groups_md5, "exact_dup_cross_class_groups": cross_class_md5,
    "neardup_hamming_max": NEARDUP_HAMMING_MAX, "neardup_candidate_pairs": int(len(pairs_df)),
    "neardup_same_class_pairs": int(pairs_df.same_class.sum()) if len(pairs_df) else 0,
    "neardup_cross_class_pairs": int((~pairs_df.same_class).sum()) if len(pairs_df) else 0,
    "dup_groups_total": int(df.dup_group.nunique()),
    "split_seed": SPLIT_SEED, "split_sha256": split_hash,
    "split_counts": {k: int(v) for k, v in counts_df.sum().items()},
    "per_class_split_min": {s: int(counts_df[s].min()) for s in ("train", "val", "test")},
    "elapsed_sec": round(time.time() - t_start, 1),
}
with open(f"audit_report{SUFFIX}.json", "w") as f:
    json.dump(audit_report, f, indent=1, ensure_ascii=False)
print(json.dumps(audit_report, indent=1, ensure_ascii=False))

# %%
# ---- 7) SUMMARY + fail-fast tổng ----
print("=== AUDIT SUMMARY ===")
print(f"images: {len(df)} | classes: {len(CLASSES)} | decode errors: {len(bad)}")
print(f"exact dup: {n_dup_files} files | near-dup candidates: {len(pairs_df)} (cần review PNG/CSV)")
print(f"split counts: {counts_df.sum().to_dict()} | manifest sha256: {split_hash[:16]}...")
if not SMOKE:
    assert len(bad) == 0, f"Có {len(bad)} ảnh decode lỗi — xử lý trước khi tiếp tục"
    assert counts_df.sum().sum() == 31500 or int(len(df)) < 31500, "tổng ảnh split != 31500"
    print("FULL AUDIT PASS — split_manifest.csv dùng chung cho mọi notebook sau (attach output kernel này)")
else:
    print("SMOKE PASS — artifacts có hậu tố .smoke, KHÔNG dùng cho thí nghiệm")
print(f"total elapsed: {time.time()-t_start:.1f}s")

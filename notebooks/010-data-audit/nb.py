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
# 3. Trùng lặp, phân biệt 3 mức:
#    - **exact duplicate**: MD5 giống hệt → gộp nhóm tự động (đã xác minh).
#    - **near-duplicate candidate**: cặp dHash Hamming ≤ 6 — CHỈ BÁO CÁO (CSV + PNG) để review.
#    - **merged near-dup group**: chỉ gộp khi 2 ảnh là **mutual nearest neighbor** (không nối chuỗi A~B~C);
#      nhóm ≥ 3 ảnh do near-dup tạo ra bị đánh dấu `needs_manual_verification`.
# 4. Gom nhóm (union-find) → split 75/10/15 stratified, **mỗi nhóm ở đúng 1 split** (chống leakage).
# 5. Xuất artifacts: `audit_report.json`, `split_manifest.csv`, `class_order.json`.
#
# **Bộ nhớ pairwise:** xử lý theo block CH×N (KHÔNG materialize N×N). Ở N=31.500, CH=512:
# xor `(512, 31500, 8)` uint8 ≈ 123MB + popcount block ≈ 123MB + dist `(512, 31500)` int16 ≈ 32MB
# → peak < 400MB/block, độc lập kích thước N theo chiều block.
#
# **Lưu ý:** chạy ở chế độ SMOKE (local) chỉ dùng subset để kiểm logic; artifact khi đó có hậu tố `.smoke.*` — **không** dùng cho thí nghiệm thật.

# %%
# ---- CONFIG (tập trung) ----
import os

RUN_ID = "010-data-audit-v1"
DATASET_SLUG = "punfake/resisc45"
SPLIT_SEED = 42                 # seed tạo split — cố định toàn project
SPLIT_RATIOS = (0.75, 0.10, 0.15)
NEARDUP_HAMMING_MAX = 6         # ngưỡng dHash 64-bit cho CANDIDATE (chỉ báo cáo)
NEARDUP_MERGE_MANUAL_MIN = 3    # nhóm near-dup ≥ N ảnh → cờ needs_manual_verification
PW_BLOCK = 512                  # block pairwise Hamming (chiều block không phụ thuộc N)
SMOKE = os.environ.get("AUDIT_SMOKE", "0" if os.path.isdir("/kaggle/input") else "1") == "1"
SMOKE_CLASSES = 5
SMOKE_PER_CLASS = 60
SUFFIX = ".smoke" if SMOKE else ""
# chống nhầm cấu hình: trên Kaggle (/kaggle/input tồn tại) thì KHÔNG được bật SMOKE
assert not (os.path.isdir("/kaggle/input") and SMOKE), "FAIL-FAST: đang trên Kaggle nhưng SMOKE=True"

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

import glob as _glob

def _is_data_root(p: str) -> bool:
    try:
        return os.path.isdir(p) and len([d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]) >= 40
    except OSError:
        return False

# Kaggle mount layout có thể thay đổi (2026-10: /kaggle/input/datasets/<owner>/<slug>) —
# dùng glob recursive theo tên thư mục chuẩn của dataset, không phụ thuộc tầng mount.
DATA_ROOT = None
for hit in _glob.glob("/kaggle/input/**/NWPU-RESISC45", recursive=True) + [
        "data/raw/punfake---resisc45/NWPU-RESISC45",              # local, cwd = project root
        "../../data/raw/punfake---resisc45/NWPU-RESISC45"]:       # local, cwd = notebook dir
    if _is_data_root(hit):
        DATA_ROOT = hit
        break
if DATA_ROOT is None and os.path.isdir("/kaggle/input"):
    listing = os.listdir("/kaggle/input")
    print("DIAGNOSTIC /kaggle/input:", listing)
    for l1 in listing:
        p1 = os.path.join("/kaggle/input", l1)
        if os.path.isdir(p1):
            print("  ", l1, "→", os.listdir(p1)[:8])
assert DATA_ROOT, f"FAIL-FAST: khong tim thay dataset root; /kaggle/input = {os.listdir('/kaggle/input') if os.path.isdir('/kaggle/input') else 'KHONG TON TAI'}"
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
# ---- 3a) Exact duplicate theo MD5 (đã xác minh — gộp tự động) ----
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
# ---- 3b) Near-duplicate: dHash + pairwise Hamming theo block + mutual-NN ----
def dhash_batch(paths_: list[str], hash_size: int = 8) -> np.ndarray:
    out = np.zeros((len(paths_), hash_size * hash_size), dtype=bool)
    for i, p in enumerate(paths_):
        with Image.open(p) as im:
            g = im.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
        a = np.asarray(g, dtype=np.int16)
        out[i] = (a[:, 1:] > a[:, :-1]).reshape(-1)
    return out

POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
SELF_DIST = np.int16(999)


def pairwise_hamming_block(packed: np.ndarray, block: int = PW_BLOCK) -> np.ndarray:
    """Generator khoảng cách Hamming theo block (B, N) — KHÔNG materialize N×N.
    Peak bộ nhớ: xor (B, N, 8) uint8 + popcount block uint8 + dist (B, N) int16."""
    n = len(packed)
    for s in range(0, n, block):
        e = min(s + block, n)
        xor = np.bitwise_xor(packed[s:e, None, :], packed[None, :, :])
        dist = POPCOUNT[xor].sum(axis=2).astype(np.int16)
        dist[np.arange(e - s), np.arange(s, e)] = SELF_DIST  # loại khoảng cách với chính nó
        yield dist


def mutual_near_pairs(packed: np.ndarray, threshold: int) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """MỘT pass block (B, N): gom candidate (mọi cặp ≤ threshold, chỉ lưu list nhỏ) + NN toàn cục
    best_j của từng ảnh. Mutual = (i,j) ≤ threshold ∧ best_j[i]==j ∧ best_j[j]==i — chỉ mức này
    dùng để GỘP nhóm, tránh nối chuỗi A~B~C thành nhóm lớn không hợp lý.
    Trả về (candidates_df, mutual_df, best_j)."""
    n = len(packed)
    best_j = np.full(n, -1, dtype=np.int64)
    best_d = np.full(n, SELF_DIST, dtype=np.int16)
    recs: list[tuple[int, int, int]] = []
    for s, dist in enumerate(pairwise_hamming_block(packed)):
        e = s + len(dist)
        rows = dist.argmin(axis=1)
        rows_d = dist[np.arange(e - s), rows]
        bj = best_j[s:e]
        upd = rows_d < best_d[s:e]
        bj[upd] = rows[upd]  # rows là chỉ số cột toàn cục (dist có N cột)
        best_d[s:e] = np.where(upd, rows_d, best_d[s:e])
        ii, jj = np.where(dist <= threshold)
        for a, b in zip(ii, jj):  # noqa: B905
            i, j = int(s + a), int(b)
            if i < j:
                recs.append((i, j, int(dist[a, b])))
    cand = pd.DataFrame(recs, columns=["i", "j", "hamming"])
    mut = [(i, j, d) for i, j, d in cand.itertuples(index=False) if best_j[i] == j and best_j[j] == i]
    return cand, pd.DataFrame(mut, columns=["i", "j", "hamming"]), best_j

# self-test trên dữ liệu tổng hợp TRƯỚC khi tin kết quả thật: 2 cặp mutual + 1 cặp không mutual
_rng_t = np.random.default_rng(0)
_base = _rng_t.random(64) < 0.5
_ht = np.zeros((5, 64), dtype=bool)
_ht[0] = _base
_ht[1] = _base.copy()
_ht[1][7] = ~_ht[1][7]                                     # d(0,1)=1 → mutual
_ht[2] = _rng_t.random(64) < 0.5
_ht[3] = _ht[2].copy()
_ht[3][3] = ~_ht[3][3]
_ht[3][40] = ~_ht[3][40]                                   # d(2,3)=2 → mutual
_ht[4] = _ht[1].copy()
for _b in (0, 1, 2):
    _ht[4][_b] = ~_ht[4][_b]                               # d(1,4)=3 → candidate nhưng NN(1)=0 → KHÔNG mutual
_packed_test = np.packbits(_ht, axis=1)
_cand_t, _mut_t, _bj_t = mutual_near_pairs(_packed_test, threshold=3)
assert sorted(map(tuple, _cand_t[["i", "j"]].values)) == [(0, 1), (1, 4), (2, 3)], f"test candidate fail:\n{_cand_t}"
assert sorted(map(tuple, _mut_t[["i", "j"]].values)) == [(0, 1), (2, 3)], f"test mutual fail:\n{_mut_t}"
assert _bj_t[4] == 1 and _bj_t[1] == 0, f"test NN fail: {_bj_t}"
print("self-test mutual_near_pairs: PASS (mutual đúng 2 cặp; cặp (1,4) chỉ là candidate, không gộp)")

t0 = time.time()
paths = [os.path.join(DATA_ROOT, r.class_name, os.path.basename(r.image_id)) for r in df.itertuples()]
hashes = dhash_batch(paths)
packed = np.packbits(hashes, axis=1)
print(f"dHash xong {len(hashes)} ảnh trong {time.time()-t0:.1f}s | packed: {packed.shape} {packed.dtype}")

t0 = time.time()
pairs_df, mut_df, best_j = mutual_near_pairs(packed, NEARDUP_HAMMING_MAX)
print(f"pairwise 2 pass xong trong {time.time()-t0:.1f}s | candidates: {len(pairs_df)} | mutual-NN pairs: {len(mut_df)}")

for d_, name in ((pairs_df, "pairs_df"), (mut_df, "mut_df")):
    if len(d_):
        d_["cls_i"] = df.class_index.values[d_.i]
        d_["cls_j"] = df.class_index.values[d_.j]
        d_["same_class"] = d_.cls_i == d_.cls_j
n_cand_same = int(pairs_df.same_class.sum()) if len(pairs_df) else 0
n_cand_cross = int(len(pairs_df) - n_cand_same)
n_mut_same = int(mut_df.same_class.sum()) if len(mut_df) else 0
n_mut_cross = int(len(mut_df) - n_mut_same)
print(f"CANDIDATE ≤{NEARDUP_HAMMING_MAX}: {len(pairs_df)} (cùng lớp {n_cand_same} / khác lớp {n_cand_cross})")
print(f"MUTUAL-NN (sẽ gộp): {len(mut_df)} (cùng lớp {n_mut_same} / khác lớp {n_mut_cross})")

# %%
# ---- 3c) Xuất ứng viên để review thủ công (PNG + CSV) ----
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
        for col, idx in enumerate((int(p.i), int(p.j))):
            with Image.open(paths[idx]) as im:
                axes[row, col].imshow(im)
            axes[row, col].set_title(f"{df.image_id[idx]} (h={int(p.hamming)})", fontsize=7)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(fname, dpi=90)
    plt.close(fig)

if len(pairs_df):
    grid_pairs(pairs_df[~pairs_df.same_class], f"near_dup_cross_class{SUFFIX}.png", 20)
    grid_pairs(pairs_df[pairs_df.same_class], f"near_dup_same_class{SUFFIX}.png", 20)
    out = pairs_df.copy()
    out["image_i"] = df.image_id.values[out.i]
    out["image_j"] = df.image_id.values[out.j]
    out["is_mutual"] = [(best_j[i] == j and best_j[j] == i) for i, j in zip(out.i, out.j)]
    out.to_csv(f"near_dup_pairs{SUFFIX}.csv", index=False)
print("Đã xuất candidate near-dup (PNG + CSV, có cờ is_mutual) — review trước khi tin nhóm")

# %%
# ---- 4) Union-find: exact (tự động) + mutual near-dup (có điều kiện) ----
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

for md5, g in df.groupby("md5").groups.items():  # mức 1: exact — đã xác minh
    idxs = list(g)
    for x in idxs[1:]:
        union(idxs[0], x)
exact_roots = {find(i) for i in range(len(df)) if df.md5.duplicated(keep=False).iloc[i]}
for p in mut_df.itertuples():  # mức 2: mutual NN gần nhất
    union(int(p.i), int(p.j))

df["dup_group"] = [find(i) for i in range(len(df))]
grp_sizes = df.groupby("dup_group").size()
grp_classes = df.groupby("dup_group")["class_name"].nunique()

# dup_kind theo nhóm: single | exact (toàn bộ cùng md5) | near (gộp nhờ mutual-NN) | mixed
def group_kind(g: pd.DataFrame) -> str:
    if len(g) == 1:
        return "single"
    same_md5_mult = g.md5.duplicated(keep=False).any()
    multi_md5 = g.md5.nunique() > 1
    if same_md5_mult and multi_md5:
        return "mixed"
    return "exact" if not multi_md5 else "near"

kinds = {g: group_kind(df.loc[idx]) for g, idx in df.groupby("dup_group").groups.items()}
df["dup_kind"] = df.dup_group.map(kinds)
# nhóm near/mixed có ≥ NEARDUP_MERGE_MANUAL_MIN ảnh → cần xác minh thủ công
flagged = df[df.dup_kind.isin(("near", "mixed"))].groupby("dup_group").size()
flagged_ids = set(flagged[flagged >= NEARDUP_MERGE_MANUAL_MIN].index)
df["needs_manual_verification"] = df.dup_group.isin(flagged_ids)
n_flagged_groups = len(flagged_ids)
print(f"nhóm >1 ảnh: {int((grp_sizes > 1).sum())} | lớn nhất: {int(grp_sizes.max())} ảnh | đa lớp: {int(grp_classes.gt(1).sum())}")
print(f"dup_kind phân bố: {df.drop_duplicates('dup_group').dup_kind.value_counts().to_dict()}")
print(f"nhóm near/mixed cần xác minh (≥{NEARDUP_MERGE_MANUAL_MIN} ảnh): {n_flagged_groups}")

# preview nhóm lớn nhất nếu có
if int(grp_sizes.max()) > 1:
    gmax = grp_sizes.idxmax()
    members = df[df.dup_group == gmax]
    fig, axes = plt.subplots(1, min(len(members), 10), figsize=(2 * min(len(members), 10), 2.4))
    axes = np.atleast_1d(axes)
    for ax, (_, r) in zip(axes, members.head(10).iterrows()):
        with Image.open(os.path.join(DATA_ROOT, r.class_name, os.path.basename(r.image_id))) as im:
            ax.imshow(im)
        ax.set_title(f"{r.image_id}", fontsize=6)
        ax.axis("off")
    fig.suptitle(f"nhóm lớn nhất dup_group={gmax} ({len(members)} ảnh, kind={kinds[gmax]})", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"largest_group{SUFFIX}.png", dpi=90)
    plt.close(fig)

# %%
# ---- 5) Split 75/10/15 group-aware, stratified theo lớp ----
rng = np.random.default_rng(SPLIT_SEED)
unit = df[["image_id", "class_name", "class_index", "dup_group", "dup_kind", "needs_manual_verification"]].copy()
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
        gs = int(unit.loc[unit.dup_group == g, "grp_size"].iloc[0])
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
out_manifest = unit[["image_id", "class_name", "class_index", "dup_group", "dup_kind",
                     "needs_manual_verification", "split"]].sort_values("image_id")
out_manifest.to_csv(f"split_manifest{SUFFIX}.csv", index=False)
with open(f"class_order{SUFFIX}.json", "w") as f:
    json.dump(CLASS_ORDER, f, indent=1)

split_hash = hashlib.sha256(out_manifest.to_csv(index=False).encode()).hexdigest()

audit_report = {
    "run_id": RUN_ID, "smoke": SMOKE, "dataset": DATASET_SLUG,
    "data_root_seen": DATA_ROOT, "n_images": int(len(df)), "n_classes": len(CLASSES),
    "sizes": size_stats, "decode_errors": bad,
    "exact_dup_files": n_dup_files, "exact_dup_groups": n_dup_groups_md5, "exact_dup_cross_class_groups": cross_class_md5,
    "neardup_hamming_max": NEARDUP_HAMMING_MAX,
    "neardup_candidate_pairs": int(len(pairs_df)),
    "neardup_candidate_same_class": n_cand_same, "neardup_candidate_cross_class": n_cand_cross,
    "neardup_mutual_pairs": int(len(mut_df)), "neardup_mutual_same_class": n_mut_same, "neardup_mutual_cross_class": n_mut_cross,
    "merge_policy": "exact md5 (auto) + mutual-NN hamming<=%d; candidate chi report" % NEARDUP_HAMMING_MAX,
    "groups_needing_manual_verification": n_flagged_groups,
    "dup_groups_total": int(df.dup_group.nunique()),
    "largest_group_size": int(grp_sizes.max()),
    "largest_group_kind": str(kinds[grp_sizes.idxmax()]),
    "cross_class_groups": int(grp_classes.gt(1).sum()),
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
print(f"exact dup: {n_dup_files} files | near-dup candidates: {len(pairs_df)} | mutual (gộp): {len(mut_df)}")
print(f"groups needing manual verification: {n_flagged_groups} | largest group: {int(grp_sizes.max())} ({kinds[grp_sizes.idxmax()]})")
print(f"split counts: {counts_df.sum().to_dict()} | manifest sha256: {split_hash[:16]}...")
if not SMOKE:
    assert len(bad) == 0, f"Có {len(bad)} ảnh decode lỗi — xử lý trước khi tiếp tục"
    assert int(counts_df.sum().sum()) == len(df) == 31500, "tổng ảnh split != 31500"
    print("FULL AUDIT PASS — split_manifest.csv dùng chung cho mọi notebook sau (attach output kernel này)")
else:
    print("SMOKE PASS — artifacts có hậu tố .smoke, KHÔNG dùng cho thí nghiệm")
print(f"total elapsed: {time.time()-t_start:.1f}s")

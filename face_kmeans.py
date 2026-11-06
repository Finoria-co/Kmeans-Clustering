import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

#CONFIG
IMAGE_FOLDER = "faces"
K = 3
N_RUNS = 5
IMG_SIZE = 32  # resize รูปเป็น 32x32 ก่อน flatten

#โหลดรูปและแปลงเป็น feature vector
def load_and_flatten(path, size=IMG_SIZE):
    img = plt.imread(path).astype(float)
    if img.max() > 1.0:
        img = img / 255.0
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.ndim == 3:
        img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    h, w = img.shape
    row_idx = (np.arange(size) * h / size).astype(int)
    col_idx = (np.arange(size) * w / size).astype(int)
    return img[np.ix_(row_idx, col_idx)].flatten()

image_paths = glob(os.path.join(IMAGE_FOLDER, "*.jpg")) + \
              glob(os.path.join(IMAGE_FOLDER, "*.png")) + \
              glob(os.path.join(IMAGE_FOLDER, "*.jpeg"))

embeddings = []
valid_paths = []

print("กำลังโหลดรูป...")
for path in image_paths:
    try:
        vec = load_and_flatten(path)
        embeddings.append(vec)
        valid_paths.append(path)
        print(f"  ✓ {os.path.basename(path)}")
    except Exception as e:
        print(f"  ✗ {os.path.basename(path)}: {e}")

if len(embeddings) == 0:
    print("ไม่พบรูปกรุณาตรวจสอบโฟลเดอร์")
    exit()

X = np.array(embeddings)
print(f"\nพบรูปทั้งหมด {len(X)} รูป → แบ่งเป็น {K} กลุ่ม\n")

#K-Means (รันหลายรอบ เลือกผลดีที่สุด)
def run_kmeans(X, K, seed):
    np.random.seed(seed)
    C = X[np.random.permutation(len(X))[:K]].copy()
    while True:
        D = np.zeros((len(X), K))
        for i in range(K):
            D[:, i] = np.mean((X - C[i])**2, axis=1)
        idx = np.argmin(D, axis=1)
        Cold = C.copy()
        for i in range(K):
            members = X[idx == i]
            if len(members) > 0:
                C[i] = np.mean(members, axis=0)
        if np.sum(np.abs(Cold - C)) == 0:
            break
    inertia = sum(np.mean((X[idx == i] - C[i])**2) for i in range(K) if np.sum(idx == i) > 0)
    return idx, C, inertia

print(f"รัน K-Means {N_RUNS} รอบ...")
best_idx, best_C, best_inertia = None, None, float('inf')
for run in range(N_RUNS):
    idx, C, inertia = run_kmeans(X, K, seed=run)
    print(f"  รอบ {run + 1}: inertia = {inertia:.4f}")
    if inertia < best_inertia:
        best_inertia = inertia
        best_idx = idx
        best_C = C

idx = best_idx
C = best_C
print(f"\nเลือกผลที่ดีที่สุด (inertia = {best_inertia:.4f})")

#PCA 2D + Dashboard
X_c = X - X.mean(axis=0)
_, _, Vt = np.linalg.svd(X_c, full_matrices=False)
X_2d = X_c @ Vt[:2].T

COLORS = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6']

def get_nickname(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return name.split('_', 1)[1] if '_' in name else name

#representative image (ใกล้ centroid ที่สุด) ของแต่ละกลุ่ม
def get_representative(X, C, idx, group_id):
    members_i = np.where(idx == group_id)[0]
    if len(members_i) == 0:
        return None
    dists = np.mean((X[members_i] - C[group_id])**2, axis=1)
    return members_i[np.argmin(dists)]

#คำนวณ brightness เฉลี่ยของแต่ละกลุ่ม (0=มืด, 1=สว่าง)
def group_brightness(X, idx, group_id):
    return np.mean(X[idx == group_id])

#สร้าง label อธิบายกลุ่ม
brightness_per_group = [group_brightness(X, idx, g) for g in range(K)]
brightness_rank = np.argsort(brightness_per_group)
auto_label = {}
labels = ['Dark tone', 'Medium tone', 'Bright tone']
for rank, g in enumerate(brightness_rank):
    count = int(np.sum(idx == g))
    auto_label[g] = f"Group {g+1}: {labels[rank]} ({count} photos)"

#PCA scatter
fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor('#1e1e2e')
ax.set_facecolor('#2a2a3e')

for group_id in range(K):
    mask = idx == group_id
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               color=COLORS[group_id % len(COLORS)],
               s=120, edgecolors='white', linewidths=0.8,
               label=auto_label[group_id], zorder=3)
    for i in np.where(mask)[0]:
        ax.annotate(get_nickname(valid_paths[i]),
                    (X_2d[i, 0], X_2d[i, 1]),
                    fontsize=8, color='white', ha='center', va='bottom',
                    xytext=(0, 7), textcoords='offset points')
    rep_i = get_representative(X, C, idx, group_id)
    if rep_i is not None:
        ax.scatter(X_2d[rep_i, 0], X_2d[rep_i, 1],
                   color=COLORS[group_id % len(COLORS)],
                   s=300, edgecolors='white', linewidths=2,
                   marker='*', zorder=4)

ax.set_title("K-Means Clustering — PCA 2D View", color='white', fontsize=14, pad=12)
ax.set_xlabel("Principal Component 1", color='#aaaaaa', fontsize=10)
ax.set_ylabel("Principal Component 2", color='#aaaaaa', fontsize=10)
ax.tick_params(colors='#aaaaaa')
for spine in ax.spines.values():
    spine.set_edgecolor('#555577')

legend = ax.legend(facecolor='#2a2a3e', edgecolor='#555577',
                   labelcolor='white', fontsize=9, loc='lower left')
ax.text(0.98, 0.98,
        "* = most representative photo\nclose dots = similar pixel pattern",
        transform=ax.transAxes, fontsize=8, color='#aaaaaa',
        va='top', ha='right',
        bbox=dict(facecolor='#1e1e2e', edgecolor='#555577', boxstyle='round,pad=0.4'))

plt.tight_layout()
plt.savefig("pca_plot.png", dpi=120, facecolor=fig.get_facecolor())
plt.show()

print("\nผลลัพธ์การแบ่งกลุ่ม")
for group_id in range(K):
    member_paths = [valid_paths[i] for i in range(len(X)) if idx[i] == group_id]
    print(f"\n{auto_label[group_id]}:")
    for p in member_paths:
        print(f"  - {get_nickname(p)}")

print("\nSaved: pca_plot.png")

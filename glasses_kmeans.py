import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from glob import glob

matplotlib.rcParams['font.family'] = 'Tahoma'

# ==========================================
# CONFIG
# ==========================================
# เลือกระหว่างโฟลเดอร์ Data_set_CAI หรือ faces อัตโนมัติ
if os.path.exists("Data_set_CAI"):
    IMAGE_FOLDER = "Data_set_CAI"
else:
    IMAGE_FOLDER = "faces"

K = 2               # 2 กลุ่ม: ใส่แว่น (Wearing Glasses) / ไม่ใส่แว่น (No Glasses)
N_RUNS = 5          # จำนวนรอบสุ่ม K-Means เพื่อหาผลลัพธ์ที่ดีที่สุด
OUTPUT_PLOT = "glasses_plot.png"  # บันทึกแยกเป็นไฟล์ใหม่ ไม่ทับ pca_plot.png เดิม

# โหลดรูปภาพแปลงเป็น Grayscale (0.0 - 1.0) รองรับทั้ง PIL และ plt.imread
def load_image_gray(path):
    try:
        from PIL import Image
        img = Image.open(path).convert('L')
        arr = np.array(img, dtype=float)
        return arr / 255.0
    except Exception:
        img = plt.imread(path).astype(float)
        if img.max() > 1.0:
            img = img / 255.0
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        if img.ndim == 3:
            img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        return img

# ตรวจจับกรอบใบหน้า (Face Bounding Box) เพื่อตัดขอบภาพและพื้นหลังออก
def get_face_box(img):
    h, w = img.shape
    center_bright = np.mean(img[h//5 : 4*h//5, w//5 : 4*w//5])
    edge_bright   = (np.mean(img[:h//5, :]) + np.mean(img[4*h//5:, :]) +
                     np.mean(img[:, :w//5]) + np.mean(img[:, 4*w//5:])) / 4

    if center_bright > edge_bright:
        face_mask = img > np.percentile(img, 40)
    else:
        face_mask = img < np.percentile(img, 60)

    rows = np.where(face_mask.any(axis=1))[0]
    cols = np.where(face_mask.any(axis=0))[0]

    if len(rows) < 10 or len(cols) < 10:
        return 0, h, 0, w
    return rows[0], rows[-1], cols[0], cols[-1]

# สกัด Feature สำหรับการจำแนกการใส่แว่น:
# 1. feat_bridge: ความคมชัดขอบแนวนอนบริเวณดั้งแว่น (Nose Bridge) เทียบกับผิวหน้า
# 2. feat_frame:  ความหนาแน่นขอบรอบดวงตาและกรอบแว่นเทียบกับผิวหน้า (Eye & Frame)
# 3. feat_eye:    ความเข้มของเส้นขอบรอบดวงตารวม
def get_glasses_features(path):
    img = load_image_gray(path)
    r0, r1, c0, c1 = get_face_box(img)
    fh = max(r1 - r0, 10)
    fw = max(c1 - c0, 10)
    face = img[r0:r1, c0:c1]

    # กำหนด Region of Interest (ROI) บนใบหน้า
    # - บริเวณดวงตาและกรอบแว่น (28% - 52% ของความสูงใบหน้า)
    eye_roi = face[int(fh * 0.28) : int(fh * 0.52), int(fw * 0.18) : int(fw * 0.82)]
    # - บริเวณดั้งแว่นเชื่อมระหว่างสองตา (33% - 46% ของความสูง, กึ่งกลาง 40% - 60%)
    bridge_roi = face[int(fh * 0.33) : int(fh * 0.46), int(fw * 0.40) : int(fw * 0.60)]
    # - บริเวณแก้มเป็นผิวหน้าอ้างอิงความเนียนตามธรรมชาติ (60% - 75% ของความสูง)
    cheek_roi = face[int(fh * 0.60) : int(fh * 0.75), int(fw * 0.20) : int(fw * 0.80)]

    # คำนวณ Gradients (Sobel Edges)
    gy, gx = np.gradient(eye_roi)
    eye_mag = np.mean(np.sqrt(gx**2 + gy**2))

    bgy, bgx = np.gradient(bridge_roi)
    bridge_horiz = np.mean(np.abs(bgy))  # ขอบแนวนอนสะพานแว่น
    bridge_mag   = np.mean(np.sqrt(bgx**2 + bgy**2))

    cgy, cgx = np.gradient(cheek_roi)
    cheek_mag = np.mean(np.sqrt(cgx**2 + cgy**2)) + 1e-5

    # คำนวณอัตราส่วนเทียบกับความเนียนของผิวหน้า พร้อม log1p เพื่อลดผลกระทบจาก Outlier
    ratio_bridge = bridge_horiz / cheek_mag
    ratio_frame  = (bridge_mag + eye_mag) / (2.0 * cheek_mag)

    feat_bridge = np.log1p(ratio_bridge)
    feat_frame  = np.log1p(ratio_frame)
    feat_eye    = eye_mag

    # คะแนนวัดระดับความน่าจะเป็นของการใส่แว่น
    glasses_score = ratio_bridge * 1.5 + ratio_frame

    return [feat_bridge, feat_frame, feat_eye], glasses_score

# ดึงชื่อเล่นจากชื่อไฟล์
def get_nickname(path):
    base = os.path.basename(path)
    name = base.split('.')[0]  # ตัดส่วนขยาย (รองรับกรณี .jpg.jpg)
    return name.split('_', 1)[1] if '_' in name else name

# ค้นหาไฟล์ภาพทั้งหมด
image_paths = glob(os.path.join(IMAGE_FOLDER, "*.jpg")) + \
              glob(os.path.join(IMAGE_FOLDER, "*.png")) + \
              glob(os.path.join(IMAGE_FOLDER, "*.jpeg"))

embeddings = []
valid_paths = []
glasses_scores = []

print(f"กำลังโหลดรูปจากโฟลเดอร์ '{IMAGE_FOLDER}'...")
for path in sorted(image_paths):
    try:
        feat, score = get_glasses_features(path)
        embeddings.append(feat)
        valid_paths.append(path)
        glasses_scores.append(score)
        nick = get_nickname(path)
        print(f"  ✓ {nick:12s} ({os.path.basename(path):25s}) score={score:.2f}")
    except Exception as e:
        print(f"  ✗ {os.path.basename(path)}: {e}")

if len(embeddings) == 0:
    print(f"ไม่พบรูปภาพใน '{IMAGE_FOLDER}' กรุณาตรวจสอบโฟลเดอร์")
    exit()

X = np.array(embeddings)
glasses_scores = np.array(glasses_scores)
print(f"\nพบรูปทั้งหมด {len(X)} รูป → แบ่งเป็น {K} กลุ่ม (วัดจากการใส่แว่น)\n")

# Z-score Normalization ก่อนทำ K-Means เพื่อถ่วงน้ำหนักทุก Feature อย่างสมดุล
X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

# K-Means Clustering (รันหลายรอบเพื่อเลือก inertia ที่ดีที่สุด)
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
    idx, C, inertia = run_kmeans(X_norm, K, seed=run)
    print(f"  รอบ {run + 1}: inertia = {inertia:.4f}")
    if inertia < best_inertia:
        best_inertia = inertia
        best_idx = idx
        best_C = C

idx = best_idx
C = best_C
print(f"\nเลือกผลที่ดีที่สุด (inertia = {best_inertia:.4f})")

# PCA 2D สำหรับแสดงผลภาพใน 2 มิติ
X_c = X_norm - X_norm.mean(axis=0)
_, _, Vt = np.linalg.svd(X_c, full_matrices=False)
X_2d = X_c @ Vt[:2].T

# หาตัวแทนกลุ่ม (คนที่มีตำแหน่งใกล้ Centroid มากที่สุด)
def get_representative(X, C, idx, group_id):
    members_i = np.where(idx == group_id)[0]
    if len(members_i) == 0:
        return None
    dists = np.mean((X[members_i] - C[group_id])**2, axis=1)
    return members_i[np.argmin(dists)]

# กำหนดชื่อกลุ่มอัตโนมัติ: กลุ่มที่คะแนนเฉลี่ยสูงกว่า = ใส่แว่น, ต่ำกว่า = ไม่ใส่แว่น
avg_scores_per_group = [float(np.mean(glasses_scores[idx == g])) for g in range(K)]
sorted_groups = np.argsort(avg_scores_per_group)[::-1]  # เรียงจากคะแนนสูง -> ต่ำ

group_labels = {}
if K == 2:
    group_labels[sorted_groups[0]] = "ใส่แว่น (Glasses)"
    group_labels[sorted_groups[1]] = "ไม่ใส่แว่น (No Glasses)"
else:
    for rank, g in enumerate(sorted_groups):
        group_labels[g] = f"ระดับแว่น {rank+1}"

auto_label = {}
for g in range(K):
    count = int(np.sum(idx == g))
    avg_s = avg_scores_per_group[g]
    auto_label[g] = f"Group {g+1}: {group_labels[g]} (คะแนนเฉลี่ย={avg_s:.2f}, {count} คน)"

# พล็อต Dashboard ด้วย PCA Scatter Plot (Dark Theme สไตล์เดียวกับ face_kmeans.py)
fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor('#1e1e2e')
ax.set_facecolor('#2a2a3e')

for group_id in range(K):
    mask = idx == group_id
    # เขียวสดสำหรับใส่แว่น, ส้มแดงสำหรับไม่ใส่แว่น
    color = '#2ecc71' if group_labels[group_id].startswith("ใส่แว่น") else '#e74c3c'
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               color=color,
               s=130, edgecolors='white', linewidths=0.9,
               label=auto_label[group_id], zorder=3)
    for i in np.where(mask)[0]:
        ax.annotate(get_nickname(valid_paths[i]),
                    (X_2d[i, 0], X_2d[i, 1]),
                    fontsize=8, color='white', ha='center', va='bottom',
                    xytext=(0, 7), textcoords='offset points')
    rep_i = get_representative(X_norm, C, idx, group_id)
    if rep_i is not None:
        ax.scatter(X_2d[rep_i, 0], X_2d[rep_i, 1],
                   color=color,
                   s=320, edgecolors='white', linewidths=2.2,
                   marker='*', zorder=4)

ax.set_title("K-Means Clustering — การใส่แว่น (Glasses Detection)", color='white', fontsize=14, pad=12)
ax.set_xlabel("Principal Component 1 (Bridge & Frame Features)", color='#aaaaaa', fontsize=10)
ax.set_ylabel("Principal Component 2 (Eye Contrast & Edge Density)", color='#aaaaaa', fontsize=10)
ax.tick_params(colors='#aaaaaa')
for spine in ax.spines.values():
    spine.set_edgecolor('#555577')

legend = ax.legend(facecolor='#2a2a3e', edgecolor='#555577',
                   labelcolor='white', fontsize=9, loc='lower left')
ax.text(0.98, 0.98,
        "* = ตัวแทนกลุ่ม\nวัดจากขอบดั้งแว่น (Bridge) & ขอบกรอบแว่น (Frame)",
        transform=ax.transAxes, fontsize=8, color='#aaaaaa',
        va='top', ha='right',
        bbox=dict(facecolor='#1e1e2e', edgecolor='#555577', boxstyle='round,pad=0.4'))

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=120, facecolor=fig.get_facecolor())
plt.show()

# พิมพ์รายงานผลการจัดกลุ่มรายคนลง Terminal
print("\n" + "="*50)
print("ผลลัพธ์การแบ่งกลุ่มตามการใส่แว่น (Glasses Clustering)")
print("="*50)
for g in sorted_groups:
    print(f"\n{auto_label[g]}:")
    for i in range(len(X)):
        if idx[i] == g:
            print(f"  - {get_nickname(valid_paths[i]):15s}  (คะแนนแว่น={glasses_scores[i]:.2f})")

print(f"\nSaved result plot to: {OUTPUT_PLOT}")

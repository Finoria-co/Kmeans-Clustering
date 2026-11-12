# K-Means Clustering — Face Shape

วิชา Machine Learning | กลุ่มโปรเจกต์ Finoria

## ไฟล์ในโปรเจกต์

### `face_kmeans.py`
โค้ด Python สำหรับจัดกลุ่มรูปร่างหน้าของนักศึกษา 30 คน ด้วย K-Means Clustering (K=3)

**วิธีทำงาน:**
- โหลดรูปภาพหน้าจากโฟลเดอร์ `faces/`
- วัด **อัตราส่วนกว้าง ÷ สูง (ratio)** ของหน้าในแต่ละรูป
- จัดกลุ่มด้วย K-Means เป็น 3 กลุ่มตามรูปร่างหน้า:
  - **หน้ากลม** — ratio สูง (≈ 1.0)
  - **หน้ารูปไข่** — ratio กลาง (≈ 0.85)
  - **หน้ายาว** — ratio ต่ำ (< 0.75)
- แสดงผลด้วย PCA 2D scatter plot

**Library ที่ใช้:** `numpy`, `matplotlib` (ไม่มี import เพิ่ม)

### `pca_plot.png`
กราฟผลลัพธ์การจัดกลุ่ม แสดงตำแหน่งของแต่ละคนใน PCA 2D space พร้อม label ชื่อเล่น

## วิธีรัน

```bash
# วางรูปภาพในโฟลเดอร์ faces/ แล้วรัน
python face_kmeans.py
```

ผลลัพธ์จะแสดงในหน้าต่างกราฟและบันทึกเป็น `pca_plot.png`

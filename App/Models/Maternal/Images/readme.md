# 孕产妇及胎儿超声异常诊断指南 (Maternal & Fetal Ultrasound Abnormalities)

本仓库致力于收集、处理与分析孕产妇超声图像数据，旨在通过深度学习方法辅助识别孕期常见并发症及胎儿结构异常。

---

## 目录
- [项目概述](#项目概述)
- [主要超声检测项目](#主要超声检测项目)
  - [1. 胎盘与脐带异常](#1-胎盘与脐带异常)
  - [2. 羊水异常](#2-羊水异常)
  - [3. 宫颈及子宫结构异常](#3-宫颈及子宫结构异常)
  - [4. 胎儿发育与结构异常](#4-胎儿发育与结构异常)
- [数据格式与规范](#数据格式与规范)
- [使用说明](#使用说明)

---

## 项目概述
在产科超声检查中，早期且准确地发现孕产妇及胎儿的异常情况对降低孕产妇死亡率和新生儿出生缺陷率至关重要。本项目包含用于评估超声影像中关键解剖结构及病变区域的数据集与AI模型。

---

## 主要超声检测项目

### 1. 胎盘与脐带异常 (Placental & Umbilical Cord Abnormalities)
* **前置胎盘 (Placenta Previa):** 胎盘附着于子宫下段，甚至覆盖宫颈内口。
* **胎盘植入 (Placenta Accreta Spectrum, PAS):** 胎盘绒毛侵入子宫肌层，易引发产后大出血。
* **胎盘早剥 (Placental Abruption):** 正常位置的胎盘在胎儿娩出前部分或全部从子宫壁剥离。
* **单脐动脉 (Single Umbilical Artery, SUA):** 脐带内仅有一条脐动脉和一条脐静脉，可能伴随其他器官畸形。

### 2. 羊水异常 (Amniotic Fluid Abnormalities)
* **羊水过多 (Polyhydramnios):** 羊水暗区垂直深度 (AFV) ≥ 8cm 或羊水指数 (AFI) ≥ 25cm。
* **羊水过少 (Oligohydramnios):** 羊水最大暗区深度 (AFV) ≤ 2cm 或羊水指数 (AFI) ≤ 5cm，可能导致胎儿肺发育不良或受压。

### 3. 宫颈及子宫结构异常 (Cervical & Uterine Pathology)
* **宫颈机能不全 (Cervical Incompetence):** 孕中期宫颈管短缩（通常 < 25mm）或宫颈内口呈“Y/V/U”形扩张，易导致早产。
* **子宫肌瘤合并妊娠 (Uterine Fibroids in Pregnancy):** 评估肌瘤位置、大小及其对胎盘和产道的影响。

### 4. 胎儿发育与结构异常 (Fetal Anomalies)
* **中枢神经系统异常:** 脑积水 (Hydrocephalus)、无脑儿 (Anencephaly)、脊柱裂 (Spina Bifida)。
* **先天性心脏病 (Congenital Heart Defects):** 房间隔/室间隔缺损、法洛四联症等。
* **生长受限 (Fetal Growth Restriction, FGR):** 胎儿双顶径 (BPD)、腹围 (AC) 及股骨长 (FL) 明显低于同孕龄标准。

---

## 数据格式与规范
* **影像格式:** DICOM (.dcm) / PNG / JPG
* **标注格式:** Mask 分割图 (Segmentation) 及 Bounding Box 目标检测框 (JSON/YOLO 格式)

---

## 使用说明
将超声图像放入对应的测试文件夹中，运行分析脚本：

```bash
python predict.py --input ./data/test_ultrasound.jpg --model maternal_edge_v1.pth

# SiGAS
The code for "Siamese Generalizable-attention Instance Segmentation for Intersection Perception System" and RopeIns Dataset.
Available for public after this article is published.

| **Homologous / Heterologous** | **Params (M)** | **mAP (%)** | **AP_Car (%)** | **AP_BV (%)** | **AP_Ped (%)** | **AP_Cyc (%)** |
|:-----------------------------:|:--------------:|:-----------:|:--------------:|:-------------:|:--------------:|:--------------:|
| **Mask-RCNN**                 | 43.7           |             |                |               |                |                |
| **Cascade Mask-RCNN**         | 76.8           |             |                |               |                |                |
| **PointRend**                 | 55.7           |             |                |               |                |                |
| **SOLOv2**                    | 46.0           |             |                |               |                |                |
| **Panoptic-Deeplab**          | 30.3           | 51.0 / 80.5 | 77.6 / 93.6    | 41.8 / 81.3   | 37.7 / 65.4    | 46.8 / 81.5    |
| **Ours**                      | 38.5           | 69.3 / 85.5 | 88.8 / 95.2    | 66.1 / 87.9   | 55.0 / 73.0    | 67.2 / 85.7    |

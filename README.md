# SiGAS
The code for IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS(2024) paper "Si-GAIS: Siamese Generalizable-Attention Instance Segmentation for Intersection Perception System" and RopeIns Dataset.


 **Homologous / Heterologous** | **Params (M)** | **mAP (%)** | **AP_Car (%)** | **AP_BV (%)** | **AP_Ped (%)** | **AP_Cyc (%)** 
:-----------------------------:|:--------------:|:-----------:|:--------------:|:-------------:|:--------------:|:--------------:
 **Mask-RCNN**                 | 43.7           | 42.4 / 80.0 | 71.1 / 91.8    | 30.4 / 77.5   | 32.3 / 70.4    | 35.8 / 80.5    
 **Cascade Mask-RCNN**         | 76.8           | 46.4 / 81.3 | 73.5 / 92.3    | 34.3 / 79.8   | 34.1 / 69.7    | 43.8 / 83.3    
 **PointRend**                 | 55.7           | 44.6 / 81.2 | 74.3 / 92.9    | 29.4 / 78.2   | 35.5 / 70.8    | 39.4 / 83.0    
 **SOLOv2**                    | 46             | 22.6 / 43.5 | 50.0 / 67.7    | 17.5 / 50.1   | 5.7 / 16.6     | 17.3 / 39.7    
 **Panoptic-Deeplab**          | 30.3           | 51.0 / 80.5 | 77.6 / 93.6    | 41.8 / 81.3   | 37.7 / 65.4    | 46.8 / 81.5    
 **Ours**                      | 38.5           | **69.3** / **85.5** | **88.8** / **95.2**    | **66.1** / **87.9**   | **55.0** / **73.0**    | **67.2** / **85.7**   

ropeins dataset: https://pan.baidu.com/s/1_g7oxIiVD4UAYyHVBtT9-Q?pwd=dwzv

trained model: https://pan.baidu.com/s/1Nx0jHZ-4ceDrWY4_q5aHIQ?pwd=ahn1
If you find our work helpful for your research, please consider citing the following BibTeX entry.

```BibTeX
@ARTICLE{10559784,
  author={Wang, Huanan and Zhang, Xinyu and Wang, Hong and Jun, Li},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Si-GAIS: Siamese Generalizable-Attention Instance Segmentation for Intersection Perception System}, 
  year={2024},
  pages={1-16},
  doi={10.1109/TITS.2024.3411647}}
```

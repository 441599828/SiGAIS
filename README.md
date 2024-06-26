 SiGAS
The code for IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS(2024) paper "Si-GAIS: Siamese Generalizable-Attention Instance Segmentation for Intersection Perception System" and RopeIns Dataset.


 **Homologous / Heterologous** | **mAP (%)** | **AP_Car (%)** | **AP_BV (%)** | **AP_Ped (%)** | **AP_Cyc (%)** 
:-----------------------------:|:-----------:|:--------------:|:-------------:|:--------------:|:--------------:
 **Mask-RCNN**                 | 42.4 / 80.0 | 71.1 / 91.8    | 30.4 / 77.5   | 32.3 / 70.4    | 35.8 / 80.5    
 **Cascade Mask-RCNN**         | 46.4 / 81.3 | 73.5 / 92.3    | 34.3 / 79.8   | 34.1 / 69.7    | 43.8 / 83.3    
 **PointRend**                 | 44.6 / 81.2 | 74.3 / 92.9    | 29.4 / 78.2   | 35.5 / 70.8    | 39.4 / 83.0    
 **Mask2Former**               | 57.4 / 84.5 | 77.2 / 87.9    | 50.6 / 92.6   | 44.3 / 71.6    | 57.5 / 86.0
 **MaskDINO**                  | 61.6 / 88.5 | 84.3 / 96.3    | 54.6 / 92.6   | 47.3 / 75.9    | 60.2 / 89.3
 **OneFormer**                 | 55.2 / 82.3 | 74.6 / 84.9    | 49.2 / 89.6   | 40.0 / 70.1    | 56.9 / 84.6
 **OpenSeeD**                  | 61.3 / 87.5 | 79.6 / 90.6    | 52.8 / 91.5   | 52.3 / 78.0    | 60.5 / 89.7
 **Panoptic-Deeplab**          | 51.0 / 80.5 | 77.6 / 93.6    | 41.8 / 81.3   | 37.7 / 65.4    | 46.8 / 81.5    
 **Ours**                      | **69.3** / **85.5** | **88.8** / **95.2**    | **66.1** / **87.9**   | **55.0** / **73.0**    | **67.2** / **85.7**   

ropeins dataset: https://pan.baidu.com/s/1_g7oxIiVD4UAYyHVBtT9-Q?pwd=dwzv

trained model: https://pan.baidu.com/s/1Nx0jHZ-4ceDrWY4_q5aHIQ?pwd=ahn1
If you find our work helpful for your research, please consider citing the following BibTeX entry.

```BibTeX
@ARTICLE{wang2024sigais,
  author={Wang, Huanan and Zhang, Xinyu and Wang, Hong and Jun, Li},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Si-GAIS: Siamese Generalizable-Attention Instance Segmentation for Intersection Perception System}, 
  year={2024},
  pages={1-16},
  doi={10.1109/TITS.2024.3411647}}
```

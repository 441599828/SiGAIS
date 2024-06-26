export CUDA_VISIBLE_DEVICES=0,1,2,3
python train_net.py --config-file /home/zkyd/code/sigais_final/Si-GAIS/SiGAS_code/configs/R52_bs32_544_960.yaml --num-gpus 4 MODEL.PAN False MODEL.INSPAN True OUTPUT_DIR "./00_fuxian"
# python train_net.py --config-file /home/tsinghua-adept/SiGAS_beta/configs/R52_bs32_544_960.yaml --num-gpus 4 MODEL.PAN False MODEL.INSPAN True OUTPUT_DIR "./03_pitch_1"
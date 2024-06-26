export CUDA_VISIBLE_DEVICES=0,1,2,3
python train_net.py --config-file ./configs/R52_bs32_544_960.yaml --num-gpus 4 OUTPUT_DIR "./sigais"

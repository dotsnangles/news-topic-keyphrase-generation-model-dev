# for int8 and peft training
# nohup CUDA_VISIBLE_DEVICES=0 python train.py > /dev/null 2>&1&

# for peft and deepspeed(ZeRO) training
# accelerate config --config_file ./conf/accelerate.yaml
# nohup accelerate launch --config_file ./conf/accelerate.yaml train.py > /dev/null 2>&1&
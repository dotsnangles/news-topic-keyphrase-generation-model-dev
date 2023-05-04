# for int8 and peft training
CUDA_VISIBLE_DEVICES=0 python train.py

# for peft and deepspeed(ZeRO) training
# accelerate config
# accelerate launch train.py


python train.py -a RepVGG-B1  --dist-backend 'nccl'  --lr 0.001 --pretrained --world-size 1 -b 32 --rank 0 --workers 16 /home/liyongjing/Egolee_2021/data/TrainData/train_fall_down/rep_vgg_format

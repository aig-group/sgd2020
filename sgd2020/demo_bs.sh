seed='1'
arch='resnet18'
dataset='cifar10'
num_channels='3'
num_classes='10'
lr='0.01'
bs='64'
epochs='1000'

for arch in 'resnet18' 'vgg16_bn'
do
    for dataset in 'cifar10' 'cifar100'
    do
        for bs in 64 128 192 256 320 384 448 512
        do
            python3 main.py \
            --seed ${seed} \
            --arch ${arch}  \
            --dataset ${dataset} \
            --num_channels ${num_channels} \
            --num_classes ${num_classes} \
            --lr ${lr} \
            --batch_size ${bs} \
            --epochs ${epochs} \
            --save_dir ./runs_bs/${arch}/${dataset}/lr=${lr}_bs=${bs}_${seed}
        done
    done
done

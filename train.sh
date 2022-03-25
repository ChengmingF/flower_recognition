for net in densenet161 resnet18 mobilenet_v2 vgg16
do
    python train.py --network $net
    python train.py --network $net --new_fc True
done


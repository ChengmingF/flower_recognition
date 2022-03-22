for net in densenet161 resnet18 mobilenet_v2
do
    for new_fc in True False
    do
        python train.py --network $net --$new_fc
    done
done

python train.py --network vgg16

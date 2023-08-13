dataname='CIFAR10LT CIFAR100LT'
IF='1 10 50 100 200'
losses='CrossEntropyLoss BalancedSoftmax'

for d in $dataname
do
    for f in $IF
    do
        for l in $losses
        do
            if [ $f == '1' ] && [ $l == 'BalancedSoftmax' ]; then
                continue
            else
                echo "dataset: $d, loss: $l, IF: $f"
                python main.py --config configs.yaml \
                            DEFAULT.exp_name $l-IF_$f \
                            DATASET.name $d \
                            DATASET.imbalance_type exp \
                            DATASET.imbalance_factor $f \
                            LOSS.name $l
            fi
        done
    done
done


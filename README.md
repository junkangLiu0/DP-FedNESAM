# Setup

- torch == 2.1
- torchvision == 0.19
- ray == 1.0

# Dataset

- CIFAR10, CIFAR100
- NLP datasets, such as SST-2, QQP, MRPC, and QNLI 

# Model

- GNResNet-10 (Group Norm ResNet-10), GNResNet-18 (Group Norm ResNet-18)
- Vision Transformer (ViT-Base)
- Roberta-Base

# Run

```python
python DP_new.py --alg DP-FedNESAM --lr 0.1 --data_name CIFAR100 --alpha_value 0.6 --alpha 0.9 --epoch 301 --extname CIFAR100 --lr_decay 0.998 --gamma 0.2 --CNN resnet10 --E 5 --batch_size 50 --gpu 0 --num_gpus_per 0.1 --normalization GN --selection 0.1 --print 0 --dp_sigma 0.8 --rho 0.1 --C 1.0 --momentum 0 --num_workers 500 --pre 1 --ls_sigma 0 --maxnorm 10 --clip True --preprint 1
```

Explanations of arguments:

- `alg`: DP-FedAvg-per, DP-FedSAM-per

- `alpha_value`: parameter of Dirichlet Distribution, controling the level of Non-IID

- `E`: local training epochs for each client

- `selection`: the selection fraction of total clients in each round

- `dp_sigma`: noise multiplier for DP

- `C`: the threshold of clipping in DP

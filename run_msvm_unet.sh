python train.py -d synapse -r 0 -c msvm_unet_synapse
python train.py -d synapse -r 1 -c msvm_unet_synapse
python train.py -d synapse -r 2 -c msvm_unet_synapse

python test.py -d synapse -m msvm_unet

python train.py -d acdc -r 0 -c msvm_unet_acdc
python train.py -d acdc -r 1 -c msvm_unet_acdc
python train.py -d acdc -r 2 -c msvm_unet_acdc

python test.py -d acdc -m msvm_unet

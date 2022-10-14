# Template

An ever evolving template I use for all the projects I code. 

File structure I like to follow: 

          
```
main.py -> For all the random seed initialisation and iterations over epochs and splits
├── datasets
│   ├── dataloader.py -> This is usually dataset specific.
│   ├── transforms.py -> Custom transforms for the dataloader and augmentations if any.
│   ├── data.py -> Main file that calls specific datasets and transforms.
├── models
│   ├── model.py -> File where I define the model.
│   ├── loss.py -> File for Loss class that defines all the criteria.
│   ├── enums.py -> Stores all enums.
│   ├── optimiser.py -> Defines the optimiser and schedulers. 
├── utils
│   ├── ddp_utils.py -> DDP related functions
│   ├── utils.py -> Other writer and logger utils. 
│   ├── pytorch_utils.py -> For loading from file and saving checkpoints and such
├── configs -> This is used for OmegaConf and hydra andstores different configs for models, optimsers and wandb used for the project.
│   ├── datasets 
│       ├── config.yaml
│   ├── models 
│       ├── config.yaml
│   ├── optimiser 
│       ├── config.yaml
│   ├── wandb 
│       ├── config.yaml
```

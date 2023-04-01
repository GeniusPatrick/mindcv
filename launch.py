import os
import yaml


with open("cfg_list.txt", "r") as f:
    cfg_list = [line.strip() for line in f.readlines()]

for cfg in cfg_list:
    data_dir = "/home/cvgroup/dataset/imagenet2012"
    output_dir = "outputs"
    with open(cfg, "r") as f:
        model = yaml.safe_load(f)['model']
    cmd = f"mpirun -np 8 python -u train.py -c {cfg} --data_dir {data_dir} --ckpt_save_dir {os.path.join(output_dir, model)} --val_while_train True"
    print(f"Running command {cmd}")
    os.system(cmd)

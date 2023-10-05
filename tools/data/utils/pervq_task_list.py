
def make_local_config(exp_name, vname):
    config_data = ""
    with open(f'configs/train/local/{exp_name}.py', 'r') as f:
        for line in f:
            line = line.replace('video_name',vname)
            line = line.replace('/home/lr','/gdata/lirui')
            line = line.replace('/gdata/lirui/dataset/YouTube-VOS','/dev/shm')
            config_data += line

    with open(f'configs/train/ypb/pervq/{vname}.py',"w") as f:
        f.write(config_data)

list_path = '/home/lr/dataset/YouTube-VOS/2018/train/youtube2018_train_list.txt'

with open(list_path, 'r') as f:
    for idx, line in enumerate(f.readlines()):
        sample = dict()
        vname, num_frames = line.strip('\n').split()
        make_local_config('train_vqvae_perv_cluster', vname)
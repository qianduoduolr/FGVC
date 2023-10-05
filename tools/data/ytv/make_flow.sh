cd /gdata/lirui/project/mmpt
python -m torch.distributed.launch --nproc_per_node=4  tools/data/utils/prepare_youtube_flow.py --num-gpu=4
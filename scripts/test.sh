# shellcheck disable=SC2164
# cd /home/can/Desktop/research/REDFormer  ## go to the REDFormer dir
export PYTHONPATH=$PWD/:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 \
python ./tools/test.py \
./projects/configs/redformer/redformer06.py \
./work_dirs/redformer06/latest.pth  \
--eval bbox

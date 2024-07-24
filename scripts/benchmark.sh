# shellcheck disable=SC2164
# cd /home/can/Desktop/research/REDFormer  ## go to the REDFormer dir
export PYTHONPATH=$PWD/:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 \
python ./tools/analysis_tools/benchmark.py \
./projects/configs/redformer/redformer.py \
--checkpoint ./work_dirs/redformer/latest.pth 
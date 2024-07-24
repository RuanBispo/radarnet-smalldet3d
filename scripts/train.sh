# shellcheck disable=SC2164
export PYTHONPATH=$PWD/:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 \
python ./tools/train.py \
  ./projects/configs/redformer/redformer06.py #\
  # --resume-from ./work_dirs/redformer/epoch_7.pth #\
  # --resume-from ./work_dirs/redformer/epoch_5.pth 
  # --resume-from ./work_dirs/redformer/epoch_2.pth \
  # --no-validate #\

# shellcheck disable=SC2164
export PYTHONPATH=$PWD/:$PYTHONPATH

path='/home/lasi/Documents/ruan/MFREDFormer/REDFormer/work_dirs/redformer'
python tools/analysis_tools/analyze_logs.py \
plot_curve $path/20240417_172832.log.json \
--title "REDFormer 3sweeps - + img_proc (0.6)" \
--keys loss_cls loss_bbox loss_context \
--out $path/20240417_172832.png
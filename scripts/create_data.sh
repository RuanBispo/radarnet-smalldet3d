# shellcheck disable=SC2164
python tools/create_data.py nuscenes \
  --root-path /home/lasi/Downloads/datasets/nuscenes/ \
  --out-dir /home/lasi/Downloads/datasets/nuscenes/occ_radar_metadata/ \
  --max-sweeps 6 \
  --extra-tag nuscenes \
  --version v1.0 \
  --canbus /home/lasi/Downloads/datasets/nuscenes/

nuscenes_path='/home/ruan/Downloads/nuScenes/'

docker run -d \
  --restart=always \
  --gpus all \
  --shm-size 15gb \
  -v "${PWD}/:/code" \
  -v "${nuscenes_path}:/code/data/nuScenes" \
  --name "radarnet" \
  radarnet-image \
  tail -f /dev/null

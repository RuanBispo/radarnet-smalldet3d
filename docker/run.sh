docker run \
  -it \
  --gpus all \
  -v "${PWD}/../:/code" \
  -v "nuscenes-path:/data" \
  image-name \
  /bin/bash

# Instructions:
### -it (run iteratively)
### --gpus all (use all GPUS. You can also select one)
### -v (mounted folders. The first is the code folder, 
###     and the second the data. You should change nuscenes-path
###     variable using your path)
### image-name (it is the name you gave when you built the image)
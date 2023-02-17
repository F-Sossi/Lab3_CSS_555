# Lab3_CSS_555
Cuda Lab CSS535

Lab File setup:

Clone project into folder

IDE Setup:

Visual studio 2022 

1. Open new cuda project (in a seperate folder)

2. Solution -> add external file (add all files from the cloned folder)

NOTE: This will keep all the source files in the cloned folder because we all have differnt development environments so you will have a seperate environmet of your choosing.

3. After changes made to files save and push from the cloned folder.

4. If you make any other .h files make sure they are put in the cloned folder and relinked to your chosen environment so we only have one copy floating around.


Please add to this for those using nvcc or other IDE so we are keeping this updated

Linux / local or remote

1. nvcc -g kernel.cu -o lab3

Nvprof command-line call to get the cach transactions
nvprof --metrics l1_cache_global_hit_rate,l1_cache_local_hit_rate

See below link. nvprof --query-metrics outputs lots of metric, a couple I used above
https://simoncblyth.bitbucket.io/env/notes/cuda/cuda_profiling/

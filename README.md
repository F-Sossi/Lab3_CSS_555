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

1. nvcc -arch=sm_86 -lcublas kernel.cu -o lab3

Note: -arch=sm86 is your compute capability in this case 8.6

For debugging:

nvcc -g -arch=sm_86 -lcublas kernel.cu -o lab3

For device debugging:

nvcc -g -G -arch=sm_86 -lcublas kernel.cu -o lab3


Profile:

1. ncu -o profile lab3


Nvprof command-line call to get the cach transactions
nvprof --metrics l1_cache_global_hit_rate,l1_cache_local_hit_rate

See below link. nvprof --query-metrics outputs lots of metric, a couple I used above
https://simoncblyth.bitbucket.io/env/notes/cuda/cuda_profiling/

Git Workflow:

clone to local machine: git clone <hhtps>
update: git pull
Make working branch: git branch <name of feature>
checkout branch: git checkout <name of branch>
Push branch to repo (so we can all see it): git push --set-upstream origin <name of branch>

NOTE: to check which branch you are on: git staus

Merging:

go to branch in github (drop down where it says main)
on the branch you want to merge hit contribute and fill out pull request


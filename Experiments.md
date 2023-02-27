Experiment 1:

DATA: REALDATA set
N value: 100
THREAD_PER_BLOCK: 32
TILE_SIZE: 32


Part 1 Configurations:

n = 8192

1. 31 threads per block, 253 blocks 
2. 20 threads per block, 410 blocks 
3. 32 threads per block, 256 blocks
4. 500 threads per block, 17 blocks
4.5 512 threads per block, 16 blocks
5. n = 8096, 1024 threads per block, 8 blocks
6. n = 8192, 1024 threads per block, 8 blocks
7. 128 threads per block, 64 blocks 


Part 2 Configurations:

n = 8192

1. 31 threads per block, 253 blocks
2. 32 threads per block, 256 blocks
3.5 128 threads per block, 64 blocks 

Higher threads/block did not compute or profile correctly and were excluded 


Part 3 Configurations:
... in progress

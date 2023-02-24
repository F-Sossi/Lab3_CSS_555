#Details about the device:
3 dimensions max (x, y, z)

##Grids / SMs:
128 max resident grids per device
68 count of SMs

##Blocks:
16 max resident blocks per multiprocessor (SM)

##Warps:
32 warp size
48 max resident warps

##Threads / SPs:
1536 max resident threads per multiprocessor (SM)
1024 max threads per block (this is also the max x or y dimension of a block)
8704 CUDA cores (SPs)

##Memory sizes:
64 K Number of 32-bit regular registers per multiprocessor
128 KB per SM for the L1 cache
5 MB for the L2 cache


#Experiment 1:

DATA: REALDATA set
N value: 100
THREAD_PER_BLOCK: 32
TILE_SIZE: 32


#Experiment 2:

N value: 4096 (divisible by 32)
<<< 128, 32 >>> (4096 / 32 = 128 - exceeds the max of resident blocks with minimum threads in a warp)
<<< 108, 38 >>> (4096 / 38 = 107.8 blocks; trying a slight difference to the one above)
<<< 32, 128 >>> (8704 SPs / 68 SMs = 128 threads; will some of the SMs not be used?)
<<< 34, 124 >>> (4096 / 124 = 30.03 blocks; trying a slight difference to the one above)


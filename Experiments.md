Experiment 1:

DATA: REALDATA set
N value: 100
THREAD_PER_BLOCK: 32
TILE_SIZE: 32


Part 1 Configurations:

n = 8192

Threads per block, number of blocks, n, time with memory allocation (nanoseconds), time without memory allocation (nanoseconds)
1. 31 threads per block, 253 blocks, 8192, 291871300, 39000
2. 20 threads per block, 410 blocks, 8192, 297227100, 23700
3. 32 threads per block, 256 blocks, 8192, 292321800, 20600
4. 500 threads per block, 17 blocks, 8192, 226543100, 16800 \n
exp 4.5. 512 threads per block, 16 blocks, 8192, 299462500, 19000
5. n = 8096, 1024 threads per block, 8 blocks - excluded from timing because n is different  
6. n = 8192, 1024 threads per block, 8 blocks, 8192, 282626900, 16500
7. 128 threads per block, 64 blocks, 8192, 295976300, 15700


Part 2 Configurations:

n = 8192
--- no timing performed, error in part2 ver1_1 kernel ---
Threads per block, number of blocks, n, time with memory allocation (nanoseconds), time without memory allocation (nanoseconds)
1. 31 threads per block, 253 blocks, 8192, 
2. 32 threads per block, 256 blocks
3. 500 threads per block, 17 blocks  (tile size 400?)
4. 1024 threads per block, 8 blocks (tile size = 1024) 
5. 128 threads per block, 64 blocks 


Part 3 Configurations:
n = 8192
Threads per block, number of blocks, n, time with memory allocation (nanoseconds), time without memory allocation (nanoseconds)
1. 31 threads per block, 253 blocks, 8192, 299211800, 16500
2. 32 threads per block, 256 blocks, 8192, 290932700, 15000
3. 500 threads per block, 17 blocks, 8192, 284501400, 15900
4. 1024 threads per block, 8 blocks, 8192, 274519400, 16100
5. 128 threads per block, 64 blocks, 8192, 284349500, 15900

Slightly better performance with part 3 vs. part 1. Part 2 shared memory usage seems to slow the matrix-vector multiplication down a little 

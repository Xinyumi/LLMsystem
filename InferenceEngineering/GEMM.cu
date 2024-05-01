# GEMM优化
__global__
void matrix_mul(float *A,float *B,float *C,int const M,const int K,int const N){
 
        int row=blockIdx.y*blockDim.y+threadIdx.y;
        int col=blockIdx.x*blockDim.x+threadIdx.x;
 
        __shared__ float tile_A[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float tile_B[BLOCK_SIZE][BLOCK_SIZE];
 
        float sum=0;
        for(int stride=0;stride<=K/BLOCK_SIZE;++stride){
                int id_m=row*K+stride*BLOCK_SIZE+threadIdx.x;
                if(row<M&&stride*BLOCK_SIZE+threadIdx.x<K){
                        tile_A[threadIdx.y][threadIdx.x]=A[id_m];
                }
                __syncthreads();
 
                int id_n=(stride*BLOCK_SIZE+threadIdx.y)*N+col;
                if(col<N&&stride*BLOCK_SIZE+threadIdx.y<K){
                        tile_B[threadIdx.y][threadIdx.x]=B[id_n];
                }
                __syncthreads();
 
                for(int i=0;i<BLOCK_SIZE;++i){
                        sum+=tile_A[threadIdx.y][i]*tile_B[i][threadIdx.x];
                }
                __syncthreads();
        }
        if(row<M&&col<N){
                C[row*N+col]=sum;
        }
}

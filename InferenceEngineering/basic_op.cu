# Conv
__global__
void conv2d(const float *input,const float *kernel,const int in_size,const int ker_size,const int stride,float *output){
 
        int row=blockIdx.y*blockDim.y+threadIdx.y;
        int col=blockIdx.x*blockDim.x+threadIdx.x;
 
        const int out_size=(in_size-ker_size)/stride+1;
 
        float sum=0;
        if(row<out_size&&col<out_size){
                for(int i=0;i<ker_size;++i){
                        for(int j=0;j<ker_size;++j){
                                sum+=input[(row+i)*in_size+(col+j)]*kernel[i*ker_size+j];
                        }
                }
                output[row*out_size+col]=sum;
        }
        __syncthreads();
}


#maxpool
__global__
void maxpool(float *input,const int in_size,const int ker_size,const int stride,float *output){
 
        int row=blockIdx.y*blockDim.y+threadIdx.y;
        int col=blockIdx.x*blockDim.x+threadIdx.x;
 
        int out_size=(in_size-ker_size)/stride+1;
 
        if(row<out_size&&col<out_size){
                float max=0;
                for(int i=0;i<ker_size;++i){
                        for(int j=0;j<ker_size;++j){
                                float curr=input[(row*stride+i)*in_size+(col*stride+j)];
                                max=max<curr?curr:max;
                        }
                }
                output[row*out_size+col]=max;
        }
}


#im2col
# im2col的原理主要就是将输入矩阵中，需要进行操作的每一个子矩阵转变为一个列向量，然后使用矩阵乘法来进行计算
__global__
void im2col(const float *input,const int in_size,const int ker_size,const int stride,float *output){
 
        int idy=blockIdx.y*blockDim.y+threadIdx.y;
        int idx=blockIdx.x*blockDim.x+threadIdx.x;
 
        int out_size=(in_size-ker_size)/stride+1;
        int width=out_size*out_size;
        int height=ker_size*ker_size;
 
        if(idx<width && idy<height){
                int row=idx/out_size;
                int col=idx%out_size;
 
                int input_row=row+(idy/ker_size);
                int input_col=col+(idy%ker_size);
 
                int input_index=input_row*in_size+input_col;
                output[idy*width+idx]=input[input_index];
        }
}

#

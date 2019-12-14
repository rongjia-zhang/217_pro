/******************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/
#define PIx2 6.2831853071795864769252867665590058f
#include <stdio.h>
#include <stdlib.h>

__global__ void ComputePhiMagGPU(int numK, const float* phiR, const float* phiI, float* phiMag){

    /********************************************************************
     *
     * Compute the magnitude of Fourier Transform at each sample point
     *
     ********************************************************************/
     int tid = threadIdx.x + blockIdx.x * blockDim.x;
     
     // INSERT KERNEL CODE HERE
     if(tid<numK){
     float real = phiR[tid];
     float imag = phiI[tid];
     phiMag[tid] = real*real + imag*imag;
    }     
}

void basicComputePhiMagGPU(int numK, float* phiR, float* phiI, float* phiMag){

     // Initialize thread block and kernel grid dimensions
     const unsigned int BLOCK_SIZE = 1024;
     dim3 DimGrid((numK-1)/BLOCK_SIZE + 1,1,1);
     dim3 DimBlock(BLOCK_SIZE,1,1);
     
     // Call the kernel for calculating magnitude of Phi
     ComputePhiMagGPU<<<DimGrid,DimBlock>>>(numK, phiR, phiI, phiMag);
}


__global__ void ComputeQGPU(int numK, int numX, const struct kValues* kVals, const float* x, const float* y, const float* z,float* Qr, float* Qi){

    /********************************************************************
     *
     * Calculate Q at each voxel point
     *
     ********************************************************************/
     int tid = threadIdx.x + blockIdx.x * blockDim.x;
     
     // INSERT KERNEL CODE HERE
     if(tid<numX){
     Qr[tid] = 0; Qi[tid] = 0;
     }
     
     // register allocate voxel inputs and outputs
     float reg_x = x[tid]; float reg_y = y[tid]; float reg_z = z[tid];
     float reg_Qr = Qr[tid]; float reg_Qi = Qi[tid];
     
     //loop over all the sample points
     for(int m = 0; m < numK; m++){
     
     // 
     float exp = 2 * PI * (kVals[m].Kx * reg_x + kVals[m].Ky * reg_y + kVals[m].Kz * reg_z);

     reg_Qr += kVals[m].PhiMag * cos(exp); reg_Qi += kVals[m].PhiMag * sin(exp);
    } 
    Qr[tid] = reg_Qr; Qi[tid] = reg_Qi;    
}

void basicComputeQGPU(int numK, int numX, struct kValues* kVals, float* x, float* y, float* z,float* Qr, float* Qi){

     // Initialize thread block and kernel grid dimensions
     const unsigned int BLOCK_SIZE = 1024;
     dim3 DimGrid((numX-1)/BLOCK_SIZE + 1,1,1);
     dim3 DimBlock(BLOCK_SIZE,1,1);
     
     // Call the kernel for calculating Q matrix
     ComputeQGPU<<<DimGrid,DimBlock>>>(numK, numX, kVals, x, y, z, Qr, Qi);     
}

/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* 
 * C code for creating the Q data structure for fast convolution-based 
 * Hessian multiplication for arbitrary k-space trajectories.
 *
 * Inputs:
 * kx - VECTOR of kx values, same length as ky and kz
 * ky - VECTOR of ky values, same length as kx and kz
 * kz - VECTOR of kz values, same length as kx and ky
 * x  - VECTOR of x values, same length as y and z
 * y  - VECTOR of y values, same length as x and z
 * z  - VECTOR of z values, same length as x and y
 * phi - VECTOR of the Fourier transform of the spatial basis 
 *      function, evaluated at [kx, ky, kz].  Same length as kx, ky, and kz.
 *
 * recommended g++ options:
 *  -O3 -lm -ffast-math -funroll-all-loops
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <malloc.h>

#include "parboil.h"
#include "file.h"
#include "computeQ.cc"

#include "kernel_reg.cu"

int 
main (int argc, char *argv[]) {
  int numX, numK; 	/* Number of X and K values */
  int original_numK;	/* Number of K values in input file */
  float *kx, *ky, *kz;		/* K trajectory (3D vectors) */
  float *x, *y, *z;		/* X coordinates (3D vectors) */

  float *x_d, *y_d, *z_d;
  float *Qr_d, *Qi_d;
  float *phiI_d, *phiR_d, *phiMag_d;
  struct kValues* kVals_d;

  float *phiR, *phiI;		/* Phi values (complex) */
  float *phiMag;		/* Magnitude of Phi */
  float *Qr, *Qi;		/* Q signal (complex) */
  struct kValues* kVals;

  struct pb_Parameters* params;
  struct pb_TimerSet timers;

  pb_InitializeTimerSet(&timers);

  /* Read command line */
  params = pb_ReadParameters(&argc, argv);
  if ((params->inpFiles[0] == NULL) || (params->inpFiles[1] != NULL))
    {

      fprintf(stderr, "Expecting one input filename\n");
      exit(-1);
    }
  
  /* Read in data */
  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  inputData(params->inpFiles[0],
	    &original_numK, &numX,
	    &kx, &ky, &kz,
	    &x, &y, &z,
	    &phiR, &phiI);

  /* Reduce the number of k-space samples if a number is given
   * on the command line */
  if (argc < 2)
    numK = original_numK;
  else
    {
      int inputK;
      char *end;
      inputK = strtol(argv[1], &end, 10);
      if (end == argv[1])
	{
	  fprintf(stderr, "Expecting an integer parameter\n");
	  exit(-1);
	}

      numK = MIN(inputK, original_numK);
    }

  printf("%d pixels in output; %d samples in trajectory; using %d samples\n",
         numX, original_numK, numK);

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  /* Create CPU data structures */
  createDataStructsCPU(numK, numX, &phiMag, &Qr, &Qi);
  
  // Allocate device variables----------------------------------------------phi
  // variables for calculating magnitude of phi 
  cudaMalloc((void**)&phiR_d,sizeof(float)*numK);
  cudaMalloc((void**)&phiI_d,sizeof(float)*numK);
  cudaMalloc((void**)&phiMag_d,sizeof(float)*numK);
  
  cudaDeviceSynchronize();
  
  // Copy host variables to device---------------------------------------------
  // variables for calculating magnitude of phi 
  cudaMemcpy(phiR_d,phiR,sizeof(float)*numK,cudaMemcpyHostToDevice);
  cudaMemcpy(phiI_d,phiI,sizeof(float)*numK,cudaMemcpyHostToDevice);
  cudaMemcpy(phiMag_d,phiMag,sizeof(float)*numK,cudaMemcpyHostToDevice);
  
  cudaDeviceSynchronize();
  
  // Launch kernel-------------------------------------------------------------
  basicComputePhiMagGPU(numK, phiR_d, phiI_d, phiMag_d);
  
  // Copu device variables to host----------------------------------------------
  cudaMemcpy(phiMag,phiMag_d,sizeof(float)*numK,cudaMemcpyDeviceToHost);
  
  // Free memory----------------------------------------------------------------
  cudaFree(phiR_d);
  cudaFree(phiI_d);
  cudaFree(phiMag_d);
  
  //--------------------------------------------------------------------------end
  
  kVals = (struct kValues*)calloc(numK, sizeof (struct kValues));
  int k;
  for (k = 0; k < numK; k++) {
    kVals[k].Kx = kx[k];
    kVals[k].Ky = ky[k];
    kVals[k].Kz = kz[k];
    kVals[k].PhiMag = phiMag[k];
  }
  
  // Allocate Device variables---------------------------------------------------Q
  // variables for calculating Q element
  cudaMalloc((void**)&x_d, sizeof(float)*numX);
  cudaMalloc((void**)&y_d, sizeof(float)*numX);
  cudaMalloc((void**)&z_d, sizeof(float)*numX);
  cudaMalloc((void**)&Qr_d, sizeof(float)*numX);
  cudaMalloc((void**)&Qi_d, sizeof(float)*numX);
  cudaMalloc((void**)&kVals_d, sizeof(struct kValues));
  
  cudaDeviceSynchronize();
  
  // Copy host variables to device------------------------------------------------
  // variables for calculating Q element
  cudaMemcpy(x_d,x,sizeof(float)*numX,cudaMemcpyHostToDevice);
  cudaMemcpy(y_d,y,sizeof(float)*numX,cudaMemcpyHostToDevice);
  cudaMemcpy(z_d,z,sizeof(float)*numX,cudaMemcpyHostToDevice);
  cudaMemcpy(Qr_d,Qr,sizeof(float)*numX,cudaMemcpyHostToDevice);
  cudaMemcpy(Qi_d,Qi,sizeof(float)*numX,cudaMemcpyHostToDevice);
  cudaMemcpy(kVals_d,kVals,sizeof(struct kValues),cudaMemcpyHostToDevice);
  
  cudaDeviceSynchronize();
  
  // Launch kernel-------------------------------------------------------------
  basicComputeQGPU(numK, numX, kVals_d, x_d, y_d, z_d, Qr_d, Qi_d);
  
  // Copu device variables to host----------------------------------------------
  cudaMemcpy(Qr,Qr_d,sizeof(float)*numX,cudaMemcpyDeviceToHost);
  cudaMemcpy(Qi,Qi_d,sizeof(float)*numX,cudaMemcpyDeviceToHost);
  
  // Free memory----------------------------------------------------------------
  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);
  cudaFree(Qr_d);
  cudaFree(Qi_d);
  cudaFree(kVals_d);
  
  //--------------------------------------------------------------------------end

  if (params->outFile)
    {
      /* Write Q to file */
      pb_SwitchToTimer(&timers, pb_TimerID_IO);
      outputData(params->outFile, Qr, Qi, numX);
      pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
    }

  free (kx);
  free (ky);
  free (kz);
  free (x);
  free (y);
  free (z);
  free (phiR);
  free (phiI);
  free (phiMag);
  free (kVals);
  free (Qr);
  free (Qi);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(params);

  return 0;
}

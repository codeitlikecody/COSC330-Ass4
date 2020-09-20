/*
 * CUDA Mandelbrot set image generator
 *
 * Author: Chris Cody September 2020
 *
 * Code from the the following sources was referenced and used:
 *   * COSC330 lectures and tutorials
 *   * "Building Parallel Programs: SMPs, Clusters And Java", Alan Kaminsky
 *   * libbmp for all bitmap operations.
 *
 *
 * This program generates bitmap images of the Mandelbrot set fractal.
 *
 * To compile use: make
 * To compile and run with a 1920x1080 image: make run
 *
 * Usage:
 * mandelbrotCUDA image_width image_height
 *
 */

// include libraries
#include <stdio.h>
#include <stdlib.h>

// Include headers
#include "bmpfile.h"

// Define constants
#define VERBOSE 1
#define SUCCESS 1
#define FAILURE 0
#define BYTES_IN_KB 1024

// Mandelbrot constants
#define RESOLUTION 8700.0	// fractal resolution/detail
#define XCENTER -0.55		// X coord center of the Mandlebrot space
#define YCENTER 0.6			// X coord center of the Mandlebrot space
#define MAX_ITER 1000		// Max iterations to calculate for each pixel
#define MIN_WIDTH 100		// min width of output image
#define MAX_WIDTH 19200		// max width of output image
#define MIN_HEIGHT 100		// min height of output image
#define MAX_HEIGHT 10800	// max height of output image
#define COLOR 3 			// three for the RGB color space

// Color and bitmap constants
#define COLOR_DEPTH 255				// color depth of each color
#define COLOR_MAX 240.0				// max color value
#define GRADIENT_COLOR_MAX 230.0	// max gradient color
#define BIT_DEPTH 32				// bit bepth of bmp file
#define FILENAME "my_mandelbrot_fractal.bmp"	// default bmp filename

// CUDA constants
#define THREADS_PER_BLOCK 128		// number of threads to start in each block

// function definitions - see below main() for function body
int parseArgs(int argc, char *argv[], int *width, int *height);

__device__ void GroundColorMix(double* color, double pixel, double min, double max);

__global__ void mandelbrot(double *pixels, const int width, const int height,
		const int numPixels, const int xoffset, const int yoffset);


/*
 * CUDA Mandelbrot generator
 *
 * This program is based on examples from COSC330 and was modified for use
 * with CUDA by Chris Cody in September 2020
 *
 * This program uses the algorithm outlined in:
 *   "Building Parallel Programs: SMPs, Clusters And Java", Alan Kaminsky
 *
 * This program requires libbmp for all bitmap operations.
 *
 * Return: void
 */
int main(int argc, char **argv) {

	// Initialize CL arg variables
	int width, height, totalPixels;

	// parse CL args
	if (!parseArgs(argc, argv, &width, &height)) {
		exit(EXIT_FAILURE);
	}

	// Calculate image size and offset
	totalPixels = width * height;
	size_t elements = totalPixels;
	size_t size = totalPixels * sizeof(double) * COLOR;
	int xoffset = -(width - 1) / 2;
	int yoffset = (height - 1) / 2;

	// generate some helpful verbose text
	if (VERBOSE) {
		fprintf(stderr, "Creating a fractal of size %ipx x %ipx\n", width,
				height);
		fprintf(stderr, "Total pixels: %i million (%i)\n",
				totalPixels / 1000000, totalPixels);
		fprintf(stderr, "Total memory required: %ld MB (%ld bytes)\n",
				size / BYTES_IN_KB / BYTES_IN_KB, size);
	}

	// Allocate and verify the host memory
	double *h_Pixels = (double *) malloc(size);
	if (h_Pixels == NULL) {
		fprintf(stderr, "Failed to allocate host memory!\n");
		exit(EXIT_FAILURE);
	}

	// Allocate and verify the device memory
	double *d_Pixels = NULL;
	cudaError_t cudaError = cudaSuccess;
	cudaError = cudaMalloc((void **) &d_Pixels, size);
	if (cudaError != cudaSuccess) {
		fprintf(stderr,
				"Could not allocate memory on the CUDA device. Error:%s\n",
				cudaGetErrorString(cudaError));
		exit(EXIT_FAILURE);
	}

	// Launch the Mandelbrot CUDA Kernel
	int blocksPerGrid = (elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	if (VERBOSE) {
		fprintf(stderr, "CUDA kernel launched with %d blocks of %d threads\n",
				blocksPerGrid,
				THREADS_PER_BLOCK);
	}

	mandelbrot<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_Pixels, width, height,
			totalPixels, xoffset, yoffset);

	// Copy data back from device
	if (VERBOSE) {
		fprintf(stderr, "Copying image data back from the CUDA device\n");
	}

	cudaError=cudaMemcpy(h_Pixels, d_Pixels, size, cudaMemcpyDeviceToHost);
	if (cudaError != cudaSuccess) {
			fprintf(stderr,
					"Could not copy image data back from the CUDA device. Error:%s\n",
				cudaGetErrorString(cudaError));
		exit(EXIT_FAILURE);
	}

	// Initialize the bitmap variables
	bmpfile_t *bmp;
	rgb_pixel_t pixel = { 0, 0, 0, 0 };
	bmp = bmp_create(width, height, BIT_DEPTH);
	int col, row;

	if (VERBOSE) {
		fprintf(stderr, "Generating bitmap\n");
	}

	// generate bitmap image
	for (int i = 0; i < totalPixels; i++) {
		col = i % (width - 1);
		row = i / (width - 1);
		pixel.red = h_Pixels[i];
		pixel.green = h_Pixels[i + totalPixels];
		pixel.blue = h_Pixels[i + totalPixels * 2];
		bmp_set_pixel(bmp, col, row, pixel);
	}

	// save bitmap
	bmp_save(bmp, FILENAME);
	bmp_destroy(bmp);

	fprintf(stderr, "Complete!\nFractal saved to file: ./%s\n", FILENAME);

	// free memory and exit
	free(h_Pixels);
	cudaError=cudaFree(d_Pixels);
		if (cudaError!= cudaSuccess) {
				fprintf(stderr,
						"Could not free memory on the CUDA device. Error:%s\n",
					cudaGetErrorString(cudaError));
			exit(EXIT_FAILURE);
		}
	exit(EXIT_SUCCESS);
}

/*
 * Parse and validate command line args
 *
 * Arguments:
 * int argc - number of arguments supplied
 * char *argv[] - array of command line arguments
 * int *width - width of output file
 * int *height - height of output file
 *
 * Return: FAILURE or SUCCESS values defined above
 */
int parseArgs(int argc, char *argv[], int *width, int *height) {
	// validate the number of args
	if (argc == 3) { // parse and validate args

		// parse and validate the image width
		if ((*width = atoi(argv[1])) < MIN_WIDTH || *width > MAX_WIDTH) {
			fprintf(stderr, "Width of image must be between %i and %i\n",
			MIN_WIDTH, MAX_WIDTH);
			return (FAILURE);
		}

		// parse and validate the image height
		if ((*height = atoi(argv[2])) < MIN_HEIGHT || *height > MAX_HEIGHT) {
			fprintf(stderr, "Height of image must be between %i and %i\n",
			MIN_HEIGHT, MAX_HEIGHT);
			return (FAILURE);
		}

	} else { // incorrect number of args supplied
		fprintf(stderr, "Usage: %s image_width image_height\n", argv[0]);
		return (FAILURE);
	}

	return (SUCCESS);
}

/*
 * Computes the RGB pixel values for Mandelbrot set
 *
 * Arguments:
 * double *pixels - array to store the image result in.
 * 		Stores all red pixels first, followed by blue, then green.
 * 		Size must be: 3 * numPixels * sizeof(double)
 * const int width - width of image to output
 * const int height - height of image to output
 * const int numPixels - total number of pixels in the image
 * const int xoffset - x offset (x center of image)
 * const int yoffset - y offset (y center of image)
 *
 * Return: void
 */
__global__ void mandelbrot(double *pixels, const int width, const int height,
		const int numPixels, const int xoffset, const int yoffset) {

	// calculate starting pixel for this thread
	long i = blockDim.x * blockIdx.x + threadIdx.x;

	// calculate Mandelbrot set for all pixels assigned to this thread
	while (i < numPixels) {

		// work out the x/y position of this pixel
		int xPosition = i % (width - 1);
		int yPosition = i / (width - 1);

		// determine where in the Mandelbrot set the pixel is referencing
		double x = XCENTER + (xoffset + xPosition) / RESOLUTION;
		double y = YCENTER + (yoffset - yPosition) / RESOLUTION;

		// define variables for Mandelbrot calculation
		double a = 0;
		double b = 0;
		double aold = 0;
		double bold = 0;
		double zmagsqr = 0;
		int iter = 0;
		double pixel;

		// check if the x/y coord are part of the Mandelbrot set - refer to the algorithm
		while (iter < MAX_ITER && zmagsqr <= 4.0) {
			++iter;
			a = (aold * aold) - (bold * bold) + x;
			b = 2.0 * aold * bold + y;

			zmagsqr = a * a + b * b;

			aold = a;
			bold = b;

			pixel =
					(COLOR_MAX
							- ((((float) iter / ((float) MAX_ITER)
									* GRADIENT_COLOR_MAX))));
		}

		// Set the color of the pixel
		double color[3];
		GroundColorMix(color, pixel, 1, COLOR_DEPTH);
		for (int y = 0; y < COLOR; y++) {
			pixels[i + y * numPixels] = color[y];
		}

		// iterate to next pixel
		i += blockDim.x * gridDim.x;
	}
}

/*
 * Computes the color gradient for each pixel in the Mandelbrot image
 *
 * Arguments:
 * double* color - location to store color values.
 * double pixel - value of pixel gradient (from 0 to 360)
 * double min - min color value for each pixel
 * double max - max color value for each pixel
 *
 * Check wiki for more details on the colour science: en.wikipedia.org/wiki/HSL_and_HSV
 *
 * Return: void
 */
__device__ void GroundColorMix(double* color, double pixel, double min,
		double max) {
	/*
	 * Red = 0
	 * Green = 1
	 * Blue = 2
	 */
	double posSlope = (max - min) / 60;
	double negSlope = (min - max) / 60;

	if (pixel < 60) {
		color[0] = max;
		color[1] = posSlope * pixel + min;
		color[2] = min;
		return;
	} else if (pixel < 120) {
		color[0] = negSlope * pixel + 2.0 * max + min;
		color[1] = max;
		color[2] = min;
		return;
	} else if (pixel < 180) {
		color[0] = min;
		color[1] = max;
		color[2] = posSlope * pixel - 2.0 * max + min;
		return;
	} else if (pixel < 240) {
		color[0] = min;
		color[1] = negSlope * pixel + 4.0 * max + min;
		color[2] = max;
		return;
	} else if (pixel < 300) {
		color[0] = posSlope * pixel - 4.0 * max + min;
		color[1] = min;
		color[2] = max;
		return;
	} else {
		color[0] = max;
		color[1] = min;
		color[2] = negSlope * pixel + 6 * max;
		return;
	}
}


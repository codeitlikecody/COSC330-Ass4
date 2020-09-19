#include <stdio.h>
#include <stdlib.h>
#include "bmpfile.h"

// Define constants
#define VERBOSE 1
#define SUCCESS 1
#define FAILURE 0

// Mandelbrot constants
#define RESOLUTION 8700.0	// fractal resolution/detail
#define XCENTER -0.55
#define YCENTER 0.6
#define MAX_ITER 1000
#define MIN_WIDTH 100		// min width of output image
#define MAX_WIDTH 19200		// max width of output image
#define MIN_HEIGHT 100		// min height of output image
#define MAX_HEIGHT 10800	// max height of output image
#define COLOURS 3 			// three for the RGB colour space
#define FILENAME "my_mandelbrot_fractal.bmp"
#define BYTES_IN_KB 1024

// Colour constants
#define BIT_DEPTH 32
#define COLOUR_DEPTH 255
#define COLOUR_MAX 240.0
#define GRADIENT_COLOUR_MAX 230.0

// CUDA constants
#define THREADS_PER_BLOCK 128

// function definitions - see below main() for function body
int parseArgs(int argc, char *argv[], int *width, int *height);

void GroundColorMix(double* color, double x, double min, double max);

__global__ void mandelbrot(double *pixels, const int width, const int height,
		const int numPixels, const int xoffset, const int yoffset);

/* Mandelbrot Set Image Demonstration
 *
 * This is a simple single-process/single thread implementation
 * that computes a Mandelbrot set and produces a corresponding
 * Bitmap image. The program demonstrates the use of a colour
 * gradient
 *
 * This program uses the algorithm outlined in:
 *   "Building Parallel Programs: SMPs, Clusters And Java", Alan Kaminsky
 *
 * This program requires libbmp for all bitmap operations.
 *
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
	size_t size = totalPixels * sizeof(double);
	int xoffset = -(width - 1) / 2;
	int yoffset = (height - 1) / 2;

	if (VERBOSE) {
		fprintf(stderr, "Creating a fractal of size %ipx x %ipx\n", width, height);
		fprintf(stderr, "Total pixels: %i million (%i)\n",totalPixels/1000000, totalPixels);
		fprintf(stderr, "Total memory required: %ld MB (%ld bytes)\n", size/BYTES_IN_KB/BYTES_IN_KB, size);
	}

	// Allocate the host array
	double *h_Pixels = (double *) malloc(size);

	// Allocate the device array
	double *d_Pixels = NULL;
	cudaMalloc((void **) &d_Pixels, size);

	// Launch the VectorAdd CUDA Kernel
	int blocksPerGrid = (elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	if (VERBOSE) {
		fprintf(stderr, "CUDA kernel launch with %d blocks of %d threads\n",
				blocksPerGrid,
				THREADS_PER_BLOCK);
	}

	mandelbrot<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_Pixels, width, height,
			totalPixels,xoffset,yoffset);

	// Copy data back from device
	if (VERBOSE) {
		fprintf(stderr,
				"Copy output data from the CUDA device to the host memory\n");
	}

	cudaMemcpy(h_Pixels, d_Pixels, size, cudaMemcpyDeviceToHost);

	// Initialize the bitmap variables
	bmpfile_t *bmp;
	rgb_pixel_t pixel = { 0, 0, 0, 0 };
	bmp = bmp_create(width, height, BIT_DEPTH);
	double color[3];
	int col,row;

	// colour pixels based on Mandelbrot values
	if (VERBOSE) {
		fprintf(stderr,
				"Colouring pixels and generating bitmap\n");
	}

	for (int i = 0; i < totalPixels; i++) {

		col = i % (width-1);
		row = i / (width-1);
		GroundColorMix(color, h_Pixels[i], 1, COLOUR_DEPTH);
		pixel.red = color[0];
		pixel.green = color[1];
		pixel.blue = color[2];
		bmp_set_pixel(bmp, col, row, pixel);
	}


	//save bitmap
	bmp_save(bmp, FILENAME);
	bmp_destroy(bmp);

	fprintf(stderr, "Complete!\nFractal saved to file: ./%s\n", FILENAME);


	// Free memory and exit
	cudaFree(d_Pixels);
	free(h_Pixels);
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

/**
 * Computes the color gradient
 * color: the output vector
 * x: the gradient (between 0 and 360)
 * min and max: variation of the RGB channels (Move3D 0 -> 1)
 * Check wiki for more details on the colour science: en.wikipedia.org/wiki/HSL_and_HSV
 */
void GroundColorMix(double* color, double x, double min, double max) {
	/*
	 * Red = 0
	 * Green = 1
	 * Blue = 2
	 */
	double posSlope = (max - min) / 60;
	double negSlope = (min - max) / 60;

	if (x < 60) {
		color[0] = max;
		color[1] = posSlope * x + min;
		color[2] = min;
		return;
	} else if (x < 120) {
		color[0] = negSlope * x + 2.0 * max + min;
		color[1] = max;
		color[2] = min;
		return;
	} else if (x < 180) {
		color[0] = min;
		color[1] = max;
		color[2] = posSlope * x - 2.0 * max + min;
		return;
	} else if (x < 240) {
		color[0] = min;
		color[1] = negSlope * x + 4.0 * max + min;
		color[2] = max;
		return;
	} else if (x < 300) {
		color[0] = posSlope * x - 4.0 * max + min;
		color[1] = min;
		color[2] = max;
		return;
	} else {
		color[0] = max;
		color[1] = min;
		color[2] = negSlope * x + 6 * max;
		return;
	}
}

/**
 * Computes the mandelbrot stuff
 */
__global__ void

mandelbrot(double *pixels, const int width, const int height,
		const int numPixels, const int xoffset, const int yoffset) {

	// calculate first pixel for this thread
	long i = blockDim.x * blockIdx.x + threadIdx.x;

	// calculate mandelbrot set for all pixels assigned to this thread
	while (i < numPixels) {

		// work out the x/y position of this pixel
		int xPosition = i % (width-1);
		int yPosition = i / (width-1);

		//Determine where in the mandelbrot set, the pixel is referencing
		double x = XCENTER + (xoffset + xPosition) / RESOLUTION;
		double y = YCENTER + (yoffset - yPosition) / RESOLUTION;

		//Mandelbrot stuff

		double a = 0;
		double b = 0;
		double aold = 0;
		double bold = 0;
		double zmagsqr = 0;
		int iter = 0;

		//Check if the x,y coord are part of the mendelbrot set - refer to the algorithm
		while (iter < MAX_ITER && zmagsqr <= 4.0) {
			++iter;
			a = (aold * aold) - (bold * bold) + x;
			b = 2.0 * aold * bold + y;

			zmagsqr = a * a + b * b;

			aold = a;
			bold = b;

			pixels[i] =
					(COLOUR_MAX
							- ((((float) iter / ((float) MAX_ITER)
									* GRADIENT_COLOUR_MAX))));
		}

		// iterate to next pixel
		i += blockDim.x * gridDim.x;
	}
}

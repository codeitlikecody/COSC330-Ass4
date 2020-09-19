#include <stdio.h>
#include <stdlib.h>
#include "bmpfile.h"

// Define constants
#define VERBOSE 1
#define FILENAME "my_mandelbrot_fractal.bmp"

/*Mandelbrot values*/
#define RESOLUTION 8700.0
#define XCENTER -0.55
#define YCENTER 0.6
#define MAX_ITER 1000
#define MIN_WIDTH 100
#define MAX_WIDTH 19200
#define MIN_HEIGHT 100
#define MAX_HEIGHT 10800

// Define success and failure return values
#define SUCCESS 1
#define FAILURE 0

/*Colour Values*/
#define COLOUR_DEPTH 255
#define COLOUR_MAX 240.0
#define GRADIENT_COLOUR_MAX 230.0

// function definitions - see below main() for function body
int parseArgs(int argc, char *argv[], int *width, int *height);

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

	int width, height;

	// parse CL args
	if (!parseArgs(argc, argv, &width, &height)) {
		exit(EXIT_FAILURE);
	}

	fprintf(stderr, "Creating a fractal: %ipx x %ipx\n",width, height);


	bmpfile_t *bmp;
	rgb_pixel_t pixel = { 0, 0, 0, 0 };
	int xoffset = -(width - 1) / 2;
	int yoffset = (height - 1) / 2;
	bmp = bmp_create(width, height, 32);
	int col = 0;
	int row = 0;
	for (col = 0; col < width; col++) {
		for (row = 0; row < height; row++) {

			//Determine where in the mandelbrot set, the pixel is referencing
			double x = XCENTER + (xoffset + col) / RESOLUTION;
			double y = YCENTER + (yoffset - row) / RESOLUTION;

			//Mandelbrot stuff

			double a = 0;
			double b = 0;
			double aold = 0;
			double bold = 0;
			double zmagsqr = 0;
			int iter = 0;
			double x_col;
			double color[3];
			//Check if the x,y coord are part of the mendelbrot set - refer to the algorithm
			while (iter < MAX_ITER && zmagsqr <= 4.0) {
				++iter;
				a = (aold * aold) - (bold * bold) + x;
				b = 2.0 * aold * bold + y;

				zmagsqr = a * a + b * b;

				aold = a;
				bold = b;

			}

			/* Generate the colour of the pixel from the iter value */
			/* You can mess around with the colour settings to use different gradients */
			/* Colour currently maps from royal blue to red */
			x_col =
					(COLOUR_MAX
							- ((((float) iter / ((float) MAX_ITER)
									* GRADIENT_COLOUR_MAX))));
			GroundColorMix(color, x_col, 1, COLOUR_DEPTH);
			pixel.red = color[0];
			pixel.green = color[1];
			pixel.blue = color[2];
			bmp_set_pixel(bmp, col, row, pixel);

		}

	}

	bmp_save(bmp, FILENAME);
	bmp_destroy(bmp);

	fprintf(stderr, "Complete!\nFractal saved to file: ./%s\n",FILENAME);

	return 0;
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

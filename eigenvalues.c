#include "return_codes.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EPS 1e-8
typedef unsigned int uint;

double scalar_product(const double *vector1, const double *vector2, uint n);
double norm(const double *vector, uint n);
void qr_algorithm(double *matrix, uint n);
void multiply_matrix(const double *matrix1, const double *matrix2, uint n, double *result);
void transpose(double *matrix, uint n);
void gram_schmidt(const double *matrix, double *result, uint n);
void malloc_check(void *cell);

void malloc_check(void *cell)
{
	if (cell == NULL)
	{
		fprintf(stderr, "Error: not enough memory");
		exit(ERROR_OUT_OF_MEMORY);
	}
}

double scalar_product(const double *vector1, const double *vector2, const uint n)
{
	double result = 0;
	for (int i = 0; i < n; i++)
	{
		result += vector1[i] * vector2[i];
	}
	return result;
}

double norm(const double *vector, const uint n)
{
	double result = 0;
	for (int i = 0; i < n; i++)
	{
		result += vector[i] * vector[i];
	}
	result = sqrt(result);
	return result;
}

void qr_algorithm(double *matrix, const uint n)
{
	double *Q = (double *)malloc(n * n * sizeof(double));
	malloc_check(Q);
	gram_schmidt(matrix, Q, n);
	double *tempQ = (double *)malloc(n * n * sizeof(double));
	malloc_check(tempQ);
	memcpy(tempQ, Q, n * n * sizeof(double));
	transpose(tempQ, n);
	double *R = (double *)calloc(n * n, sizeof(double));
	malloc_check(R);
	multiply_matrix(tempQ, matrix, n, R);
	free(tempQ);
	multiply_matrix(R, Q, n, matrix);
	free(Q);
	free(R);
}

void multiply_matrix(const double *matrix1, const double *matrix2, const uint n, double *result)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			result[i * n + j] = 0;
			for (int k = 0; k < n; k++)
			{
				result[i * n + j] += matrix1[i * n + k] * matrix2[k * n + j];
			}
		}
	}
}

void transpose(double *matrix, const uint n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = i + 1; j < n; j++)
		{
			double temp = matrix[i * n + j];
			matrix[i * n + j] = matrix[j * n + i];
			matrix[j * n + i] = temp;
		}
	}
}

void gram_schmidt(const double *matrix, double *result, const uint n)
{
	double *b_vectors = (double *)malloc(n * n * sizeof(double));
	malloc_check(b_vectors);
	double *a_vector = (double *)malloc(n * sizeof(double));
	malloc_check(a_vector);
	double *proj = (double *)malloc(n * n * sizeof(double));
	malloc_check(proj);
	for (int k = 0; k < n; k++)
	{
		for (int i = 0; i < n; i++)
		{
			a_vector[i] = matrix[i * n + k];
			b_vectors[k * n + i] = a_vector[i];
		}
		for (int j = 0; j < k; j++)
		{
			double x = scalar_product(a_vector, &b_vectors[j * n], n);
			double y = scalar_product(&b_vectors[j * n], &b_vectors[j * n], n);
			for (int i = 0; i < n; i++)
			{
				if (fabs(y) < EPS)
				{
					proj[j * n + i] = 0;
					continue;
				}
				proj[j * n + i] = x / y * b_vectors[j * n + i];
				b_vectors[k * n + i] -= proj[j * n + i];
			}
		}
		double normal = norm(&b_vectors[k * n], n);
		if (fabs(normal) < EPS)
		{
			normal = 1;
		}
		for (int i = 0; i < n; i++)
		{
			result[k + n * i] = b_vectors[k * n + i] / normal;
		}
	}
	free(b_vectors);
	free(a_vector);
	free(proj);
}

int print_answer(const double *matrix, const uint n, const char *filename)
{
	FILE *file = fopen(filename, "w");
	if (!file)
	{
		fprintf(stderr, "Cannot open file: %s\n", filename);
		return ERROR_CANNOT_OPEN_FILE;
	}

	for (int i = 0; i < n; i++)
	{
		if (fabs(matrix[(i + 1) * n + i]) < EPS)
		{
			fprintf(file, "%g\n", matrix[i * n + i]);
		}
		else
		{
			double m = (matrix[i * n + i] + matrix[(i + 1) * n + i + 1]) / 2;
			double d = (matrix[i * n + i] * matrix[(i + 1) * n + i + 1]) - (matrix[(i + 1) * n + i] * matrix[i * n + i + 1]);
			double imag = sqrt(fabs(m * m - d));
			i++;
			fprintf(file, "%g +%gi\n", m, imag);
			fprintf(file, "%g -%gi\n", m, imag);
		}
	}
	fclose(file);
	return SUCCESS;
}

int read_matrix(const char *filename, double **matrix, uint *n)
{
	FILE *file = fopen(filename, "rb");
	if (!file)
	{
		fprintf(stderr, "Cannot open file: %s\n", filename);
		return ERROR_CANNOT_OPEN_FILE;
	}

	if (fscanf(file, "%d", n) != 1)
	{
		fclose(file);
		fprintf(stderr, "Invalid data\n");
		return ERROR_DATA_INVALID;
	}

	*matrix = (double *)malloc(sizeof(double) * (*n) * (*n));
	if (!*matrix)
	{
		fclose(file);
		fprintf(stderr, "Out of memory\n");
		return ERROR_OUT_OF_MEMORY;
	}

	for (int i = 0; i < *n; ++i)
	{
		for (int j = 0; j < *n; ++j)
		{
			if (fscanf(file, "%lf", &(*matrix)[i * (*n) + j]) != 1)
			{
				fclose(file);
				free(*matrix);
				fprintf(stderr, "Invalid data\n");
				return ERROR_DATA_INVALID;
			}
		}
	}
	fclose(file);
	return SUCCESS;
}

int main(int argc, char **argv)
{
	if (argc != 3)
	{
		fprintf(stderr, "Not 3 arguments\n");
		return ERROR_PARAMETER_INVALID;
	}
	double *matrix;
	uint n;
	int result = read_matrix(argv[1], &matrix, &n);
	if (result != SUCCESS)
	{
		fprintf(stderr, "Error: %d\n", result);
		return result;
	}
	for (int i = 0; i < 2451; i++)
	{
		qr_algorithm(matrix, n);
	}
	result = print_answer(matrix, n, argv[2]);
	if (result != SUCCESS)
	{
		fprintf(stderr, "Error: %d\n", result);
		free(matrix);
		return result;
	}
	free(matrix);
	return SUCCESS;
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern struct {
    float *biases[2];
    float **weights[2];
    int nb_neurons[2];
} net;

static float dot(float *v1, float *v2, int size)
{
    int i;
    float sum = 0.0f;
    for (i = 0; i < size; i++)
    {
        sum += v1[i] * v2[i];
    }
    return sum;
}

static float sigmoid(float v)
{
    return 1.0f / (1.0f + exp(-v));
}

static void compute(float *input, int input_size, float *output)
{
    int l;

    float *prev_activation = malloc(input_size * sizeof(float));
    for(int i = 0; i < input_size; i++)
        prev_activation[i] = input[i];
    int prev_activation_size = input_size;

    for(l = 0; l < 2; l++)
    {
        float *lb = net.biases[l];
        float **lw = net.weights[l];

        int n;
        float *activation = malloc(net.nb_neurons[l] * sizeof(float));
        for(n=0; n < net.nb_neurons[l]; n++)
        {
            /* compute activation */
            activation[n] = sigmoid(dot(lw[n], prev_activation, prev_activation_size) + lb[n]);
        }
        free(prev_activation);
        prev_activation = activation;
        prev_activation_size = net.nb_neurons[l];
    }

    for(int i = 0; i < net.nb_neurons[1]; i++)
        output[i] = prev_activation[i];
    free(prev_activation);
}

static int read_int(FILE *f)
{
    unsigned char v[4];
    fread(v,4,1,f);
    return (v[0] << 24) + (v[1] << 16) + (v[2] << 8) + v[3];
}

static int max_index(float *output)
{
    float max = -10000.0f;
    int index = 0;
    for (int i = 0; i < 10; i++)
    {
        if (output[i] > max)
        {
            max = output[i];
            index = i;
        }
    }
    return index;
}

int main()
{
    FILE *fi = fopen("t10k-images.idx3-ubyte","rb");
    FILE *fl = fopen("t10k-labels.idx1-ubyte", "rb");

    fseek(fi, 4, SEEK_SET);
    fseek(fl, 8, SEEK_SET);

    int nb_samples = read_int(fi);
    int rows = read_int(fi);
    int cols = read_int(fi);

    unsigned char *labels = malloc(nb_samples);
    fread(labels, 1, nb_samples, fl);

    unsigned char *img = malloc(rows * cols * sizeof(char));
    float *input = malloc(rows * cols * sizeof(float));
    float *output = malloc(10 * sizeof(float));

    int nb_success = 0;
    for (int i = 0; i < nb_samples; i++)
    {
        fread(img, 1, rows * cols, fi);
        for(int j = 0; j < rows * cols; j++)
            input[j] = (float)img[j] / 255.0f;
        compute(input, rows * cols, output);
        if (max_index(output) == labels[i])
            nb_success++;
    }
    printf("%d / %d\n", nb_success, nb_samples);
    fclose(fi);
    fclose(fl);
    free(labels);
    return 0;
}
#include <stdio.h>
#include <string.h>
#include "aod3d.h"

bool init_kernel(som_net *net)
{
  net->kernel_size = 3;
  int kernel_size = net->kernel_size;
  net->kernel = (float **)malloc(kernel_size * sizeof(float *));
  if (net->kernel == NULL)
    return false;
  for (int i = 0; i < kernel_size; i++) {
    net->kernel[i] = (float *)malloc(kernel_size * sizeof(float));
    if (net->kernel[i] == NULL)
      return false;
  }
  if (kernel_size%2 == 0) {
    printf("Kernel must be odd\n");
    return false;
  }
  float sigma = 1.0; //TODO: move to parameters
  int x = (-1)*kernel_size/(int)2;
  int y = x;
  for (int i = 0; i < kernel_size; i++)
    for (int j = 0; j < kernel_size; j++) {
      net->kernel[i][j] = gauss(sigma, x++, y++);
    }
}

bool init_net(som_net *net)
{
  init_kernel(net);
  int size = net->mt_n_rows*net->mt_n_cols*net->mt_n_layers*sizeof(rgb_pixel);
  rgb_pixel *m = NULL;
  m = (rgb_pixel *)malloc(size);
  if (m == NULL)
    return false;
  net->mt = m;
  int img_n_rows = net->sm_n_rows;
  int img_n_cols = net->sm_n_cols;
  for (int l = 0; l < net->mt_n_layers; l++)
    for (int i = 0; i < img_n_rows; i++) {
      int ii = i + net->offset;
      for (int j = 0; j < img_n_cols; j++) {
        int jj = j + net->offset;
        //TODO: sort out init_img[i, j, 0, n_cols, 3];
        m[ind3d(ii, jj, l, net->mt_n_cols, net->mt_n_layers)].r =\
                                  net->init_img[ind3d(i, j, 0, img_n_cols, 3)];
        m[ind3d(ii, jj, l, net->mt_n_cols, net->mt_n_layers)].g =\
                                  net->init_img[ind3d(i, j, 1, img_n_cols, 3)];
        m[ind3d(ii, jj, l, net->mt_n_cols, net->mt_n_layers)].b =\
                                  net->init_img[ind3d(i, j, 2, img_n_cols, 3)];
      }
    }
}

void get_net_state(unsigned char *data, int layer, som_net *net)
{
  float *img = (float *)data;
  int l = layer;
  int mt_n_layers = net->mt_n_layers;
  rgb_pixel *m = net->mt;
  for (int i = 0; i < net->sm_n_rows; i++) {
    int ii = i + net->offset;
    for (int j = 0; j < net->sm_n_cols; j++) {
      int jj = j + net->offset;
      img[ind3d(i, j, 0, net->sm_n_cols, 3)] =\
                            m[ind3d(ii, jj, l, net->mt_n_cols, mt_n_layers)].r;
      img[ind3d(i, j, 1, net->sm_n_cols, 3)] =\
                            m[ind3d(ii, jj, l, net->mt_n_cols, mt_n_layers)].g;
      img[ind3d(i, j, 2, net->sm_n_cols, 3)] =\
                            m[ind3d(ii, jj, l, net->mt_n_cols, mt_n_layers)].b;
    }
  }
}

inline static void delete_foregr_pixel(som_net *net, int i, int j, int n_cols)
{
  net->fg[ind2d(i, j, n_cols)] = 0;
}

inline static void delete_sm_pixel(som_net *net, int i, int j)
{
  net->sm[ind2d(i, j, net->sm_n_cols)] = 0;
}

static bool exists_ms_pixel(som_net *net, int i, int j)
{
  if (net->sm[ind2d(i, j, net->sm_n_cols)] == 1)
    return true;
  return false;
}

void set_background(som_net *net, int i, int j, int ii, int jj, int l_hit)
{
  net->bg[ind3d(i, j, 0, net->sm_n_cols, 3)] =\
             net->mt[ind3d(ii, jj, l_hit, net->mt_n_cols, net->mt_n_layers)].r;
  net->bg[ind3d(i, j, 1, net->sm_n_cols, 3)] =\
             net->mt[ind3d(ii, jj, l_hit, net->mt_n_cols, net->mt_n_layers)].g;
  net->bg[ind3d(i, j, 2, net->sm_n_cols, 3)] =\
             net->mt[ind3d(ii, jj, l_hit, net->mt_n_cols, net->mt_n_layers)].b;
}

void update_weights2d(som_net *net, float i_r, float i_g, float i_b, int ii,
                      int jj, int l_hit)
{
  for (int n = 0; n < net->kernel_size; n++) {
    for (int m = 0; m < net->kernel_size; m++) {
      float a = net->alpha * net->kernel[n][m];
      int i_ind = ii - net->offset + n;
//      int j_ind = jj - net->offset + n; //FIXME ?
      int j_ind = jj - net->offset + m; //FIXME ?
      int ind = ind3d(i_ind, j_ind, l_hit, net->mt_n_cols, net->mt_n_layers);
      net->mt[ind].r += a*(i_r - net->mt[ind].r);
      net->mt[ind].g += a*(i_g - net->mt[ind].g);
      net->mt[ind].b += a*(i_b - net->mt[ind].b);
    }
  }
}

inline static void update_weights1d_(som_net *net, float i_r, float i_g,
                                     float i_b, int i, int j, int l, int i_k)
{
  float a = net->alpha1d * net->kernel1d[i_k];
  int ind = ind3d(i, j, l, net->mt_n_cols, net->mt_n_layers);
  net->mt[ind].r += a*(i_r - net->mt[ind].r);
  net->mt[ind].g += a*(i_g - net->mt[ind].g);
  net->mt[ind].b += a*(i_b - net->mt[ind].b);
}

inline static void update_weights1d(som_net *net, float i_r, float i_g,
                                    float i_b, int i, int j, int l_hit)
{
  int i_k;
  for (int l = l_hit, i_k = 0; l < net->mt_n_layers; l++, i_k++)
    update_weights1d_(net, i_r, i_g, i_b, i, j, l, i_k);
  for (int l = l_hit, i_k = 0; l >= 0; l--, i_k++)
    update_weights1d_(net, i_r, i_g, i_b, i, j, l, i_k);
}

static inline void set_foreground(som_net *net, int i, int j)
{
  net->fg[ind2d(i, j, net->sm_n_cols)] = 1;
}

static float dst(som_net *net, float *img, int i, int j, int ii, int jj,
                 float *i_r, float *i_g, float *i_b, int *l_hit)
{
  float d_min = FLT_MAX;
  for (int l = 0; l < net->mt_n_layers; l++) {
    float r = net->mt[ind3d(ii, jj, l, net->mt_n_cols, net->mt_n_layers)].r;
    float g = net->mt[ind3d(ii, jj, l, net->mt_n_cols, net->mt_n_layers)].g;
    float b = net->mt[ind3d(ii, jj, l, net->mt_n_cols, net->mt_n_layers)].b;
    *i_r = img[ind3d(i, j, 0, net->sm_n_cols, 3)];
    *i_g = img[ind3d(i, j, 1, net->sm_n_cols, 3)];
    *i_b = img[ind3d(i, j, 2, net->sm_n_cols, 3)];
    float d = dst_bgr(r, g, b, *i_r, *i_g, *i_b);
    if (d < d_min) {
      d_min = d;
      *l_hit = l;
    }
  }
  return d_min;
}

static void update_weights(som_net *net, float i_r, float i_g, float i_b,
                           int ii, int jj, int l_hit)
{
  update_weights2d(net, i_r, i_g, i_b, ii, jj, l_hit);
  update_weights1d(net, i_r, i_g, i_b, ii, jj, l_hit);
}

void update(float *img, som_net *net)
{
  float tau = 60;
  float k_b = 1;
  float k_a = 0.1;
  float k_c = tau;
  int   l_hit = 0;
  float i_r, i_g, i_b;

  for (int i = 0; i < net->sm_n_rows; i++) { // sm_n_rows same as image n rows
    int ii = i + net->offset;
    for (int j = 0; j < net->sm_n_cols; j++) { // sm_n_cols same as image n cols 
      int jj = j + net->offset;
      float d_min = dst(net, img, i, j, ii, jj, &i_r, &i_g, &i_b, &l_hit);
      if (d_min <= net->epsilon) {                          // background pixel
        update_weights(net, i_r, i_g, i_b, ii, jj, l_hit);
        set_background(net, i, j, ii, jj, l_hit);
        delete_foregr_pixel(net, i, j, net->sm_n_cols); // TODO: remove parameter img_n_cols
        net->count[i][j] = MAX(0, (net->count[i][j] - k_a));
        if (net->count[i][j] == 0)
          delete_sm_pixel(net, i, j);
      }
      else if (exists_ms_pixel(net, i, j) == true) {        // old stopped pixel
        delete_foregr_pixel(net, i, j, net->sm_n_cols);
        net->count[i][j] = 0;
      }
      else if (net->fg[ind2d(i, j, net->sm_n_cols)] == 1) { // old moving pixel
        net->count[i][j] = MIN(tau, net->count[i][j] + k_b);
      }
      else {                                                // new moving pixel
        net->count[i][j] = MAX(1, net->count[i][j] - k_c);
        set_foreground(net, i, j);
      }
      if (net->count[i][j] == tau) {                        // new stopped pixel
        delete_foregr_pixel(net, i, j, net->sm_n_cols);
        net->count[i][j] = 0;
        net->sm[ind2d(i, j, net->sm_n_cols)] = 1;
      }
    }
  }
}

void destroy_net(som_net *net)
{
  if (net->mt != NULL)
    free(net->mt);
  if (net->kernel != NULL) {
    for (int i = 0; i < net->kernel_size; i++) {
      if (net->kernel[i] != NULL)
        free(net->kernel[i]);
    }
    free(net->kernel);
  }
}


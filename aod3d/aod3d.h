/*
 * Abandoned objects realization according to:
 * Stopped Object Detection by Learning Foreground Model in Videos
 * IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 24, NO. 5,
 * MAY 2013
 * Lucia Maddalena, Member, IEEE, and Alfredo Petrosino, Senior Member, IEEE
 */

#ifndef _AOD3D_H_
#define _AOD3D_H_

#include <stdbool.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

#define M_PI acos(-1.0)

typedef struct
{
  float r, g, b;
} rgb_pixel;

typedef struct
{
  int        mt_n_rows;
  int        mt_n_cols;
  int        mt_n_layers;
  int        sm_n_rows;
  int        sm_n_cols;
  int        offset;
  int        kernel_size;
  float      epsilon;
  float      alpha;
  float      alpha1d;
  float     *fg;
  float     *bg;
  float     *init_img;
  rgb_pixel *mt; // mt is the network name in the article
  float     *sm;
  float    **count;
  float    **kernel; // Gaussian kernel
  float     *kernel1d; // kernel for updates along layers
} som_net;

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

inline int ind2d(int i_row, int i_col, int num_cols)
{
  return num_cols*i_row + i_col;
}

inline int ind3d(int i_row, int i_col, int i_dept, int num_cols,
                 int num_layers)
{
  return i_dept + num_layers*(i_col + num_cols*i_row);
}

inline float dst_bgr(float b0, float g0, float r0,
                     float b1, float g1, float r1)
{
  return sqrt(pow(b0 - b1, 2) + pow(g0 - g1, 2) + pow(r0 - r1, 2));
}

inline float gauss(float sigma, float x, float y)
{
  return (1.0/(2*M_PI*sigma*sigma))*exp((-1)*((x*x + y*y)/(2*sigma*sigma)));
}

bool init_net(som_net *net);
void update(float *img, som_net *net);
void destroy_net();

#endif


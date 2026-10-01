#ifndef H3_LORA_H
#define H3_LORA_H

#include "h3_gpu.h"
#include "h3_safetensors.h"
#include <stddef.h>
#include <stdint.h>

typedef struct h3_lora h3_lora;
typedef struct {
    h3_gpu_tensor *a;
    h3_gpu_tensor *b;
    uint32_t rank;
    uint32_t input_dim;
    uint32_t output_dim;
} h3_lora_factor;

h3_lora *h3_lora_open(const char *path, float scale,
                      char *error, size_t error_size);
void h3_lora_free(h3_lora *lora);
float h3_lora_scale(const h3_lora *lora);
int h3_lora_load_factor(const h3_lora *lora, h3_gpu *gpu,
                        const char *module, uint32_t input_dim,
                        uint32_t output_dim, h3_lora_factor *factor,
                        char *error, size_t error_size);
void h3_lora_factor_free(h3_lora_factor *factor);
int h3_lora_apply(h3_gpu *gpu, const h3_lora *lora,
                  const h3_lora_factor *factor,
                  const h3_gpu_tensor *input, h3_gpu_tensor *output,
                  h3_gpu_tensor *scratch, h3_gpu_tensor *delta,
                  uint32_t rows, char *error, size_t error_size);
#endif

#include "h3_lora.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct h3_lora {
    h3_st_header header;
    float scale;
};

static void fail(char *error, size_t n, const char *fmt, ...) {
    if (!error || !n) return;
    va_list ap; va_start(ap, fmt); vsnprintf(error, n, fmt, ap); va_end(ap);
}

h3_lora *h3_lora_open(const char *path, float scale,
                      char *error, size_t error_size) {
    if (!path || !*path || (scale != 0.0f && scale != 1.0f)) {
        fail(error, error_size, "Turbo LoRA path is empty or scale is not 0/1");
        return NULL;
    }
    h3_lora *l = calloc(1, sizeof(*l));
    if (!l) return NULL;
    if (!h3_st_read_header(path, &l->header, error, error_size)) {
        free(l); return NULL;
    }
    l->scale = scale;
    return l;
}

void h3_lora_free(h3_lora *lora) {
    if (!lora) return;
    h3_st_free_header(&lora->header);
    free(lora);
}
float h3_lora_scale(const h3_lora *lora) { return lora ? lora->scale : 0.0f; }

void h3_lora_factor_free(h3_lora_factor *f) {
    if (!f) return;
    h3_gpu_tensor_free(f->a); h3_gpu_tensor_free(f->b);
    memset(f, 0, sizeof(*f));
}

int h3_lora_load_factor(const h3_lora *lora, h3_gpu *gpu,
                        const char *module, uint32_t input_dim,
                        uint32_t output_dim, h3_lora_factor *factor,
                        char *error, size_t error_size) {
    if (!factor) return 0;
    memset(factor, 0, sizeof(*factor));
    if (!lora) return 1;
    char an[192], bn[192];
    snprintf(an, sizeof(an), "%s.lora_A.weight", module);
    snprintf(bn, sizeof(bn), "%s.lora_B.weight", module);
    const h3_st_tensor *a = h3_st_find(&lora->header, an);
    const h3_st_tensor *b = h3_st_find(&lora->header, bn);
    if (!a && !b) return 1;
    if (!a || !b || a->dtype != H3_DTYPE_BF16 || b->dtype != H3_DTYPE_BF16 ||
        a->ndim != 2 || b->ndim != 2 || a->shape[1] != input_dim ||
        b->shape[0] != output_dim || a->shape[0] != b->shape[1] ||
        a->shape[0] == 0 || a->shape[0] > UINT32_MAX) {
        fail(error, error_size, "Turbo LoRA shape mismatch for %s", module);
        return 0;
    }
    factor->rank = (uint32_t)a->shape[0];
    factor->input_dim = input_dim;
    factor->output_dim = output_dim;
    factor->a = h3_gpu_tensor_load_bf16(gpu, lora->header.path,
                                         a->file_offset,
                                         (size_t)a->shape[0] * a->shape[1]);
    factor->b = h3_gpu_tensor_load_bf16(gpu, lora->header.path,
                                         b->file_offset,
                                         (size_t)b->shape[0] * b->shape[1]);
    if (!factor->a || !factor->b) {
        fail(error, error_size, "cannot load Turbo LoRA factor %s: %s",
             module, h3_gpu_error(gpu));
        h3_lora_factor_free(factor); return 0;
    }
    return 1;
}

int h3_lora_apply(h3_gpu *gpu, const h3_lora *lora,
                  const h3_lora_factor *f,
                  const h3_gpu_tensor *input, h3_gpu_tensor *output,
                  h3_gpu_tensor *scratch, h3_gpu_tensor *delta,
                  uint32_t rows, char *error, size_t error_size) {
    if (!lora || !f || !f->rank || h3_lora_scale(lora) == 0.0f) return 1;
    int nax = getenv("H3_TURBO_NAX_LORA") != NULL;
    int a_ok = nax ?
        h3_gpu_linear_bf16_nax_force(gpu, scratch, input, f->a, NULL,
                                     rows, f->input_dim, f->rank) :
        h3_gpu_linear_bf16(gpu, scratch, input, f->a, NULL,
                           rows, f->input_dim, f->rank);
    int fused_accum = nax && getenv("H3_TURBO_NAX_LORA_ACCUM") != NULL &&
                      f->rank == 64 && rows >= 128 &&
                      (f->output_dim % 64) == 0;
    if (a_ok && fused_accum) {
        if (h3_gpu_lora_accum_bf16_nax(gpu, output, scratch, f->b, rows,
                                       f->rank, f->output_dim)) return 1;
        fail(error, error_size, "Turbo LoRA fused accumulate failed: %s",
             h3_gpu_error(gpu));
        return 0;
    }
    int b_ok = a_ok && (nax ?
        h3_gpu_linear_bf16_nax_force(gpu, delta, scratch, f->b, NULL,
                                     rows, f->rank, f->output_dim) :
        h3_gpu_linear_bf16(gpu, delta, scratch, f->b, NULL,
                           rows, f->rank, f->output_dim));
    if (!a_ok || !b_ok ||
        !h3_gpu_add_bf16(gpu, output, output, delta,
                         rows * f->output_dim)) {
        fail(error, error_size, "Turbo LoRA apply failed: %s", h3_gpu_error(gpu));
        return 0;
    }
    return 1;
}

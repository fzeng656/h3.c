#include "h3_text_encoder.h"

#include "h3_weights.h"

#include <math.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum {
    TEXT_LAYERS = 50,
    TEXT_VOCAB = 151936,
    TEXT_HIDDEN = 5120,
    TEXT_INTERMEDIATE = 25600,
    TEXT_QUERY_HEADS = 64,
    TEXT_KV_HEADS = 8,
    TEXT_HEAD_DIM = 128,
    TEXT_QUERY_DIM = TEXT_QUERY_HEADS * TEXT_HEAD_DIM,
    TEXT_KV_DIM = TEXT_KV_HEADS * TEXT_HEAD_DIM,
    TEXT_ROPE_HALF = TEXT_HEAD_DIM / 2,
    TEXT_DEFERRED_WEIGHTS = 1 + TEXT_LAYERS * 11
};

static const float TEXT_RMS_EPSILON = 1e-6f;
static const float TEXT_ROPE_THETA = 5000000.0f;

static float round_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= UINT32_C(0xffff0000);
    memcpy(&value, &bits, sizeof(value));
    return value;
}

typedef struct {
    h3_gpu_tensor *input_norm;
    h3_gpu_tensor *query;
    h3_gpu_tensor *key;
    h3_gpu_tensor *value;
    h3_gpu_tensor *query_norm;
    h3_gpu_tensor *key_norm;
    h3_gpu_tensor *attention_output;
    h3_gpu_tensor *post_norm;
    h3_gpu_tensor *gate;
    h3_gpu_tensor *up;
    h3_gpu_tensor *down;
    h3_gpu_tensor *query_i8, *query_scale;
    h3_gpu_tensor *key_i8, *key_scale;
    h3_gpu_tensor *value_i8, *value_scale;
    h3_gpu_tensor *attention_output_i8, *attention_output_scale;
    h3_gpu_tensor *gate_i8, *gate_scale;
    h3_gpu_tensor *up_i8, *up_scale;
    h3_gpu_tensor *down_i8, *down_scale;
} text_layer_weights;

typedef struct {
    const h3_weight_store *store;
    h3_gpu *gpu;
    h3_gpu_tensor *deferred[TEXT_DEFERRED_WEIGHTS];
    size_t deferred_count;
    char *error;
    size_t error_size;
} load_context;

typedef struct {
    const h3_weight_store *store;
    int layer;
    text_layer_weights *weights;
    int lane;
    int lanes;
    int ok;
    char error[512];
} text_layer_prefetch;

typedef struct {
    int occupied;
    int active;
    int layer;
    int lanes;
    load_context load;
    text_layer_weights weights;
    text_layer_prefetch prefetch[8];
    pthread_t threads[8];
    int started[8];
} text_prefetch_slot;

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static h3_gpu_tensor *defer(load_context *load, h3_gpu_tensor *tensor) {
    if (!tensor) return NULL;
    if (load->deferred_count >= TEXT_DEFERRED_WEIGHTS) {
        fail(load->error, load->error_size, "Qwen deferred weight registry overflow");
        h3_gpu_tensor_free(tensor);
        return NULL;
    }
    load->deferred[load->deferred_count++] = tensor;
    return tensor;
}

static void retire_deferred(load_context *load) {
    for (size_t index = 0; index < load->deferred_count; index++)
        h3_gpu_tensor_free(load->deferred[index]);
    load->deferred_count = 0;
}

static h3_gpu_tensor *load_1d(load_context *load, const char *name,
                              uint64_t width) {
    uint64_t shape[] = {width};
    return defer(load, h3_weight_load_bf16(load->store, load->gpu, name, 1,
                                            shape, load->error,
                                            load->error_size));
}

static h3_gpu_tensor *load_2d(load_context *load, const char *name,
                              uint64_t rows, uint64_t columns) {
    uint64_t shape[] = {rows, columns};
    return defer(load, h3_weight_load_bf16(load->store, load->gpu, name, 2,
                                            shape, load->error,
                                            load->error_size));
}

static int load_sparse_embedding_rows(const h3_weight_store *store,
                                      h3_gpu_tensor *hidden,
                                      const uint32_t *token_ids,
                                      size_t token_count,
                                      char *error, size_t error_size) {
    const char *name = "model.language_model.embed_tokens.weight";
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor || !header || tensor->dtype != H3_DTYPE_BF16 ||
        tensor->ndim != 2 || tensor->shape[0] != TEXT_VOCAB ||
        tensor->shape[1] != TEXT_HIDDEN) {
        fail(error, error_size, "Qwen sparse embedding weight is incompatible");
        return 0;
    }
    int fd = open(header->path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        fail(error, error_size, "cannot open Qwen embedding weight: %s",
             strerror(errno));
        return 0;
    }
    size_t row_bytes = (size_t)TEXT_HIDDEN * sizeof(uint16_t);
    uint16_t *row = malloc(row_bytes);
    if (!row) {
        close(fd);
        fail(error, error_size, "out of memory reading sparse Qwen embedding");
        return 0;
    }
    int ok = 1;
    for (size_t index = 0; index < token_count && ok; index++) {
        uint64_t offset = tensor->file_offset +
            (uint64_t)token_ids[index] * (uint64_t)row_bytes;
        size_t done = 0;
        while (done < row_bytes) {
            ssize_t got = pread(fd, (unsigned char *)row + done,
                                row_bytes - done, (off_t)(offset + done));
            if (got < 0 && errno == EINTR) continue;
            if (got <= 0) {
                fail(error, error_size, "cannot read Qwen embedding row %u: %s",
                     token_ids[index], got < 0 ? strerror(errno) :
                                               "unexpected end of file");
                ok = 0;
                break;
            }
            done += (size_t)got;
        }
        if (ok && !h3_gpu_tensor_write_bf16_range(
                hidden, index * TEXT_HIDDEN, row, TEXT_HIDDEN)) {
            fail(error, error_size, "cannot write sparse Qwen embedding row");
            ok = 0;
        }
    }
    free(row);
    close(fd);
    return ok;
}

static int text_prefetch_threads(void) {
    const char *value = getenv("H3_QWEN_PREFETCH");
    if (!value || !*value) return 8;
    if (!strcmp(value, "0")) return 0;
    char *tail = NULL;
    long threads = strtol(value, &tail, 10);
    if (!tail || *tail || threads < 1) threads = 1;
    if (threads > 8) threads = 8;
    return (int)threads;
}

static int text_prefetch_depth(const h3_gpu *gpu) {
    const char *value = getenv("H3_QWEN_PREFETCH_DEPTH");
    if (!value || !*value) return h3_gpu_is_m5(gpu) ? 3 : 2;
    char *tail = NULL;
    long depth = strtol(value, &tail, 10);
    if (!tail || *tail || depth < 1) depth = 1;
    if (depth > 6) depth = 6;
    return (int)depth;
}

static int layer_weights_load(load_context *load, int layer,
                              text_layer_weights *weights) {
    char prefix[96];
    int length = snprintf(prefix, sizeof(prefix),
                          "model.language_model.layers.%d.", layer);
    if (length < 0 || (size_t)length >= sizeof(prefix)) {
        fail(load->error, load->error_size, "cannot format Qwen layer name");
        return 0;
    }
#define LOAD_1D(field, suffix, width) do {                                      \
    char name[192];                                                             \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                     \
    weights->field = load_1d(load, name, width);                                \
    if (!weights->field) return 0;                                              \
} while (0)
#define LOAD_2D(field, suffix, rows, columns) do {                              \
    char name[192];                                                             \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                     \
    weights->field = load_2d(load, name, rows, columns);                        \
    if (!weights->field) return 0;                                              \
} while (0)
    LOAD_1D(input_norm, "input_layernorm.weight", TEXT_HIDDEN);
    LOAD_2D(query, "self_attn.q_proj.weight", TEXT_QUERY_DIM, TEXT_HIDDEN);
    LOAD_2D(key, "self_attn.k_proj.weight", TEXT_KV_DIM, TEXT_HIDDEN);
    LOAD_2D(value, "self_attn.v_proj.weight", TEXT_KV_DIM, TEXT_HIDDEN);
    LOAD_1D(query_norm, "self_attn.q_norm.weight", TEXT_HEAD_DIM);
    LOAD_1D(key_norm, "self_attn.k_norm.weight", TEXT_HEAD_DIM);
    LOAD_2D(attention_output, "self_attn.o_proj.weight", TEXT_HIDDEN,
            TEXT_QUERY_DIM);
    LOAD_1D(post_norm, "post_attention_layernorm.weight", TEXT_HIDDEN);
    LOAD_2D(gate, "mlp.gate_proj.weight", TEXT_INTERMEDIATE, TEXT_HIDDEN);
    LOAD_2D(up, "mlp.up_proj.weight", TEXT_INTERMEDIATE, TEXT_HIDDEN);
    LOAD_2D(down, "mlp.down_proj.weight", TEXT_HIDDEN, TEXT_INTERMEDIATE);
#undef LOAD_1D
#undef LOAD_2D
    return 1;
}

static h3_gpu_tensor *allocate_bf16(load_context *load, size_t elements) {
    h3_gpu_tensor *tensor = defer(
        load, h3_gpu_tensor_new_bf16(load->gpu, elements));
    if (!tensor) {
        fail(load->error, load->error_size,
             "cannot allocate prefetched Qwen weight: %s",
             h3_gpu_error(load->gpu));
    }
    return tensor;
}

static int layer_weights_allocate(load_context *load,
                                  text_layer_weights *weights) {
#define ALLOCATE(field, elements) do {                                         \
    weights->field = allocate_bf16(load, (elements));                          \
    if (!weights->field) return 0;                                              \
} while (0)
    ALLOCATE(input_norm, TEXT_HIDDEN);
    ALLOCATE(query, (size_t)TEXT_QUERY_DIM * TEXT_HIDDEN);
    ALLOCATE(key, (size_t)TEXT_KV_DIM * TEXT_HIDDEN);
    ALLOCATE(value, (size_t)TEXT_KV_DIM * TEXT_HIDDEN);
    ALLOCATE(query_norm, TEXT_HEAD_DIM);
    ALLOCATE(key_norm, TEXT_HEAD_DIM);
    ALLOCATE(attention_output, (size_t)TEXT_HIDDEN * TEXT_QUERY_DIM);
    ALLOCATE(post_norm, TEXT_HIDDEN);
    ALLOCATE(gate, (size_t)TEXT_INTERMEDIATE * TEXT_HIDDEN);
    ALLOCATE(up, (size_t)TEXT_INTERMEDIATE * TEXT_HIDDEN);
    ALLOCATE(down, (size_t)TEXT_HIDDEN * TEXT_INTERMEDIATE);
#undef ALLOCATE
    return 1;
}

static int read_weight_bf16(const h3_weight_store *store, const char *name,
                            int ndim, const uint64_t *shape,
                            h3_gpu_tensor *destination,
                            char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor) {
        fail(error, error_size, "required weight is absent: %s", name);
        return 0;
    }
    if (tensor->dtype != H3_DTYPE_BF16 || tensor->ndim != ndim) {
        fail(error, error_size,
             "weight %s has dtype/rank %s/%d, expected BF16/%d", name,
             h3_dtype_name(tensor->dtype), tensor->ndim, ndim);
        return 0;
    }
    uint64_t elements = 1;
    for (int dimension = 0; dimension < ndim; dimension++) {
        if (tensor->shape[dimension] != shape[dimension]) {
            fail(error, error_size,
                 "weight %s shape mismatch at dimension %d", name,
                 dimension);
            return 0;
        }
        if (shape[dimension] && elements > UINT64_MAX / shape[dimension]) {
            fail(error, error_size, "weight %s shape overflows", name);
            return 0;
        }
        elements *= shape[dimension];
    }
    if (elements > SIZE_MAX ||
        !h3_gpu_tensor_read_file_bf16(destination, header->path,
                                      tensor->file_offset, (size_t)elements,
                                      error, error_size)) {
        if (error && error_size && !error[0])
            fail(error, error_size, "cannot prefetch %s", name);
        return 0;
    }
    return 1;
}

static int layer_weights_read_lane(const h3_weight_store *store, int layer,
                                   text_layer_weights *weights,
                                   int lane, int lanes,
                                   char *error, size_t error_size) {
    char prefix[96];
    int length = snprintf(prefix, sizeof(prefix),
                          "model.language_model.layers.%d.", layer);
    if (length < 0 || (size_t)length >= sizeof(prefix)) {
        fail(error, error_size, "cannot format Qwen layer name");
        return 0;
    }
#define READ_1D(field, suffix, width) do {                                     \
    int selected = item++ % lanes == lane;                                     \
    char name[192];                                                             \
    uint64_t shape[] = {width};                                                 \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    if (selected && !read_weight_bf16(                                         \
                          store, name, 1, shape, weights->field,               \
                          error, error_size)) return 0;                         \
} while (0)
#define READ_2D(field, suffix, rows, columns) do {                             \
    int selected = item++ % lanes == lane;                                     \
    char name[192];                                                             \
    uint64_t shape[] = {rows, columns};                                         \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    if (selected && !read_weight_bf16(                                         \
                          store, name, 2, shape, weights->field,               \
                          error, error_size)) return 0;                         \
} while (0)
    int item = 0;
    READ_1D(input_norm, "input_layernorm.weight", TEXT_HIDDEN);
    READ_2D(query, "self_attn.q_proj.weight", TEXT_QUERY_DIM, TEXT_HIDDEN);
    READ_2D(key, "self_attn.k_proj.weight", TEXT_KV_DIM, TEXT_HIDDEN);
    READ_2D(value, "self_attn.v_proj.weight", TEXT_KV_DIM, TEXT_HIDDEN);
    READ_1D(query_norm, "self_attn.q_norm.weight", TEXT_HEAD_DIM);
    READ_1D(key_norm, "self_attn.k_norm.weight", TEXT_HEAD_DIM);
    READ_2D(attention_output, "self_attn.o_proj.weight", TEXT_HIDDEN,
            TEXT_QUERY_DIM);
    READ_1D(post_norm, "post_attention_layernorm.weight", TEXT_HIDDEN);
    READ_2D(gate, "mlp.gate_proj.weight", TEXT_INTERMEDIATE, TEXT_HIDDEN);
    READ_2D(up, "mlp.up_proj.weight", TEXT_INTERMEDIATE, TEXT_HIDDEN);
    READ_2D(down, "mlp.down_proj.weight", TEXT_HIDDEN, TEXT_INTERMEDIATE);
#undef READ_1D
#undef READ_2D
    return 1;
}

static void *layer_prefetch_main(void *opaque) {
    text_layer_prefetch *prefetch = opaque;
    prefetch->ok = layer_weights_read_lane(
        prefetch->store, prefetch->layer, prefetch->weights,
        prefetch->lane, prefetch->lanes,
        prefetch->error, sizeof(prefetch->error));
    return NULL;
}

static void prefetch_slot_retire(text_prefetch_slot *slot) {
    if (!slot || !slot->occupied) return;
    if (slot->active) {
        for (int lane = 0; lane < slot->lanes; lane++) {
            if (slot->started[lane])
                pthread_join(slot->threads[lane], NULL);
        }
    }
    retire_deferred(&slot->load);
    memset(slot, 0, sizeof(*slot));
}

static int prefetch_slot_start(text_prefetch_slot *slot,
                               const h3_weight_store *store, h3_gpu *gpu,
                               int layer, int lanes,
                               char *error, size_t error_size) {
    if (!slot || slot->occupied || lanes < 1 || lanes > 8) return 0;
    memset(slot, 0, sizeof(*slot));
    slot->occupied = 1;
    slot->active = 1;
    slot->layer = layer;
    slot->lanes = lanes;
    slot->load.store = store;
    slot->load.gpu = gpu;
    slot->load.error = error;
    slot->load.error_size = error_size;
    if (!layer_weights_allocate(&slot->load, &slot->weights)) {
        prefetch_slot_retire(slot);
        return 0;
    }
    for (int lane = 0; lane < lanes; lane++) {
        text_layer_prefetch *prefetch = &slot->prefetch[lane];
        prefetch->store = store;
        prefetch->layer = layer;
        prefetch->weights = &slot->weights;
        prefetch->lane = lane;
        prefetch->lanes = lanes;
        if (pthread_create(&slot->threads[lane], NULL,
                           layer_prefetch_main, prefetch) == 0) {
            slot->started[lane] = 1;
        } else {
            prefetch->ok = layer_weights_read_lane(
                store, layer, &slot->weights, lane, lanes,
                prefetch->error, sizeof(prefetch->error));
        }
    }
    return 1;
}

static int prefetch_slot_take(text_prefetch_slot *slot,
                              load_context *load,
                              text_layer_weights *weights,
                              char *error, size_t error_size) {
    if (!slot || !slot->occupied || !load || !weights) return 0;
    if (slot->active) {
        for (int lane = 0; lane < slot->lanes; lane++) {
            if (slot->started[lane])
                pthread_join(slot->threads[lane], NULL);
        }
        slot->active = 0;
    }
    for (int lane = 0; lane < slot->lanes; lane++) {
        if (!slot->prefetch[lane].ok) {
            fail(error, error_size,
                 "Qwen layer %d prefetch lane %d failed: %s",
                 slot->layer, lane,
                 slot->prefetch[lane].error[0] ?
                     slot->prefetch[lane].error : "unknown error");
            prefetch_slot_retire(slot);
            return 0;
        }
    }
    *load = slot->load;
    *weights = slot->weights;
    slot->load.deferred_count = 0;
    memset(slot, 0, sizeof(*slot));
    return 1;
}

static void prefetch_slots_retire(text_prefetch_slot *slots, int count) {
    for (int index = 0; index < count; index++)
        prefetch_slot_retire(&slots[index]);
}

static int gpu_operation(h3_gpu *gpu, int ok, char *error, size_t error_size,
                         const char *operation, int layer) {
    if (ok) return 1;
    if (layer >= 0) {
        fail(error, error_size, "Qwen layer %d %s failed: %s", layer,
             operation, h3_gpu_error(gpu));
    } else {
        fail(error, error_size, "Qwen %s failed: %s", operation,
             h3_gpu_error(gpu));
    }
    return 0;
}

static h3_gpu_tensor *defer_new_i8(load_context *load, size_t elements) {
    return defer(load, h3_gpu_tensor_new_i8(load->gpu, elements));
}

static h3_gpu_tensor *defer_new_f32(load_context *load, size_t elements) {
    return defer(load, h3_gpu_tensor_new_f32(load->gpu, elements));
}

static int prepare_int8_weight(load_context *load,
                               const h3_gpu_tensor *source,
                               uint32_t rows, uint32_t columns,
                               uint32_t group_size,
                               h3_gpu_tensor **quantized,
                               h3_gpu_tensor **scales) {
    uint32_t groups = group_size ? columns / group_size : 1u;
    *quantized = defer_new_i8(load, (size_t)rows * columns);
    *scales = defer_new_f32(load, (size_t)rows * groups);
    if (!*quantized || !*scales) return 0;
    int ok = group_size ?
        h3_gpu_quantize_weight_int8_grouped(
            load->gpu, *quantized, *scales, source, rows, columns, group_size) :
        h3_gpu_quantize_weight_int8(load->gpu, *quantized, *scales,
                                    source, rows, columns);
    const char *weight_only = getenv("H3_QWEN_INT8_WEIGHT_ONLY");
    if (ok && group_size && weight_only && *weight_only &&
        strcmp(weight_only, "0"))
        ok = h3_gpu_dequantize_int8_grouped_bf16(
            load->gpu, (h3_gpu_tensor *)source, *quantized, *scales,
            rows, columns, group_size);
    return ok;
}

static int layer_weights_quantize_int8(load_context *load,
                                       text_layer_weights *weight,
                                       uint32_t group_size) {
    if (!h3_gpu_begin(load->gpu)) return 0;
    int ok = prepare_int8_weight(load, weight->query,
                                 TEXT_QUERY_DIM, TEXT_HIDDEN,
                                 group_size, &weight->query_i8, &weight->query_scale) &&
        prepare_int8_weight(load, weight->key,
                            TEXT_KV_DIM, TEXT_HIDDEN,
                            group_size, &weight->key_i8, &weight->key_scale) &&
        prepare_int8_weight(load, weight->value,
                            TEXT_KV_DIM, TEXT_HIDDEN,
                            group_size, &weight->value_i8, &weight->value_scale) &&
        prepare_int8_weight(load, weight->attention_output,
                            TEXT_HIDDEN, TEXT_QUERY_DIM,
                            group_size, &weight->attention_output_i8,
                            &weight->attention_output_scale) &&
        prepare_int8_weight(load, weight->gate,
                            TEXT_INTERMEDIATE, TEXT_HIDDEN,
                            group_size, &weight->gate_i8, &weight->gate_scale) &&
        prepare_int8_weight(load, weight->up,
                            TEXT_INTERMEDIATE, TEXT_HIDDEN,
                            group_size, &weight->up_i8, &weight->up_scale) &&
        prepare_int8_weight(load, weight->down,
                            TEXT_HIDDEN, TEXT_INTERMEDIATE,
                            group_size, &weight->down_i8, &weight->down_scale) &&
        h3_gpu_submit(load->gpu);
    return ok;
}

enum {
    QWEN_I8_QKV = 1u << 0,
    QWEN_I8_ATTN_OUT = 1u << 1,
    QWEN_I8_GATE_UP = 1u << 2,
    QWEN_I8_DOWN = 1u << 3,
    QWEN_I8_ALL = QWEN_I8_QKV | QWEN_I8_ATTN_OUT |
                  QWEN_I8_GATE_UP | QWEN_I8_DOWN
};

static unsigned qwen_int8_mask(void) {
    const char *enabled = getenv("H3_QWEN_INT8_RUNTIME");
    if (!enabled || !*enabled || !strcmp(enabled, "0")) return 0;
    const char *stage = getenv("H3_QWEN_INT8_STAGE");
    if (!stage || !*stage || !strcmp(stage, "all")) return QWEN_I8_ALL;
    if (!strcmp(stage, "qkv")) return QWEN_I8_QKV;
    if (!strcmp(stage, "out")) return QWEN_I8_ATTN_OUT;
    if (!strcmp(stage, "gateup")) return QWEN_I8_GATE_UP;
    if (!strcmp(stage, "down")) return QWEN_I8_DOWN;
    if (!strcmp(stage, "attention")) return QWEN_I8_QKV | QWEN_I8_ATTN_OUT;
    if (!strcmp(stage, "mlp")) return QWEN_I8_GATE_UP | QWEN_I8_DOWN;
    return QWEN_I8_ALL;
}

static uint32_t qwen_int8_group_size(void) {
    const char *value = getenv("H3_QWEN_INT8_GROUP");
    if (!value || !*value || !strcmp(value, "0")) return 0;
    char *end = NULL;
    unsigned long parsed = strtoul(value, &end, 10);
    if (!end || *end || (parsed != 32 && parsed != 64 &&
                         parsed != 128 && parsed != 256 &&
                         parsed != 512 && parsed != 1024)) return 0;
    return (uint32_t)parsed;
}

static int encode_layer(h3_gpu *gpu, const text_layer_weights *weight,
                        uint32_t tokens, unsigned int8_mask,
                        uint32_t int8_group_size, int activation_only,
                        h3_gpu_tensor *quantized_input,
                        h3_gpu_tensor *input_scales,
                        h3_gpu_tensor *hidden,
                        h3_gpu_tensor *norm, h3_gpu_tensor *query,
                        h3_gpu_tensor *key, h3_gpu_tensor *value,
                        h3_gpu_tensor *attention_heads,
                        h3_gpu_tensor *attention_output,
                        h3_gpu_tensor *gate, h3_gpu_tensor *up,
                        h3_gpu_tensor *mlp_output,
                        h3_gpu_tensor *rope_cos, h3_gpu_tensor *rope_sin,
                        int layer, char *error, size_t error_size) {
#define OP(call, label) do {                                                    \
    if (!gpu_operation(gpu, (call), error, error_size, label, layer)) return 0; \
} while (0)
#define I8_LINEAR(out, input, wi8, wscale, in_dim, out_dim)                    \
    (int8_group_size ? h3_gpu_linear_int8_grouped_bf16(                        \
        gpu, (out), quantized_input, input_scales, (input), (wi8), (wscale),    \
        tokens, (in_dim), (out_dim), int8_group_size) :                        \
     h3_gpu_linear_int8_bf16(                                                   \
        gpu, (out), quantized_input, input_scales, (input), (wi8), (wscale),    \
        tokens, (in_dim), (out_dim), 0))
    OP(h3_gpu_rms_norm_bf16(gpu, norm, hidden, weight->input_norm,
                             tokens, TEXT_HIDDEN, TEXT_RMS_EPSILON),
       "input RMSNorm");
    if (activation_only)
        OP(h3_gpu_roundtrip_int8_grouped_bf16(
               gpu, norm, quantized_input, input_scales, tokens,
               TEXT_HIDDEN, int8_group_size),
           "input activation int8 roundtrip");
    OP((int8_mask & QWEN_I8_QKV) ?
        I8_LINEAR(query, norm, weight->query_i8, weight->query_scale,
                   TEXT_HIDDEN, TEXT_QUERY_DIM) :
        h3_gpu_linear_bf16(gpu, query, norm, weight->query, NULL, tokens,
                           TEXT_HIDDEN, TEXT_QUERY_DIM), "query projection");
    OP((int8_mask & QWEN_I8_QKV) ?
        I8_LINEAR(key, norm, weight->key_i8, weight->key_scale,
                   TEXT_HIDDEN, TEXT_KV_DIM) :
        h3_gpu_linear_bf16(gpu, key, norm, weight->key, NULL, tokens,
                           TEXT_HIDDEN, TEXT_KV_DIM), "key projection");
    OP((int8_mask & QWEN_I8_QKV) ?
        I8_LINEAR(value, norm, weight->value_i8, weight->value_scale,
                   TEXT_HIDDEN, TEXT_KV_DIM) :
        h3_gpu_linear_bf16(gpu, value, norm, weight->value, NULL, tokens,
                           TEXT_HIDDEN, TEXT_KV_DIM), "value projection");
    OP(h3_gpu_head_rms_norm_bf16(gpu, query, weight->query_norm, tokens,
                                  TEXT_QUERY_HEADS, TEXT_HEAD_DIM,
                                  TEXT_RMS_EPSILON), "query RMSNorm");
    OP(h3_gpu_head_rms_norm_bf16(gpu, key, weight->key_norm, tokens,
                                  TEXT_KV_HEADS, TEXT_HEAD_DIM,
                                  TEXT_RMS_EPSILON), "key RMSNorm");
    OP(h3_gpu_rope_text_bf16(gpu, query, key, rope_cos, rope_sin, tokens,
                              TEXT_QUERY_HEADS, TEXT_KV_HEADS, TEXT_HEAD_DIM),
       "RoPE");
    OP(h3_gpu_gqa_causal_bf16(gpu, attention_heads, query, key, value, tokens,
                               TEXT_QUERY_HEADS, TEXT_KV_HEADS, TEXT_HEAD_DIM,
                               1.0f / sqrtf((float)TEXT_HEAD_DIM)),
       "causal GQA");
    if (activation_only)
        OP(h3_gpu_roundtrip_int8_grouped_bf16(
               gpu, attention_heads, quantized_input, input_scales, tokens,
               TEXT_QUERY_DIM, int8_group_size),
           "attention activation int8 roundtrip");
    OP((int8_mask & QWEN_I8_ATTN_OUT) ?
        I8_LINEAR(attention_output, attention_heads,
                   weight->attention_output_i8,
                   weight->attention_output_scale, TEXT_QUERY_DIM, TEXT_HIDDEN) :
        h3_gpu_linear_bf16(gpu, attention_output, attention_heads,
                           weight->attention_output, NULL, tokens,
                           TEXT_QUERY_DIM, TEXT_HIDDEN),
       "attention output projection");
    OP(h3_gpu_add_bf16(gpu, hidden, hidden, attention_output,
                        tokens * TEXT_HIDDEN), "attention residual");
    OP(h3_gpu_rms_norm_bf16(gpu, norm, hidden, weight->post_norm, tokens,
                             TEXT_HIDDEN, TEXT_RMS_EPSILON),
       "post-attention RMSNorm");
    if (activation_only)
        OP(h3_gpu_roundtrip_int8_grouped_bf16(
               gpu, norm, quantized_input, input_scales, tokens,
               TEXT_HIDDEN, int8_group_size),
           "MLP input activation int8 roundtrip");
    OP((int8_mask & QWEN_I8_GATE_UP) ?
        I8_LINEAR(gate, norm, weight->gate_i8, weight->gate_scale,
                   TEXT_HIDDEN, TEXT_INTERMEDIATE) :
        h3_gpu_linear_bf16(gpu, gate, norm, weight->gate, NULL, tokens,
                           TEXT_HIDDEN, TEXT_INTERMEDIATE), "MLP gate");
    OP((int8_mask & QWEN_I8_GATE_UP) ?
        I8_LINEAR(up, norm, weight->up_i8, weight->up_scale,
                   TEXT_HIDDEN, TEXT_INTERMEDIATE) :
        h3_gpu_linear_bf16(gpu, up, norm, weight->up, NULL, tokens,
                           TEXT_HIDDEN, TEXT_INTERMEDIATE), "MLP up");
    OP(h3_gpu_silu_mul_bf16(gpu, gate, gate, up,
                             tokens * TEXT_INTERMEDIATE), "fused SwiGLU");
    if (activation_only)
        OP(h3_gpu_roundtrip_int8_grouped_bf16(
               gpu, gate, quantized_input, input_scales, tokens,
               TEXT_INTERMEDIATE, int8_group_size),
           "MLP output activation int8 roundtrip");
    OP((int8_mask & QWEN_I8_DOWN) ?
        I8_LINEAR(mlp_output, gate, weight->down_i8, weight->down_scale,
                   TEXT_INTERMEDIATE, TEXT_HIDDEN) :
        h3_gpu_linear_bf16(gpu, mlp_output, gate, weight->down, NULL, tokens,
                           TEXT_INTERMEDIATE, TEXT_HIDDEN), "MLP down");
    OP(h3_gpu_add_bf16(gpu, hidden, hidden, mlp_output,
                        tokens * TEXT_HIDDEN), "MLP residual");
#undef I8_LINEAR
#undef OP
    return 1;
}

void h3_text_embedding_free(h3_text_embedding *embedding) {
    if (!embedding) return;
    free(embedding->values);
    free(embedding->tags);
    memset(embedding, 0, sizeof(*embedding));
}

static int text_encode_bf16_impl(
                        const char *weight_directory,
                        const char *shader_source_path,
                        const uint32_t *token_ids, size_t token_count,
                        const h3_text_vision_span *spans, size_t span_count,
                        const uint32_t *position_ids, const uint8_t *tags,
                        int layer_count,
                        h3_text_progress progress, void *progress_opaque,
                        h3_text_embedding *output,
                        char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (!weight_directory || !shader_source_path || !token_ids || !token_count ||
        !output || token_count > UINT32_MAX ||
        layer_count < 1 || layer_count > TEXT_LAYERS ||
        token_count > UINT32_MAX / TEXT_HIDDEN ||
        token_count > UINT32_MAX / TEXT_INTERMEDIATE) {
        fail(error, error_size, "invalid Qwen text encoder arguments");
        return 0;
    }
    for (size_t index = 0; index < token_count; index++) {
        if (token_ids[index] >= TEXT_VOCAB) {
            fail(error, error_size, "Qwen token ID %u is outside the vocabulary",
                 token_ids[index]);
            return 0;
        }
        if (tags && tags[index] > 2) {
            fail(error, error_size, "Qwen presentation tag is invalid");
            return 0;
        }
    }
    size_t span_cursor = 0;
    for (size_t index = 0; index < span_count; index++) {
        const h3_text_vision_span *span = &spans[index];
        if (!span->tokens || span->start < span_cursor ||
            span->start > token_count || span->tokens > token_count - span->start ||
            !span->embeddings || !span->deepstack[0] || !span->deepstack[1] ||
            !span->deepstack[2]) {
            fail(error, error_size, "invalid Qwen vision presentation span");
            return 0;
        }
        span_cursor = span->start + span->tokens;
    }

    h3_weight_store *store = h3_weight_store_open(weight_directory, error,
                                                   error_size);
    if (!store) return 0;
    h3_gpu *gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (!gpu) {
        h3_weight_store_free(store);
        return 0;
    }
    h3_gpu_profile_set_label(gpu, "Qwen text encoder");
    load_context load = {store, gpu, {NULL}, 0, error, error_size};
    uint32_t tokens = (uint32_t)token_count;
    size_t hidden_count = token_count * TEXT_HIDDEN;
    size_t query_count = token_count * TEXT_QUERY_DIM;
    size_t kv_count = token_count * TEXT_KV_DIM;
    size_t intermediate_count = token_count * TEXT_INTERMEDIATE;
    unsigned requested_int8_mask = qwen_int8_mask();
    const char *weight_only_value = getenv("H3_QWEN_INT8_WEIGHT_ONLY");
    const char *activation_only_value = getenv("H3_QWEN_INT8_ACT_ONLY");
    int weight_only = weight_only_value && *weight_only_value &&
                      strcmp(weight_only_value, "0");
    int activation_only = activation_only_value && *activation_only_value &&
                          strcmp(activation_only_value, "0");
    unsigned int8_mask = (weight_only || activation_only) ? 0u :
                         requested_int8_mask;
    int quantize_weights = weight_only || (!activation_only && int8_mask != 0);
    int use_int8_runtime = quantize_weights || activation_only;
    uint32_t int8_group_size = use_int8_runtime ? qwen_int8_group_size() : 0;
    uint32_t padded_tokens = (tokens + 127u) & ~127u;
    uint32_t int8_scale_groups = int8_group_size ?
        TEXT_INTERMEDIATE / int8_group_size : 1u;
    if ((weight_only || activation_only) && !int8_group_size) {
        fail(error, error_size,
             "Qwen grouped int8 diagnostic requires H3_QWEN_INT8_GROUP");
        h3_gpu_free(gpu);
        h3_weight_store_free(store);
        return 0;
    }

    float *cosines = malloc(token_count * TEXT_ROPE_HALF * sizeof(*cosines));
    float *sines = malloc(token_count * TEXT_ROPE_HALF * sizeof(*sines));
    if (!cosines || !sines) {
        fail(error, error_size, "out of memory allocating Qwen RoPE tables");
        free(cosines);
        free(sines);
        h3_gpu_free(gpu);
        h3_weight_store_free(store);
        return 0;
    }
    float inverse_frequency[TEXT_ROPE_HALF];
    for (size_t index = 0; index < TEXT_ROPE_HALF; index++) {
        inverse_frequency[index] = 1.0f /
            powf(TEXT_ROPE_THETA,
                 (float)(index * 2) / (float)TEXT_HEAD_DIM);
    }
    for (size_t position = 0; position < token_count; position++) {
        for (size_t index = 0; index < TEXT_ROPE_HALF; index++) {
            size_t axis = 0;
            if (position_ids && index < 60 && index % 3 == 1) axis = 1;
            else if (position_ids && index < 60 && index % 3 == 2) axis = 2;
            float coordinate = position_ids ?
                (float)position_ids[axis * token_count + position] :
                (float)position;
            float angle = coordinate * inverse_frequency[index];
            float cosine = cosf(angle);
            float sine = sinf(angle);
            /* The explicit Qwen mRoPE path casts its tables to the BF16
             * embedding dtype. Keep F32 storage for the fused Metal kernel,
             * but pin the values to that same operation boundary. */
            cosines[position * TEXT_ROPE_HALF + index] =
                position_ids ? round_bf16(cosine) : cosine;
            sines[position * TEXT_ROPE_HALF + index] =
                position_ids ? round_bf16(sine) : sine;
        }
    }

    h3_gpu_tensor *ids = h3_gpu_tensor_from_u32(gpu, token_ids, token_count);
    h3_gpu_tensor *rope_cos = h3_gpu_tensor_from_f32(
        gpu, cosines, token_count * TEXT_ROPE_HALF);
    h3_gpu_tensor *rope_sin = h3_gpu_tensor_from_f32(
        gpu, sines, token_count * TEXT_ROPE_HALF);
    free(cosines);
    free(sines);
    h3_gpu_tensor *hidden = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *norm = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *query = h3_gpu_tensor_new_bf16(gpu, query_count);
    h3_gpu_tensor *key = h3_gpu_tensor_new_bf16(gpu, kv_count);
    h3_gpu_tensor *value = h3_gpu_tensor_new_bf16(gpu, kv_count);
    h3_gpu_tensor *attention_heads = h3_gpu_tensor_new_bf16(gpu, query_count);
    h3_gpu_tensor *attention_output = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *gate = h3_gpu_tensor_new_bf16(gpu, intermediate_count);
    h3_gpu_tensor *up = h3_gpu_tensor_new_bf16(gpu, intermediate_count);
    h3_gpu_tensor *mlp_output = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *quantized_input = use_int8_runtime ?
        h3_gpu_tensor_new_i8(gpu, (size_t)padded_tokens * TEXT_INTERMEDIATE) : NULL;
    h3_gpu_tensor *input_scales = use_int8_runtime ?
        h3_gpu_tensor_new_f32(gpu,
            (size_t)padded_tokens * int8_scale_groups) : NULL;
    h3_gpu_tensor *deepstack[3] = {NULL, NULL, NULL};
    if (span_count) {
        uint16_t *values = calloc(hidden_count, sizeof(*values));
        if (!values) {
            fail(error, error_size, "out of memory packing Qwen deepstack rows");
            goto early_cleanup;
        }
        for (size_t layer = 0; layer < 3; layer++) {
            memset(values, 0, hidden_count * sizeof(*values));
            for (size_t index = 0; index < span_count; index++) {
                const h3_text_vision_span *span = &spans[index];
                memcpy(values + span->start * TEXT_HIDDEN,
                       span->deepstack[layer],
                       span->tokens * TEXT_HIDDEN * sizeof(*values));
            }
            deepstack[layer] = h3_gpu_tensor_from_bf16(gpu, values,
                                                        hidden_count);
        }
        free(values);
    }
    h3_gpu_tensor *activations[] = {
        ids, rope_cos, rope_sin, hidden, norm, query, key, value,
        attention_heads, attention_output, gate, up, mlp_output,
        deepstack[0], deepstack[1], deepstack[2],
        quantized_input, input_scales
    };
    int ok = 1;
    for (size_t index = 0; index < sizeof(activations) / sizeof(*activations);
         index++) {
        if (!activations[index] && (index < 13 || span_count)) ok = 0;
    }
    if (use_int8_runtime && (!quantized_input || !input_scales)) ok = 0;
    if (!ok) {
        fail(error, error_size, "cannot allocate Qwen activations: %s",
             h3_gpu_error(gpu));
        goto cleanup;
    }

    const char *sparse_embedding = getenv("H3_QWEN_SPARSE_EMBED");
    if (sparse_embedding && *sparse_embedding && strcmp(sparse_embedding, "0")) {
        double embedding_started = 0.0;
        if (getenv("H3_PROFILE")) {
            struct timespec now;
            if (clock_gettime(CLOCK_MONOTONIC, &now) == 0)
                embedding_started = (double)now.tv_sec +
                    (double)now.tv_nsec * 1e-9;
        }
        if (!load_sparse_embedding_rows(store, hidden, token_ids, token_count,
                                        error, error_size)) goto cleanup;
        if (getenv("H3_PROFILE") && embedding_started > 0.0) {
            struct timespec now;
            if (clock_gettime(CLOCK_MONOTONIC, &now) == 0) {
                double ended = (double)now.tv_sec +
                    (double)now.tv_nsec * 1e-9;
                fprintf(stderr, "h3: Qwen sparse embedding %zu tokens %.3fs (%.2f MiB)\n",
                        token_count, ended - embedding_started,
                        (double)token_count * (double)TEXT_HIDDEN * sizeof(uint16_t) /
                            (1024.0 * 1024.0));
            }
        }
    } else {
        h3_gpu_tensor *embedding_weight = load_2d(
            &load, "model.language_model.embed_tokens.weight", TEXT_VOCAB,
            TEXT_HIDDEN);
        if (!embedding_weight) goto cleanup;
        if (!gpu_operation(gpu, h3_gpu_begin(gpu), error, error_size,
                           "command stream begin", -1) ||
            !gpu_operation(gpu, h3_gpu_embedding_bf16(
                                     gpu, hidden, embedding_weight, ids, tokens,
                                     TEXT_VOCAB, TEXT_HIDDEN),
                           error, error_size, "embedding lookup", -1) ||
            !gpu_operation(gpu, h3_gpu_submit(gpu), error, error_size,
                           "embedding stream submit", -1)) {
            goto cleanup;
        }
        retire_deferred(&load);
    }
    for (size_t index = 0; index < span_count; index++) {
        const h3_text_vision_span *span = &spans[index];
        if (!h3_gpu_tensor_write_bf16_range(
                hidden, span->start * TEXT_HIDDEN, span->embeddings,
                span->tokens * TEXT_HIDDEN)) {
            fail(error, error_size, "cannot splice Qwen vision embeddings");
            goto cleanup;
        }
    }

    int prefetch_threads = text_prefetch_threads();
    int prefetch_layers = prefetch_threads > 0 && layer_count > 1;
    int prefetch_depth = prefetch_layers ? text_prefetch_depth(gpu) : 0;
    text_prefetch_slot slots[6];
    memset(slots, 0, sizeof(slots));
    text_layer_weights weights;
    memset(&weights, 0, sizeof(weights));
    if (!layer_weights_load(&load, 0, &weights)) goto cleanup;
    int next_prefetch_layer = 1;
    for (int index = 0;
         index < prefetch_depth && next_prefetch_layer < layer_count;
         index++, next_prefetch_layer++) {
        if (!prefetch_slot_start(&slots[index], store, gpu,
                                 next_prefetch_layer, prefetch_threads,
                                 error, error_size)) {
            prefetch_slots_retire(slots, prefetch_depth);
            goto cleanup;
        }
    }
    if (quantize_weights && !layer_weights_quantize_int8(&load, &weights, int8_group_size)) {
        fail(error, error_size, "cannot quantize Qwen layer 0 to int8: %s",
             h3_gpu_error(gpu));
        prefetch_slots_retire(slots, prefetch_depth);
        goto cleanup;
    }
    for (int layer = 0; layer < layer_count; layer++) {
        int layer_ok =
            gpu_operation(gpu, h3_gpu_begin(gpu), error, error_size,
                          "layer stream begin", layer) &&
            encode_layer(gpu, &weights, tokens, int8_mask, int8_group_size,
                         activation_only, quantized_input, input_scales, hidden, norm, query, key,
                         value, attention_heads, attention_output, gate, up,
                         mlp_output, rope_cos, rope_sin, layer,
                         error, error_size) &&
            (!(layer < 3 && span_count) ||
             gpu_operation(gpu, h3_gpu_add_bf16(
                 gpu, hidden, hidden, deepstack[layer],
                 tokens * TEXT_HIDDEN), error, error_size,
                 "deepstack residual", layer)) &&
            gpu_operation(gpu, h3_gpu_submit(gpu), error, error_size,
                          "layer stream submit", layer);

        retire_deferred(&load);
        if (!layer_ok) {
            prefetch_slots_retire(slots, prefetch_depth);
            goto cleanup;
        }
        if (progress) progress(layer + 1, TEXT_LAYERS, progress_opaque);
        if (layer + 1 >= layer_count) continue;

        memset(&weights, 0, sizeof(weights));
        if (prefetch_layers) {
            text_prefetch_slot *next = NULL;
            for (int index = 0; index < prefetch_depth; index++) {
                if (slots[index].occupied &&
                    slots[index].layer == layer + 1) {
                    next = &slots[index];
                    break;
                }
            }
            if (!next || !prefetch_slot_take(next, &load, &weights,
                                              error, error_size)) {
                if (!next)
                    fail(error, error_size,
                         "Qwen layer %d is absent from the prefetch ring",
                         layer + 1);
                prefetch_slots_retire(slots, prefetch_depth);
                goto cleanup;
            }
            if (next_prefetch_layer < layer_count) {
                if (!prefetch_slot_start(
                        next, store, gpu, next_prefetch_layer,
                        prefetch_threads, error, error_size)) {
                    prefetch_slots_retire(slots, prefetch_depth);
                    goto cleanup;
                }
                next_prefetch_layer++;
            }
        } else if (!layer_weights_load(&load, layer + 1, &weights)) {
            goto cleanup;
        }
        if (quantize_weights &&
            !layer_weights_quantize_int8(&load, &weights, int8_group_size)) {
            fail(error, error_size, "cannot quantize Qwen layer %d to int8: %s",
                 layer + 1, h3_gpu_error(gpu));
            prefetch_slots_retire(slots, prefetch_depth);
            goto cleanup;
        }
    }

    output->values = malloc(hidden_count * sizeof(*output->values));
    if (!output->values) {
        fail(error, error_size, "out of memory reading Qwen output");
        goto cleanup;
    }
    if (!h3_gpu_tensor_read_bf16(hidden, output->values, hidden_count) ||
        !h3_gpu_get_stats(gpu, &output->gpu_stats)) {
        h3_text_embedding_free(output);
        fail(error, error_size, "cannot read completed Qwen output");
        goto cleanup;
    }
    const char *qwen_dump = getenv("H3_QWEN_DUMP");
    if (qwen_dump && *qwen_dump) {
        FILE *dump = fopen(qwen_dump, "wb");
        if (dump) {
            (void)fwrite(output->values, sizeof(*output->values),
                         hidden_count, dump);
            fclose(dump);
        }
    }
    output->tokens = token_count;
    output->width = TEXT_HIDDEN;
    if (tags) {
        output->tags = malloc(token_count * sizeof(*output->tags));
        if (!output->tags) {
            h3_text_embedding_free(output);
            fail(error, error_size, "out of memory reading Qwen tags");
            goto cleanup;
        }
        memcpy(output->tags, tags, token_count * sizeof(*output->tags));
    }
    ok = 1;
    goto finished;

cleanup:
    ok = 0;
finished:
    retire_deferred(&load);
    for (size_t index = 0; index < sizeof(activations) / sizeof(*activations);
         index++) {
        h3_gpu_tensor_free(activations[index]);
    }
    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    return ok;

early_cleanup:
    h3_gpu_tensor_free(ids);
    h3_gpu_tensor_free(rope_cos);
    h3_gpu_tensor_free(rope_sin);
    h3_gpu_tensor_free(hidden);
    h3_gpu_tensor_free(norm);
    h3_gpu_tensor_free(query);
    h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(value);
    h3_gpu_tensor_free(attention_heads);
    h3_gpu_tensor_free(attention_output);
    h3_gpu_tensor_free(gate);
    h3_gpu_tensor_free(up);
    h3_gpu_tensor_free(mlp_output);
    for (size_t index = 0; index < 3; index++)
        h3_gpu_tensor_free(deepstack[index]);
    retire_deferred(&load);
    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    return 0;
}

int h3_text_encode_bf16(const char *weight_directory,
                        const char *shader_source_path,
                        const uint32_t *token_ids, size_t token_count,
                        h3_text_progress progress, void *progress_opaque,
                        h3_text_embedding *output,
                        char *error, size_t error_size) {
    return text_encode_bf16_impl(
        weight_directory, shader_source_path, token_ids, token_count,
        NULL, 0, NULL, NULL, TEXT_LAYERS, progress, progress_opaque,
        output, error, error_size);
}

int h3_text_encode_multimodal_bf16(
                        const char *weight_directory,
                        const char *shader_source_path,
                        const uint32_t *token_ids, size_t token_count,
                        const h3_text_vision_span *spans, size_t span_count,
                        const uint32_t *position_ids, const uint8_t *tags,
                        h3_text_progress progress, void *progress_opaque,
                        h3_text_embedding *output,
                        char *error, size_t error_size) {
    if (!spans || !span_count || !position_ids || !tags) {
        if (output) memset(output, 0, sizeof(*output));
        fail(error, error_size, "multimodal Qwen presentation is incomplete");
        return 0;
    }
    return text_encode_bf16_impl(
        weight_directory, shader_source_path, token_ids, token_count,
        spans, span_count, position_ids, tags, TEXT_LAYERS,
        progress, progress_opaque,
        output, error, error_size);
}

int h3_text_encode_multimodal_layers_bf16(
                        const char *weight_directory,
                        const char *shader_source_path,
                        const uint32_t *token_ids, size_t token_count,
                        const h3_text_vision_span *spans, size_t span_count,
                        const uint32_t *position_ids, const uint8_t *tags,
                        int layer_count,
                        h3_text_progress progress, void *progress_opaque,
                        h3_text_embedding *output,
                        char *error, size_t error_size) {
    if (!spans || !span_count || !position_ids || !tags) {
        if (output) memset(output, 0, sizeof(*output));
        fail(error, error_size, "multimodal Qwen presentation is incomplete");
        return 0;
    }
    return text_encode_bf16_impl(
        weight_directory, shader_source_path, token_ids, token_count,
        spans, span_count, position_ids, tags, layer_count,
        progress, progress_opaque, output, error, error_size);
}

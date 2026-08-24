#include "h3_dit_schedule.h"

#include <math.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    TIME_INPUT = 256,
    TIME_HIDDEN = 5376,
    BLOCK_OUTPUT = H3_DIT_MODALITIES * H3_DIT_ADALN_SLOTS * H3_DIT_HIDDEN,
    FINAL_OUTPUT = 2 * H3_DIT_HIDDEN
};

struct h3_dit_schedule {
    h3_gpu *gpu;
    int steps;
    uint32_t time_rows;
    uint32_t *video_rows;
    uint32_t *audio_rows;
    uint32_t *visual_condition_rows;
    uint32_t *audio_condition_rows;
    h3_gpu_tensor *blocks[H3_DIT_BLOCKS];
    h3_gpu_tensor *final;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int gpu_op(h3_gpu *gpu, int ok, char *error, size_t error_size,
                  const char *operation) {
    if (ok) return 1;
    fail(error, error_size, "%s: %s", operation, h3_gpu_error(gpu));
    return 0;
}

static h3_gpu_tensor *weight_f32_1d(const h3_weight_store *store, h3_gpu *gpu,
                                    const char *name, uint64_t width,
                                    char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_f32(store, gpu, name, 1, shape, error, error_size);
}

static h3_gpu_tensor *weight_f32_2d(const h3_weight_store *store, h3_gpu *gpu,
                                    const char *name, uint64_t rows,
                                    uint64_t columns, char *error,
                                    size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_f32(store, gpu, name, 2, shape, error, error_size);
}

static h3_gpu_tensor *weight_bf16_1d(const h3_weight_store *store, h3_gpu *gpu,
                                     const char *name, uint64_t width,
                                     char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_bf16(store, gpu, name, 1, shape, error, error_size);
}

static h3_gpu_tensor *weight_bf16_2d(const h3_weight_store *store, h3_gpu *gpu,
                                     const char *name, uint64_t rows,
                                     uint64_t columns, char *error,
                                     size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_bf16(store, gpu, name, 2, shape, error, error_size);
}

static void free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

static int prepare_rows(h3_dit_schedule *schedule,
                        const h3_sigma_schedule *sigmas,
                        int visual_condition, int audio_condition,
                        float **features_out, char *error,
                        size_t error_size) {
    schedule->steps = sigmas->steps;
    schedule->video_rows = calloc((size_t)sigmas->steps,
                                  sizeof(*schedule->video_rows));
    schedule->audio_rows = calloc((size_t)sigmas->steps,
                                  sizeof(*schedule->audio_rows));
    if (visual_condition)
        schedule->visual_condition_rows = calloc(
            (size_t)sigmas->steps, sizeof(*schedule->visual_condition_rows));
    if (audio_condition)
        schedule->audio_condition_rows = calloc(
            (size_t)sigmas->steps, sizeof(*schedule->audio_condition_rows));
    if (!schedule->video_rows || !schedule->audio_rows ||
        (visual_condition && !schedule->visual_condition_rows) ||
        (audio_condition && !schedule->audio_condition_rows)) {
        fail(error, error_size, "out of memory allocating timestep row maps");
        return 0;
    }
    uint32_t count = 0;
    for (int step = 0; step < sigmas->steps; step++) {
        float video = 1.0f - sigmas->video[step];
        float audio = 1.0f - sigmas->audio[step];
        if (video == audio) {
            schedule->video_rows[step] = count;
            schedule->audio_rows[step] = count++;
        } else if (video < audio) {
            schedule->video_rows[step] = count++;
            schedule->audio_rows[step] = count++;
        } else {
            schedule->audio_rows[step] = count++;
            schedule->video_rows[step] = count++;
        }
    }
    uint32_t visual_condition_row = UINT32_MAX;
    uint32_t audio_condition_row = UINT32_MAX;
    if (visual_condition) visual_condition_row = count++;
    if (audio_condition) audio_condition_row = count++;
    for (int step = 0; step < sigmas->steps; step++) {
        float video = 1.0f - sigmas->video[step];
        float audio = 1.0f - sigmas->audio[step];
        if (visual_condition)
            schedule->visual_condition_rows[step] = video >= 0.999f ?
                schedule->video_rows[step] : visual_condition_row;
        if (audio_condition)
            schedule->audio_condition_rows[step] = audio >= 1.0f ?
                schedule->audio_rows[step] : audio_condition_row;
    }
    schedule->time_rows = count;
    if (!count || count > UINT32_MAX / TIME_INPUT) {
        fail(error, error_size, "invalid number of timestep rows");
        return 0;
    }
    float *times = calloc(count, sizeof(*times));
    float *features = malloc((size_t)count * TIME_INPUT * sizeof(*features));
    if (!times || !features) {
        free(times);
        free(features);
        fail(error, error_size, "out of memory allocating timestep features");
        return 0;
    }
    for (int step = 0; step < sigmas->steps; step++) {
        times[schedule->video_rows[step]] = 1.0f - sigmas->video[step];
        times[schedule->audio_rows[step]] = 1.0f - sigmas->audio[step];
    }
    if (visual_condition) times[visual_condition_row] = 0.999f;
    if (audio_condition) times[audio_condition_row] = 1.0f;
    for (uint32_t row = 0; row < count; row++) {
        for (uint32_t index = 0; index < TIME_INPUT / 2; index++) {
            float frequency = expf(-logf(10000.0f) *
                                   (float)index / (float)(TIME_INPUT / 2));
            float angle = times[row] * frequency;
            features[(size_t)row * TIME_INPUT + index] = cosf(angle);
            features[(size_t)row * TIME_INPUT + TIME_INPUT / 2 + index] =
                sinf(angle);
        }
    }
    free(times);
    *features_out = features;
    return 1;
}

static h3_gpu_tensor *time_embeddings(const h3_weight_store *weights,
                                      h3_gpu *gpu, uint32_t rows,
                                      const float *features, char *error,
                                      size_t error_size) {
    h3_gpu_tensor *input = h3_gpu_tensor_from_f32(
        gpu, features, (size_t)rows * TIME_INPUT);
    h3_gpu_tensor *in_w = weight_f32_2d(weights, gpu,
        "time_embedder.proj_in.weight", TIME_HIDDEN, TIME_INPUT,
        error, error_size);
    h3_gpu_tensor *in_b = weight_f32_1d(weights, gpu,
        "time_embedder.proj_in.bias", TIME_HIDDEN, error, error_size);
    h3_gpu_tensor *out_w = weight_f32_2d(weights, gpu,
        "time_embedder.proj_out.weight", H3_DIT_TIME_DIM, TIME_HIDDEN,
        error, error_size);
    h3_gpu_tensor *out_b = weight_f32_1d(weights, gpu,
        "time_embedder.proj_out.bias", H3_DIT_TIME_DIM, error, error_size);
    h3_gpu_tensor *hidden = h3_gpu_tensor_new_f32(
        gpu, (size_t)rows * TIME_HIDDEN);
    h3_gpu_tensor *activated = h3_gpu_tensor_new_f32(
        gpu, (size_t)rows * TIME_HIDDEN);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(
        gpu, (size_t)rows * H3_DIT_TIME_DIM);
    h3_gpu_tensor *bf16 = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * H3_DIT_TIME_DIM);
    h3_gpu_tensor *silu = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * H3_DIT_TIME_DIM);
    h3_gpu_tensor *result = NULL;
    h3_gpu_tensor *all[] = {input, in_w, in_b, out_w, out_b, hidden,
                            activated, output, bf16, silu};
    for (size_t index = 0; index < sizeof(all) / sizeof(*all); index++) {
        if (!all[index]) {
            if (!error || !*error)
                fail(error, error_size, "cannot allocate timestep tensors: %s",
                     h3_gpu_error(gpu));
            goto cleanup;
        }
    }
    if (!gpu_op(gpu, h3_gpu_begin(gpu), error, error_size,
                "begin timestep embedding") ||
        !gpu_op(gpu, h3_gpu_linear_f32(gpu, hidden, input, in_w, in_b, rows,
                                       TIME_INPUT, TIME_HIDDEN),
                error, error_size, "timestep input projection") ||
        !gpu_op(gpu, h3_gpu_silu_f32(gpu, activated, hidden,
                                     rows * TIME_HIDDEN),
                error, error_size, "timestep SiLU") ||
        !gpu_op(gpu, h3_gpu_linear_f32(gpu, output, activated, out_w, out_b,
                                       rows, TIME_HIDDEN, H3_DIT_TIME_DIM),
                error, error_size, "timestep output projection") ||
        !gpu_op(gpu, h3_gpu_cast_f32_to_bf16(
                        gpu, bf16, output, rows * H3_DIT_TIME_DIM),
                error, error_size, "timestep BF16 cast") ||
        !gpu_op(gpu, h3_gpu_silu_bf16(gpu, silu, bf16,
                                      rows * H3_DIT_TIME_DIM),
                error, error_size, "timestep AdaLN SiLU") ||
        !gpu_op(gpu, h3_gpu_submit(gpu), error, error_size,
                "submit timestep embedding")) {
        goto cleanup;
    }
    result = silu;
    silu = NULL;
cleanup:
    free_tensor(&input);
    free_tensor(&in_w);
    free_tensor(&in_b);
    free_tensor(&out_w);
    free_tensor(&out_b);
    free_tensor(&hidden);
    free_tensor(&activated);
    free_tensor(&output);
    free_tensor(&bf16);
    free_tensor(&silu);
    return result;
}


#define H3_ADALN_CACHE_ALIGN UINT64_C(4096)

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t steps;
    uint32_t time_rows;
    uint32_t visual_condition;
    uint32_t audio_condition;
    uint64_t key;
    uint64_t total_bytes;
} h3_adaln_cache_header;

static uint64_t adaln_align(uint64_t value) {
    return (value + H3_ADALN_CACHE_ALIGN - 1) & ~(H3_ADALN_CACHE_ALIGN - 1);
}

static uint64_t adaln_hash_bytes(uint64_t hash, const void *data, size_t size) {
    const unsigned char *p = data;
    for (size_t i = 0; i < size; i++) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t adaln_schedule_key(const h3_sigma_schedule *sigmas,
                                   int visual_condition,
                                   int audio_condition) {
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = adaln_hash_bytes(hash, &sigmas->steps, sizeof(sigmas->steps));
    hash = adaln_hash_bytes(hash, sigmas->video,
        ((size_t)sigmas->steps + 1) * sizeof(sigmas->video[0]));
    hash = adaln_hash_bytes(hash, sigmas->audio,
        ((size_t)sigmas->steps + 1) * sizeof(sigmas->audio[0]));
    hash = adaln_hash_bytes(hash, &visual_condition, sizeof(visual_condition));
    hash = adaln_hash_bytes(hash, &audio_condition, sizeof(audio_condition));
    return hash;
}

static uint64_t adaln_block_elements(const h3_dit_schedule *schedule) {
    return (uint64_t)schedule->time_rows * BLOCK_OUTPUT;
}

static uint64_t adaln_final_elements(const h3_dit_schedule *schedule) {
    return (uint64_t)schedule->time_rows * FINAL_OUTPUT;
}

static uint64_t adaln_block_offset(const h3_dit_schedule *schedule,
                                   unsigned wanted) {
    uint64_t offset = H3_ADALN_CACHE_ALIGN;
    uint64_t bytes = adaln_block_elements(schedule) * sizeof(uint16_t);
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
        offset = adaln_align(offset);
        if (block == wanted) return offset;
        offset += bytes;
    }
    return 0;
}

static uint64_t adaln_final_offset(const h3_dit_schedule *schedule) {
    uint64_t offset = H3_ADALN_CACHE_ALIGN;
    uint64_t bytes = adaln_block_elements(schedule) * sizeof(uint16_t);
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
        offset = adaln_align(offset);
        offset += bytes;
    }
    return adaln_align(offset);
}

static uint64_t adaln_total_bytes(const h3_dit_schedule *schedule) {
    return adaln_align(adaln_final_offset(schedule) +
        adaln_final_elements(schedule) * sizeof(uint16_t));
}

static int adaln_cache_path(char *path, size_t size,
                            const h3_sigma_schedule *sigmas,
                            int visual_condition, int audio_condition,
                            uint64_t *key_out) {
    const char *dir = getenv("H3_ADALN_CACHE_DIR");
    if (!dir || !*dir) return 0;
    uint64_t key = adaln_schedule_key(sigmas, visual_condition, audio_condition);
    int n = snprintf(path, size, "%s/adaln-v1-%016llx.bin", dir,
                     (unsigned long long)key);
    if (n < 0 || (size_t)n >= size) return 0;
    if (key_out) *key_out = key;
    return 1;
}

static void adaln_cache_drop_tensors(h3_dit_schedule *schedule) {
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
        h3_gpu_tensor_free(schedule->blocks[block]);
        schedule->blocks[block] = NULL;
    }
    h3_gpu_tensor_free(schedule->final);
    schedule->final = NULL;
}

static int adaln_cache_try_load(h3_dit_schedule *schedule,
                                const h3_sigma_schedule *sigmas,
                                int visual_condition, int audio_condition,
                                h3_dit_schedule_progress progress,
                                void *progress_opaque,
                                char *error, size_t error_size) {
    char path[1024];
    uint64_t key = 0;
    if (!adaln_cache_path(path, sizeof(path), sigmas, visual_condition,
                          audio_condition, &key)) return 0;
    FILE *stream = fopen(path, "rb");
    if (!stream) return 0;
    h3_adaln_cache_header header;
    size_t got = fread(&header, 1, sizeof(header), stream);
    fclose(stream);
    static const char magic[8] = {'H','3','A','D','L','N','1','\0'};
    struct stat st;
    if (got != sizeof(header) || memcmp(header.magic, magic, sizeof(magic)) ||
        header.version != 1 || header.steps != (uint32_t)sigmas->steps ||
        header.time_rows != schedule->time_rows ||
        header.visual_condition != (uint32_t)(visual_condition != 0) ||
        header.audio_condition != (uint32_t)(audio_condition != 0) ||
        header.key != key || stat(path, &st) != 0 || st.st_size < 0 ||
        (uint64_t)st.st_size < header.total_bytes ||
        header.total_bytes != adaln_total_bytes(schedule)) return 0;
    size_t block_elements = (size_t)adaln_block_elements(schedule);
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
        schedule->blocks[block] = h3_gpu_tensor_load_bf16(
            schedule->gpu, path, adaln_block_offset(schedule, block),
            block_elements);
        if (!schedule->blocks[block]) {
            adaln_cache_drop_tensors(schedule);
            if (error && error_size) error[0] = '\0';
            return 0;
        }
        if (progress) progress((int)block + 1, (int)H3_DIT_BLOCKS,
                               progress_opaque);
    }
    schedule->final = h3_gpu_tensor_load_bf16(
        schedule->gpu, path, adaln_final_offset(schedule),
        (size_t)adaln_final_elements(schedule));
    if (!schedule->final) {
        adaln_cache_drop_tensors(schedule);
        if (error && error_size) error[0] = '\0';
        return 0;
    }
    if (getenv("H3_PROFILE"))
        fprintf(stderr, "h3: AdaLN cache hit %s (%.1f MiB)\n", path,
                (double)header.total_bytes / (1024.0 * 1024.0));
    return 1;
}

static int adaln_cache_write(h3_dit_schedule *schedule,
                             const h3_sigma_schedule *sigmas,
                             int visual_condition, int audio_condition,
                             char *detail, size_t detail_size) {
    char path[1024];
    uint64_t key = 0;
    if (!adaln_cache_path(path, sizeof(path), sigmas, visual_condition,
                          audio_condition, &key)) return 1;
    const char *dir = getenv("H3_ADALN_CACHE_DIR");
    if (mkdir(dir, 0755) != 0 && errno != EEXIST) {
        snprintf(detail, detail_size, "cannot create AdaLN cache dir: %s",
                 strerror(errno));
        return 0;
    }
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) {
        snprintf(detail, detail_size, "cannot create AdaLN cache: %s",
                 strerror(errno));
        return 0;
    }
    h3_adaln_cache_header header = {
        .magic = {'H','3','A','D','L','N','1','\0'},
        .version = 0,
        .steps = (uint32_t)sigmas->steps,
        .time_rows = schedule->time_rows,
        .visual_condition = (uint32_t)(visual_condition != 0),
        .audio_condition = (uint32_t)(audio_condition != 0),
        .key = key,
        .total_bytes = adaln_total_bytes(schedule)
    };
    int ok = pwrite(fd, &header, sizeof(header), 0) == (ssize_t)sizeof(header) &&
             ftruncate(fd, (off_t)header.total_bytes) == 0;
    close(fd);
    if (!ok) {
        snprintf(detail, detail_size, "cannot initialize AdaLN cache: %s",
                 strerror(errno));
        return 0;
    }
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
        if (!h3_gpu_tensor_write_file_raw_at(
                schedule->blocks[block], path,
                adaln_block_offset(schedule, block), detail, detail_size))
            return 0;
    }
    if (!h3_gpu_tensor_write_file_raw_at(
            schedule->final, path, adaln_final_offset(schedule),
            detail, detail_size)) return 0;
    header.version = 1;
    fd = open(path, O_WRONLY | O_CLOEXEC);
    if (fd < 0) return 0;
    ok = pwrite(fd, &header, sizeof(header), 0) == (ssize_t)sizeof(header) &&
         fsync(fd) == 0;
    close(fd);
    if (getenv("H3_PROFILE") && ok)
        fprintf(stderr, "h3: AdaLN cache stored %s (%.1f MiB)\n", path,
                (double)header.total_bytes / (1024.0 * 1024.0));
    return ok;
}

h3_dit_schedule *h3_dit_schedule_precompute(
    const h3_weight_store *weights, h3_gpu *gpu,
    const h3_sigma_schedule *sigmas, int visual_condition,
    int audio_condition,
    h3_dit_schedule_progress progress, void *progress_opaque,
    char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!weights || !gpu || !sigmas || sigmas->steps < 1 ||
        sigmas->steps > H3_MAX_STEPS) {
        fail(error, error_size, "invalid AdaLN schedule arguments");
        return NULL;
    }
    h3_dit_schedule *schedule = calloc(1, sizeof(*schedule));
    if (!schedule) {
        fail(error, error_size, "out of memory creating AdaLN schedule");
        return NULL;
    }
    schedule->gpu = gpu;
    float *features = NULL;
    if (!prepare_rows(schedule, sigmas, visual_condition, audio_condition,
                      &features, error, error_size)) goto failed;
    if (adaln_cache_try_load(schedule, sigmas, visual_condition,
                             audio_condition, progress, progress_opaque,
                             error, error_size)) {
        free(features);
        return schedule;
    }
    h3_gpu_tensor *time = time_embeddings(weights, gpu, schedule->time_rows,
                                           features, error, error_size);
    free(features);
    features = NULL;
    if (!time) goto failed;

    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
        char weight_name[128], bias_name[128], operation[128];
        snprintf(weight_name, sizeof(weight_name),
                 "blocks.%u.adaln_proj.linear.weight", block);
        snprintf(bias_name, sizeof(bias_name),
                 "blocks.%u.adaln_proj.linear.bias", block);
        h3_gpu_tensor *weight = weight_bf16_2d(
            weights, gpu, weight_name, BLOCK_OUTPUT, H3_DIT_TIME_DIM,
            error, error_size);
        h3_gpu_tensor *bias = weight_bf16_1d(
            weights, gpu, bias_name, BLOCK_OUTPUT, error, error_size);
        schedule->blocks[block] = h3_gpu_tensor_new_bf16(
            gpu, (size_t)schedule->time_rows * BLOCK_OUTPUT);
        if (!weight || !bias || !schedule->blocks[block]) {
            if (!error || !*error)
                fail(error, error_size, "cannot allocate AdaLN block %u: %s",
                     block, h3_gpu_error(gpu));
            free_tensor(&weight);
            free_tensor(&bias);
            h3_gpu_tensor_free(time);
            goto failed;
        }
        snprintf(operation, sizeof(operation), "AdaLN block %u", block);
        int ok = gpu_op(gpu, h3_gpu_begin(gpu), error, error_size, operation) &&
            gpu_op(gpu, h3_gpu_linear_bf16(
                gpu, schedule->blocks[block], time, weight, bias,
                schedule->time_rows, H3_DIT_TIME_DIM, BLOCK_OUTPUT),
                error, error_size, operation) &&
            gpu_op(gpu, h3_gpu_submit(gpu), error, error_size, operation);
        free_tensor(&weight);
        free_tensor(&bias);
        if (!ok) {
            h3_gpu_tensor_free(time);
            goto failed;
        }
        if (progress) progress((int)block + 1, (int)H3_DIT_BLOCKS,
                               progress_opaque);
    }

    h3_gpu_tensor *final_w = weight_bf16_2d(
        weights, gpu, "final_layer.adaln_proj.linear.weight",
        FINAL_OUTPUT, H3_DIT_TIME_DIM, error, error_size);
    h3_gpu_tensor *final_b = weight_bf16_1d(
        weights, gpu, "final_layer.adaln_proj.linear.bias",
        FINAL_OUTPUT, error, error_size);
    schedule->final = h3_gpu_tensor_new_bf16(
        gpu, (size_t)schedule->time_rows * FINAL_OUTPUT);
    if (!final_w || !final_b || !schedule->final ||
        !gpu_op(gpu, h3_gpu_begin(gpu), error, error_size,
                "begin final AdaLN") ||
        !gpu_op(gpu, h3_gpu_linear_bf16(
            gpu, schedule->final, time, final_w, final_b, schedule->time_rows,
            H3_DIT_TIME_DIM, FINAL_OUTPUT), error, error_size,
            "final AdaLN projection") ||
        !gpu_op(gpu, h3_gpu_submit(gpu), error, error_size,
                "submit final AdaLN")) {
        if ((!error || !*error) && (!final_w || !final_b || !schedule->final))
            fail(error, error_size, "cannot allocate final AdaLN tensors: %s",
                 h3_gpu_error(gpu));
        free_tensor(&final_w);
        free_tensor(&final_b);
        h3_gpu_tensor_free(time);
        goto failed;
    }
    free_tensor(&final_w);
    free_tensor(&final_b);
    h3_gpu_tensor_free(time);
    {
        char cache_detail[384] = {0};
        if (!adaln_cache_write(schedule, sigmas, visual_condition,
                               audio_condition, cache_detail,
                               sizeof(cache_detail)))
            fprintf(stderr, "h3: warning: AdaLN cache write failed: %s\n",
                    cache_detail[0] ? cache_detail : "unknown error");
    }
    return schedule;

failed:
    free(features);
    h3_dit_schedule_free(schedule);
    return NULL;
}

void h3_dit_schedule_free(h3_dit_schedule *schedule) {
    if (!schedule) return;
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++)
        h3_gpu_tensor_free(schedule->blocks[block]);
    h3_gpu_tensor_free(schedule->final);
    free(schedule->video_rows);
    free(schedule->audio_rows);
    free(schedule->visual_condition_rows);
    free(schedule->audio_condition_rows);
    free(schedule);
}

int h3_dit_schedule_steps(const h3_dit_schedule *schedule) {
    return schedule ? schedule->steps : 0;
}

uint32_t h3_dit_schedule_time_rows(const h3_dit_schedule *schedule) {
    return schedule ? schedule->time_rows : 0;
}

uint32_t h3_dit_schedule_video_row(const h3_dit_schedule *schedule, int step) {
    return schedule && step >= 0 && step < schedule->steps ?
        schedule->video_rows[step] : UINT32_MAX;
}

uint32_t h3_dit_schedule_audio_row(const h3_dit_schedule *schedule, int step) {
    return schedule && step >= 0 && step < schedule->steps ?
        schedule->audio_rows[step] : UINT32_MAX;
}

uint32_t h3_dit_schedule_visual_condition_row(
    const h3_dit_schedule *schedule, int step) {
    return schedule && schedule->visual_condition_rows && step >= 0 &&
        step < schedule->steps ? schedule->visual_condition_rows[step] :
        UINT32_MAX;
}

uint32_t h3_dit_schedule_audio_condition_row(
    const h3_dit_schedule *schedule, int step) {
    return schedule && schedule->audio_condition_rows && step >= 0 &&
        step < schedule->steps ? schedule->audio_condition_rows[step] :
        UINT32_MAX;
}

const h3_gpu_tensor *h3_dit_schedule_block(const h3_dit_schedule *schedule,
                                           unsigned block) {
    return schedule && block < H3_DIT_BLOCKS ? schedule->blocks[block] : NULL;
}

double h3_dit_schedule_gate_score(const h3_dit_schedule *schedule,
                                  unsigned block) {
    if (!schedule || block >= H3_DIT_BLOCKS || !schedule->blocks[block])
        return -1.0;
    size_t count = (size_t)schedule->time_rows * BLOCK_OUTPUT;
    uint16_t *values = malloc(count * sizeof(*values));
    if (!values || !h3_gpu_tensor_read_bf16(schedule->blocks[block], values,
                                             count)) {
        free(values);
        return -1.0;
    }
    double total = 0.0;
    size_t samples = 0;
    for (uint32_t row = 0; row < schedule->time_rows; row++)
        for (uint32_t modality = 0; modality < H3_DIT_MODALITIES; modality++)
            for (uint32_t slot = 2; slot <= 5; slot += 3) {
                size_t base = ((size_t)row * H3_DIT_MODALITIES *
                               H3_DIT_ADALN_SLOTS +
                               (size_t)modality * H3_DIT_ADALN_SLOTS + slot) *
                              H3_DIT_HIDDEN;
                for (uint32_t column = 0; column < H3_DIT_HIDDEN; column++) {
                    uint32_t bits = (uint32_t)values[base + column] << 16;
                    float value;
                    memcpy(&value, &bits, sizeof(value));
                    total += fabs((double)value);
                }
                samples += H3_DIT_HIDDEN;
            }
    free(values);
    return samples ? total / (double)samples : -1.0;
}

void h3_dit_schedule_prune(h3_dit_schedule *schedule,
                           const uint8_t *active_blocks, size_t count) {
    if (!schedule || !active_blocks || count != H3_DIT_BLOCKS) return;
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
        if (active_blocks[block]) continue;
        h3_gpu_tensor_free(schedule->blocks[block]);
        schedule->blocks[block] = NULL;
    }
}

const h3_gpu_tensor *h3_dit_schedule_final(const h3_dit_schedule *schedule) {
    return schedule ? schedule->final : NULL;
}

int h3_dit_schedule_row_map(const h3_dit_schedule *schedule, int step,
                            const h3_layout *layout,
                            const uint8_t *text_tags, size_t text_tag_count,
                            uint32_t *rows, size_t row_count) {
    if (!schedule || step < 0 || step >= schedule->steps || !layout || !rows ||
        row_count != layout->seq_len || !layout->segments ||
        (text_tags && text_tag_count != (size_t)layout->signature[0])) return 0;
    size_t text_index = 0;
    for (size_t seg_index = 0; seg_index < layout->segment_count; seg_index++) {
        const h3_segment *segment = &layout->segments[seg_index];
        if (segment->start > segment->stop || segment->stop > row_count)
            return 0;
        uint32_t time_row;
        uint32_t tag;
        switch (segment->kind) {
        case H3_SEG_TEXT:
            time_row = schedule->video_rows[step];
            for (size_t row = segment->start; row < segment->stop; row++) {
                uint32_t text_tag = text_tags ? text_tags[text_index] : 1u;
                if (text_tag >= H3_DIT_MODALITIES) return 0;
                rows[row] = time_row * H3_DIT_MODALITIES + text_tag;
                text_index++;
            }
            continue;
        case H3_SEG_COND:
        case H3_SEG_REF_IMAGE:
            if (!schedule->visual_condition_rows) return 0;
            time_row = schedule->visual_condition_rows[step];
            tag = 0;
            break;
        case H3_SEG_REF_AUDIO:
            if (!schedule->audio_condition_rows) return 0;
            time_row = schedule->audio_condition_rows[step];
            tag = 2;
            break;
        case H3_SEG_AUDIO:
            time_row = schedule->audio_rows[step];
            tag = 2;
            break;
        case H3_SEG_VIDEO:
            time_row = schedule->video_rows[step];
            tag = 0;
            break;
        default:
            return 0;
        }
        uint32_t modulation = time_row * H3_DIT_MODALITIES + tag;
        for (size_t row = segment->start; row < segment->stop; row++)
            rows[row] = modulation;
    }
    return text_index == (size_t)layout->signature[0];
}

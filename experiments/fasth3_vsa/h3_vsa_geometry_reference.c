#include "h3_dit.h"

#include "h3_dit_schedule.h"
#include "h3_weights.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* Casting three operands up and the result back costs time linear in the
 * rows; attention itself is quadratic in them, so the trade turns over as the
 * sequence grows. Measured at H3's 56 heads of 128: f32 wins by 1.87x at 1625
 * rows, 1.76x at 5095 and 1.72x at 8192. The threshold is where Z-Image put
 * it, below which the casts stop paying for themselves. */
#define H3_DIT_F32_ATTENTION_ROWS 512u
#define H3_DIT_SOL_ATTENTION_ROWS 4096u

enum {
    TEXT_DIM = 5120,
    HIDDEN = 5376,
    HEADS = 56,
    HEAD_DIM = 128,
    INNER = HEADS * HEAD_DIM,
    FFN = 14336,
    VIDEO_CHANNELS = 24,
    VIDEO_PATCH = 96,
    AUDIO_CHANNELS = 32,
    AUDIO_STREAMS = 2,
    ROPE_FREQS = 16,
    ROPE_HALF = 48,
    SLOTS = 6,
    FINAL_SLOTS = 2
};

static void fail(char *error, size_t error_size, const char *format, ...);

void h3_vsa_geometry_free(h3_vsa_geometry *geometry) {
    if (!geometry) return;
    free(geometry->packed_to_tiled);
    free(geometry->tiled_to_packed);
    free(geometry->block_sizes);
    memset(geometry, 0, sizeof(*geometry));
}

int h3_vsa_geometry_build(const uint32_t *prefix_segments,
                          size_t prefix_segment_count,
                          uint32_t video_t, uint32_t video_h,
                          uint32_t video_w, h3_vsa_geometry *geometry,
                          char *error, size_t error_size) {
    if (!geometry || (!prefix_segments && prefix_segment_count) ||
        !video_t || !video_h || !video_w) {
        fail(error, error_size, "invalid VSA-H3 tile geometry arguments");
        return 0;
    }
    memset(geometry, 0, sizeof(*geometry));
    uint64_t prefix_rows = 0;
    uint64_t prefix_tiles = 0;
    for (size_t index = 0; index < prefix_segment_count; index++) {
        prefix_rows += prefix_segments[index];
        prefix_tiles += (prefix_segments[index] + 63u) / 64u;
    }
    uint64_t video_rows = (uint64_t)video_t * video_h * video_w;
    uint64_t tile_t = (video_t + 3u) / 4u;
    uint64_t tile_h = (video_h + 3u) / 4u;
    uint64_t tile_w = (video_w + 3u) / 4u;
    uint64_t video_tiles = tile_t * tile_h * tile_w;
    uint64_t tiles = prefix_tiles + video_tiles;
    uint64_t sequence = prefix_rows + video_rows;
    uint64_t padded = tiles * 64u;
    if (!sequence || sequence > UINT32_MAX || tiles > UINT32_MAX ||
        padded > UINT32_MAX || sequence > SIZE_MAX / sizeof(uint32_t) ||
        padded > SIZE_MAX / sizeof(uint32_t) ||
        tiles > SIZE_MAX / sizeof(uint32_t)) {
        fail(error, error_size, "VSA-H3 tile geometry is too large");
        return 0;
    }
    geometry->packed_to_tiled = malloc((size_t)sequence * sizeof(uint32_t));
    geometry->tiled_to_packed = malloc((size_t)padded * sizeof(uint32_t));
    geometry->block_sizes = malloc((size_t)tiles * sizeof(uint32_t));
    if (!geometry->packed_to_tiled || !geometry->tiled_to_packed ||
        !geometry->block_sizes) {
        h3_vsa_geometry_free(geometry);
        fail(error, error_size, "out of memory building VSA-H3 tile geometry");
        return 0;
    }
    for (uint64_t index = 0; index < padded; index++)
        geometry->tiled_to_packed[index] = UINT32_MAX;

    uint32_t packed_row = 0;
    uint32_t tile = 0;
    for (size_t segment = 0; segment < prefix_segment_count; segment++) {
        uint32_t left = prefix_segments[segment];
        while (left) {
            uint32_t count = left < 64u ? left : 64u;
            geometry->block_sizes[tile] = count;
            for (uint32_t offset = 0; offset < count; offset++) {
                uint32_t tiled_row = tile * 64u + offset;
                geometry->packed_to_tiled[packed_row] = tiled_row;
                geometry->tiled_to_packed[tiled_row] = packed_row++;
            }
            left -= count;
            tile++;
        }
    }
    for (uint32_t tt = 0; tt < video_t; tt += 4u)
        for (uint32_t hh = 0; hh < video_h; hh += 4u)
            for (uint32_t ww = 0; ww < video_w; ww += 4u) {
                uint32_t count = 0;
                uint32_t t_stop = tt + 4u < video_t ? tt + 4u : video_t;
                uint32_t h_stop = hh + 4u < video_h ? hh + 4u : video_h;
                uint32_t w_stop = ww + 4u < video_w ? ww + 4u : video_w;
                for (uint32_t t = tt; t < t_stop; t++)
                    for (uint32_t h = hh; h < h_stop; h++)
                        for (uint32_t w = ww; w < w_stop; w++) {
                            uint32_t original = (uint32_t)prefix_rows +
                                (t * video_h + h) * video_w + w;
                            uint32_t tiled_row = tile * 64u + count++;
                            geometry->packed_to_tiled[original] = tiled_row;
                            geometry->tiled_to_packed[tiled_row] = original;
                        }
                geometry->block_sizes[tile++] = count;
            }
    if (packed_row != (uint32_t)prefix_rows || tile != (uint32_t)tiles) {
        h3_vsa_geometry_free(geometry);
        fail(error, error_size, "VSA-H3 tile geometry construction drifted");
        return 0;
    }
    geometry->sequence = (uint32_t)sequence;
    geometry->padded_rows = (uint32_t)padded;
    geometry->tiles = (uint32_t)tiles;
    geometry->prefix_tiles = (uint32_t)prefix_tiles;
    geometry->video_tiles = (uint32_t)video_tiles;
    return 1;
}

typedef struct {
    h3_gpu_tensor *norm1;
    h3_gpu_tensor *norm2;
    h3_gpu_tensor *qkv;
    h3_gpu_tensor *qkv_int8;

#ifndef H3_VSA_GEOMETRY_H
#define H3_VSA_GEOMETRY_H
#include <stddef.h>
#include <stdint.h>
typedef struct {
    uint32_t sequence, padded_rows, tiles, prefix_tiles, video_tiles;
    uint32_t *packed_to_tiled, *tiled_to_packed, *block_sizes;
} h3_vsa_geometry;
int h3_vsa_geometry_build(const uint32_t *prefix_segments, size_t prefix_segment_count, uint32_t video_t, uint32_t video_h, uint32_t video_w, h3_vsa_geometry *g, char *error, size_t error_size);
void h3_vsa_geometry_free(h3_vsa_geometry *g);
#endif

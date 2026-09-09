#include "h3_vsa_geometry.h"
#include <stdio.h>
#include <stdint.h>
int main(void) {
    const uint32_t p[] = {3, 2, 4};
    h3_vsa_geometry g = {0};
    char e[256] = {0};
    if (!h3_vsa_geometry_build(p, 3, 5, 5, 9, &g, e, sizeof e)) {
        fprintf(stderr, "%s\n", e);
        return 1;
    }
    int ok = g.prefix_tiles == 3 && g.video_tiles == 12 && g.tiles == 15 &&
             g.sequence == 234 && g.padded_rows == 960;
    for (uint32_t i = 0; i < g.sequence && ok; i++) {
        uint32_t t = g.packed_to_tiled[i];
        if (!(t < g.padded_rows && g.tiled_to_packed[t] == i)) {
            fprintf(stderr, "mismatch i=%u tiled=%u back=%u\n",
                    i, t, t < g.padded_rows ? g.tiled_to_packed[t] : UINT32_MAX);
            ok = 0;
        }
    }
    printf("geometry seq=%u padded=%u tiles=%u prefix=%u video=%u inverse=%s\n",
           g.sequence, g.padded_rows, g.tiles, g.prefix_tiles, g.video_tiles,
           ok ? "ok" : "FAIL");
    h3_vsa_geometry_free(&g);
    return ok ? 0 : 2;
}

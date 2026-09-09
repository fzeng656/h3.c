#include "h3_vsa_geometry.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int bad(char *e,size_t n,const char *m){ if(e&&n) snprintf(e,n,"%s",m); return 0; }
void h3_vsa_geometry_free(h3_vsa_geometry *g){ if(!g)return; free(g->packed_to_tiled); free(g->tiled_to_packed); free(g->block_sizes); memset(g,0,sizeof(*g)); }
int h3_vsa_geometry_build(const uint32_t *p,size_t np,uint32_t vt,uint32_t vh,uint32_t vw,h3_vsa_geometry *g,char *e,size_t en){
 if(!g||(!p&&np)||!vt||!vh||!vw) return bad(e,en,"invalid VSA-H3 tile geometry arguments"); memset(g,0,sizeof(*g));
 uint64_t pr=0,pt=0; for(size_t i=0;i<np;i++){pr+=p[i]; pt+=(p[i]+63u)/64u;}
 uint64_t vr=(uint64_t)vt*vh*vw, vtiles=((vt+3u)/4u)*((vh+3u)/4u)*((vw+3u)/4u), tiles=pt+vtiles, seq=pr+vr, pad=tiles*64u;
 if(!seq||seq>UINT32_MAX||tiles>UINT32_MAX||pad>UINT32_MAX||seq>SIZE_MAX/4||pad>SIZE_MAX/4||tiles>SIZE_MAX/4) return bad(e,en,"VSA-H3 tile geometry is too large");
 g->packed_to_tiled=malloc((size_t)seq*4); g->tiled_to_packed=malloc((size_t)pad*4); g->block_sizes=malloc((size_t)tiles*4);
 if(!g->packed_to_tiled||!g->tiled_to_packed||!g->block_sizes){h3_vsa_geometry_free(g);return bad(e,en,"out of memory building VSA-H3 tile geometry");}
 for(uint64_t i=0;i<pad;i++) g->tiled_to_packed[i]=UINT32_MAX; uint32_t packed=0,tile=0;
 for(size_t s=0;s<np;s++){uint32_t left=p[s]; while(left){uint32_t count=left<64u?left:64u; g->block_sizes[tile]=count; for(uint32_t o=0;o<count;o++){uint32_t tr=tile*64u+o; g->packed_to_tiled[packed]=tr; g->tiled_to_packed[tr]=packed++;} left-=count; tile++;}}
 for(uint32_t tt=0;tt<vt;tt+=4u) for(uint32_t hh=0;hh<vh;hh+=4u) for(uint32_t ww=0;ww<vw;ww+=4u){uint32_t count=0,ts=tt+4u<vt?tt+4u:vt,hs=hh+4u<vh?hh+4u:vh,ws=ww+4u<vw?ww+4u:vw; for(uint32_t t=tt;t<ts;t++) for(uint32_t h=hh;h<hs;h++) for(uint32_t w=ww;w<ws;w++){uint32_t original=(uint32_t)pr+(t*vh+h)*vw+w,tr=tile*64u+count++; g->packed_to_tiled[original]=tr; g->tiled_to_packed[tr]=original;} g->block_sizes[tile++]=count;}
 if(packed!=(uint32_t)pr||tile!=(uint32_t)tiles){h3_vsa_geometry_free(g);return bad(e,en,"VSA-H3 tile geometry construction drifted");}
 g->sequence=(uint32_t)seq; g->padded_rows=(uint32_t)pad; g->tiles=(uint32_t)tiles; g->prefix_tiles=(uint32_t)pt; g->video_tiles=(uint32_t)vtiles; return 1; }

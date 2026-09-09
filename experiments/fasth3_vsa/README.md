# FastH3 VSA isolation experiment

Reference snapshot imported from local h3ddle vendor commit 31baf87.
Purpose: reproduce learned VSA tile-64 geometry/kernel parity on current Apple Metal compiler before integrating with h3.c runtime.

Current compiler compatibility fix: MMATile member functions explicitly qualify `this` as `thread`; required by Apple metal 32023.918.

Reference parity on this Mac: cosine 0.999998633, MAE 0.000065548, max 0.000706643.

## Verified on M5 Max / current Xcode beta

- Metal VSA parity: cosine 0.999998633, MAE 0.000065548, max 0.000706643.
- Geometry test: seq=234, padded=960, tiles=15, prefix=3, video=12, inverse=ok.
- Apple metal compiler 32023.918 requires explicit `thread` address-space qualifiers on MMATile member functions.

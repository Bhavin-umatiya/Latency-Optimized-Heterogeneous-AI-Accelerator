#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <hls_stream.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <hls_math.h>

// Vector length — set to 768 for ViT attention (matches bench_768.py)
#define VIT_N 768

typedef ap_fixed<16,8> Data;
typedef ap_axiu<16,0,0,0> pkt;

void vit_softmax(hls::stream<pkt>& in_stream, hls::stream<pkt>& out_stream);

#endif

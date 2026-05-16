#ifndef VIT_KERNELS_H
#define VIT_KERNELS_H

#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <hls_stream.h>

typedef ap_fixed<16, 8> Data;
// This 'pkt' structure includes the TLAST bit for the DMA
typedef ap_axiu<16, 0, 0, 0> pkt;

void vit_softmax(hls::stream<pkt> &in_stream, hls::stream<pkt> &out_stream);
void vit_gelu(hls::stream<pkt> &in_stream, hls::stream<pkt> &out_stream);

#endif

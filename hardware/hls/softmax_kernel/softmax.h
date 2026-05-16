#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <hls_stream.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>

typedef ap_fixed<16,8> data_t;
typedef ap_axiu<16,0,0,0> pkt;

void vit_softmax(hls::stream<pkt>& in, hls::stream<pkt>& out);

#endif

#include "gelu.h"
#include <hls_math.h>

void vit_gelu(hls::stream<pkt> &in_stream, hls::stream<pkt> &out_stream) {
    #pragma HLS interface axis port=in_stream
    #pragma HLS interface axis port=out_stream
    #pragma HLS interface s_axilite port=return bundle=control

    for (int i = 0; i < VIT_N; i++) {
#pragma HLS PIPELINE II=1
        pkt in_pkt = in_stream.read();
        Data x = in_pkt.data;

        // Fast GELU approximation: x * sigmoid(1.702 * x)
        // Reference: Hendrycks & Gimpel 2016, fast variant
        float x_f = (float)x;
        float gelu_f = x_f * (1.0f / (1.0f + hls::exp(-1.702f * x_f)));

        pkt out_pkt;
        out_pkt.data = (Data)gelu_f;
        out_pkt.last = (i == VIT_N - 1) ? 1 : 0; // TLAST: signals end of DMA packet
        out_pkt.keep = -1;
        out_stream.write(out_pkt);
    }
}

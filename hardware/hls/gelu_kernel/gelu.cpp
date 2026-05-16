#include "gelu.h"
#include <cmath>

void vit_gelu(hls::stream<pkt> &in_stream, hls::stream<pkt> &out_stream) {
    #pragma HLS interface axis port=in_stream
    #pragma HLS interface axis port=out_stream
    #pragma HLS interface s_axilite port=return bundle=control

    for(int i = 0; i < 128; i++) {
        pkt in_pkt = in_stream.read();
        Data x = in_pkt.data;
        
        // Fast GELU approximation: x * sigmoid(1.702 * x)
        float x_f = (float)x;
        float gelu_f = x_f * (1.0f / (1.0f + exp(-1.702f * x_f)));
        
        pkt out_pkt;
        out_pkt.data = (Data)gelu_f;
        out_pkt.last = (i == 127) ? 1 : 0; // The Magic Bit for DMA
        out_pkt.keep = -1;
        out_stream.write(out_pkt);
    }
}

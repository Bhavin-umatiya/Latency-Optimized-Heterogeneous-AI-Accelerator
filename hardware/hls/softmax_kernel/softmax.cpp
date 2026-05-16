#include "softmax.h"
#include <hls_math.h>

void vit_softmax(hls::stream<pkt> &in_stream, hls::stream<pkt> &out_stream) {
#pragma HLS interface axis port = in_stream
#pragma HLS interface axis port = out_stream
#pragma HLS interface s_axilite port = return bundle = control

  Data buffer[VIT_N];
#pragma HLS ARRAY_PARTITION variable=buffer cyclic factor=8 dim=1
  pkt temp_pkt;

  // 1. Read input
  for (int i = 0; i < VIT_N; i++) {
#pragma HLS PIPELINE II=1
    temp_pkt = in_stream.read();
    buffer[i] = temp_pkt.data;
  }

  // 2. Find Max (numerically stable softmax)
  Data max_val = buffer[0];
  for (int i = 1; i < VIT_N; i++) {
#pragma HLS PIPELINE II=1
    if (buffer[i] > max_val)
      max_val = buffer[i];
  }

  // 3. Compute exp and accumulate sum
  Data sum_exp = 0;
  for (int i = 0; i < VIT_N; i++) {
#pragma HLS PIPELINE II=1
    buffer[i] = (Data)hls::exp((float)(buffer[i] - max_val));
    sum_exp += buffer[i];
  }

  // 4. Normalize and Output with TLAST
  for (int i = 0; i < VIT_N; i++) {
#pragma HLS PIPELINE II=1
    pkt out_pkt;
    out_pkt.data = buffer[i] / sum_exp;
    out_pkt.last = (i == VIT_N - 1) ? 1 : 0; // TLAST: signals end of DMA packet
    out_pkt.keep = -1;
    out_stream.write(out_pkt);
  }
}

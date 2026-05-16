#include "softmax.h"
#include <cmath>

void vit_softmax(hls::stream<pkt> &in_stream, hls::stream<pkt> &out_stream) {
#pragma HLS interface axis port = in_stream
#pragma HLS interface axis port = out_stream
#pragma HLS interface s_axilite port = return bundle = control

  Data buffer[128];
  pkt temp_pkt;

  // 1. Read input
  for (int i = 0; i < 128; i++) {
    temp_pkt = in_stream.read();
    buffer[i] = temp_pkt.data;
  }

  // 2. Find Max
  Data max_val = buffer[0];
  for (int i = 1; i < 128; i++) {
    if (buffer[i] > max_val)
      max_val = buffer[i];
  }

  // 3. Compute exp and sum
  float sum_exp = 0;
  for (int i = 0; i < 128; i++) {
    buffer[i] = (Data)exp((float)(buffer[i] - max_val));
    sum_exp += (float)buffer[i];
  }

  // 4. Normalize and Output with TLAST
  for (int i = 0; i < 128; i++) {
    pkt out_pkt;
    out_pkt.data = buffer[i] / (Data)sum_exp;
    out_pkt.last = (i == 127) ? 1 : 0; // The Magic Bit for DMA
    out_pkt.keep = -1;
    out_stream.write(out_pkt);
  }
}

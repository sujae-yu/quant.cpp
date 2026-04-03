# quant.cpp Operator Specification v1

## PolarQuantize

**Inputs:** float32 tensor [N, D] where D is even
**Outputs:** block_tq_polar array, ceil(N * D / block_size) blocks
**Semantics:**
1. Split D dimensions into D/2 pairs: (x_i, y_i)
2. theta_i = atan2(y_i, x_i), normalized to [0, 2*PI]
3. radius_i = sqrt(x_i^2 + y_i^2)
4. Per-group min-max → scale = (max - min) / (2^bits)
5. Quantize: q = clamp(floor((val - min) / scale), 0, 2^bits - 1)
6. Pack: index = (rho << tbits) | theta

## QJLQuantize

**Inputs:** float32 tensor [D]
**Outputs:** block_tq_qjl
**Semantics:**
1. Generate projection matrix R ~ N(0,1) of shape [sketch_dim, D]
2. sketch = R @ input (matrix-vector product)
3. hash = pack_bits(sign(sketch))
4. norm = ||input||_2
5. outlier_idx = top-k dimensions by |input[d]|

## TurboAttention

**Inputs:** query float32 [D], quantized_keys void*, seq_len, head_dim
**Outputs:** scores float32 [seq_len]
**Semantics:**
1. For each key position: dequantize key
2. score[i] = dot(query, dequantized_key[i])

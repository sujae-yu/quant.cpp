# quant.cpp Format Specification v1

## Block Structures

### block_tq_polar (72 bytes, 128 elements)
| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 2 | rscale | Radius scale (fp16) |
| 2 | 2 | rmn | Radius minimum (fp16) |
| 4 | 2 | tscale | Theta scale (fp16) |
| 6 | 2 | tmn | Theta minimum (fp16) |
| 8 | 64 | indices | Packed rho|theta, 1 byte per pair |

Packing: `indices[i] = (rho << 2) | theta` for 2-bit each.

### block_tq_qjl (40 bytes)
| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 2 | norm | Key L2 norm (fp16) |
| 2 | 2 | outlier_norm | Outlier component norm (fp16) |
| 4 | 32 | hash | 1-bit sign packed, LSB-first |
| 36 | 4 | outlier_idx | Top-4 outlier dimension indices |

### block_tq_uniform_4b (68 bytes, 128 elements)
| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 2 | scale | (max-min)/(2^4-1) (fp16) |
| 2 | 2 | zero_point | Minimum value (fp16) |
| 4 | 64 | qs | 4-bit packed, 2 per byte, LSB-first |

## Bit Packing Convention

All bit packing follows ONNX LSB-first convention:
- 4-bit: `byte = (high_nibble << 4) | (low_nibble & 0x0F)`
- 2-bit: `byte = (v3 << 6) | (v2 << 4) | (v1 << 2) | (v0 & 0x03)`
- 1-bit: `byte = b7<<7 | b6<<6 | ... | b1<<1 | b0`

## Endianness

All multi-byte values are little-endian.

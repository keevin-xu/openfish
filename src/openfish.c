#include <openfish/openfish.h>
#include "decode.h"

#ifdef HAVE_NPU
#include "nn_npu.h"
#endif

openfish_opt_t openfish_decoder_default_opts(void) {
    openfish_opt_t opt = {100.0f, 2.0f, 0.0f, 1.0f};
    return opt;
}

size_t openfish_gpubuf_size(
    int n_timesteps,
    int batch_size,
    int state_len
) {
    const size_t num_states = (size_t)1 << (2 * state_len);
    return
        sizeof(float) * (size_t)batch_size * (n_timesteps + 1) * num_states +          // bwd_NTC
        sizeof(float) * (size_t)batch_size * (n_timesteps + 1) * num_states +          // post_NTC
        sizeof(uint8_t) * (size_t)batch_size * n_timesteps +                            // moves
        sizeof(char) * (size_t)batch_size * n_timesteps +                               // sequence
        sizeof(char) * (size_t)batch_size * n_timesteps +                               // qstring
        sizeof(beam_element_t) * (size_t)batch_size * MAX_BEAM_WIDTH * (n_timesteps + 1) + // beam_vector
        sizeof(state_t) * (size_t)batch_size * n_timesteps +                            // states
        sizeof(float) * (size_t)batch_size * n_timesteps * NUM_BASES +                  // qual_data
        sizeof(float) * (size_t)batch_size * n_timesteps +                              // base_probs
        sizeof(float) * (size_t)batch_size * n_timesteps;                               // total_probs
}

#ifdef HAVE_NPU
void openfish_silu_mul_npu(
    const float *in,
    float *out,
    uint64_t n_tokens,
    uint64_t hidden_dim
) {
    silu_mul_npu(in, out, n_tokens, hidden_dim);
}

int openfish_npu_op_enabled(const char *op) {
    return npu_op_enabled(op);
}

void openfish_linear_npu(
    const char *op,
    const float *in,
    float *out,
    const float *weight,
    uint64_t n_rows,
    uint64_t in_dim,
    uint64_t out_dim
) {
    linear_npu(op, in, out, weight, n_rows, in_dim, out_dim);
}

void openfish_rmsnorm_npu(const float *in, const float *weight, float *out, uint64_t n_rows, uint64_t dim) {
    rmsnorm_npu(in, weight, out, n_rows, dim);
}

void openfish_rope_npu(const float *in, const float *lut, float *out, uint64_t n_rows, uint64_t head_dim) {
    rope_npu(in, lut, out, n_rows, head_dim);
}

void openfish_silu_mul_npu_bf16(const uint16_t *in, uint16_t *out, uint64_t n_tokens, uint64_t hidden_dim) {
    silu_mul_npu_bf16(in, out, n_tokens, hidden_dim);
}
void openfish_linear_npu_bf16(const char *op, const uint16_t *in, uint16_t *out, const float *weight, uint64_t n_rows,
                              uint64_t in_dim, uint64_t out_dim) {
    linear_npu_bf16(op, in, out, weight, n_rows, in_dim, out_dim);
}
void openfish_rmsnorm_npu_bf16(const uint16_t *in, const float *weight, uint16_t *out, uint64_t n_rows, uint64_t dim) {
    rmsnorm_npu_bf16(in, weight, out, n_rows, dim);
}
void openfish_rope_npu_bf16(const uint16_t *in, const float *lut, uint16_t *out, uint64_t n_rows, uint64_t head_dim) {
    rope_npu_bf16(in, lut, out, n_rows, head_dim);
}
void openfish_linear_npu_bf16_f32(const char *op, const uint16_t *in, float *out, const float *weight, uint64_t n_rows,
                                  uint64_t in_dim, uint64_t out_dim) {
    linear_npu_bf16_f32(op, in, out, weight, n_rows, in_dim, out_dim);
}
void openfish_axpy_npu_bf16(float alpha, const uint16_t *x, const uint16_t *y, uint16_t *out, uint64_t n) {
    axpy_npu_bf16(alpha, x, y, out, n);
}
void openfish_silu_npu_bf16(const uint16_t *in, uint16_t *out, uint64_t n) {
    silu_npu_bf16(in, out, n);
}
void openfish_rms_qkv_rope_npu_bf16(const uint16_t *in, const float *norm_weight, const float *wqkv_weight, const float *lut,
                                    uint16_t *normed, uint16_t *qkv, uint64_t n_rows, uint64_t seq_len) {
    rms_qkv_rope_npu_bf16(in, norm_weight, wqkv_weight, lut, normed, qkv, n_rows, seq_len);
}
#endif

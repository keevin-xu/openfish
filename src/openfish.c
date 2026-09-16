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
#endif

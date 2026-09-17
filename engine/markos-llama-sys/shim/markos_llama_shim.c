// markos_llama_shim.c — plain-C surface over llama.h for markos-engine.
//
// Every function takes only scalars and opaque pointers: no llama.cpp struct
// ever crosses the FFI boundary, so the Rust side cannot drift out of sync
// with the fork's struct layouts. The shim is compiled against the exact
// fork it ships with (MARKOS_LLAMA_CPP_DIR at build time).
#include "llama.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

// ---- lifecycle ----

int markos_llama_backend_init(void) {
    llama_backend_init();
    return 0;
}

// n_gpu_layers > 0 lets the scheduler hand the graph to registered non-CPU
// backends — the ggml-axcl device, when whole-layer mode is armed.
void * markos_llama_model_load(const char * path, int n_gpu_layers) {
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    struct llama_model * m = llama_model_load_from_file(path, mp);
    return m;
}

void markos_llama_model_free(void * m) {
    llama_model_free((struct llama_model *) m);
}

int markos_llama_model_n_vocab(void * m) {
    const struct llama_vocab * v = llama_model_get_vocab((const struct llama_model *) m);
    return llama_vocab_n_tokens(v);
}

int markos_llama_vocab_eot(void * m) {
    const struct llama_vocab * v = llama_model_get_vocab((const struct llama_model *) m);
    return llama_vocab_eot(v);
}

int markos_llama_vocab_bos(void * m) {
    const struct llama_vocab * v = llama_model_get_vocab((const struct llama_model *) m);
    return llama_vocab_bos(v);
}

// returns token count; negative on error. add_bos: 1 = always prepend BOS.
int markos_llama_tokenize(void * m, const char * text, int add_bos,
                          int * out, int cap) {
    const struct llama_vocab * v = llama_model_get_vocab((const struct llama_model *) m);
    const int n = llama_tokenize(v, text, (int32_t) strlen(text), out,
                                 (int32_t) cap, add_bos != 0, false);
    return n;
}

// ---- context ----

// kv_type: 0 = f16, 1 = q8_0, 2 = q4_0. flash_attn: 1 = enabled.
void * markos_llama_context_create(void * m, unsigned n_ctx, unsigned n_batch,
                                   int threads, int kv_type, int flash_attn) {
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_batch;
    cp.n_ubatch = n_batch;
    cp.n_seq_max = 1;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    switch (kv_type) {
        case 1: cp.type_k = GGML_TYPE_Q8_0; cp.type_v = GGML_TYPE_Q8_0; break;
        case 2: cp.type_k = GGML_TYPE_Q4_0; cp.type_v = GGML_TYPE_Q4_0; break;
        default: break; // f16
    }
    cp.flash_attn_type = flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                    : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    return llama_init_from_model((struct llama_model *) m, cp);
}

void markos_llama_context_free(void * c) {
    llama_free((struct llama_context *) c);
}

// Clear the KV/memory contents — reusable contexts need this between
// requests so the next generation starts from a clean attention state.
void markos_llama_kv_clear(void * c) {
    llama_memory_clear(llama_get_memory((struct llama_context *) c), true);
}

// ---- batch ----

// The batch struct is kept shim-side; Rust holds the opaque handle.
void * markos_llama_batch_create(int n_tokens) {
    struct llama_batch * b = malloc(sizeof(struct llama_batch));
    *b = llama_batch_init(n_tokens, 0, 1);
    return b;
}

void markos_llama_batch_free(void * b) {
    struct llama_batch * bb = (struct llama_batch *) b;
    llama_batch_free(*bb);
    free(bb);
}

void markos_llama_batch_clear(void * b) {
    struct llama_batch * bb = (struct llama_batch *) b;
    bb->n_tokens = 0;
}

int markos_llama_batch_add(void * b, int token, int pos, int logits) {
    struct llama_batch * bb = (struct llama_batch *) b;
    // this fork's llama.h has no llama_batch_add helper: fill the planar
    // arrays by hand (single sequence, matching llama_batch_init(n,0,1))
    const int32_t i = bb->n_tokens;
    bb->token[i] = (llama_token) token;
    bb->pos[i] = (llama_pos) pos;
    bb->n_seq_id[i] = 1;
    bb->seq_id[i][0] = 0;
    bb->logits[i] = logits ? 1 : 0;
    bb->n_tokens = i + 1;
    return 0;
}

int markos_llama_decode(void * c, void * b) {
    return llama_decode((struct llama_context *) c, *(struct llama_batch *) b);
}

// ---- sampler ----

// Mirror of markos-engine's SamplingParams (config.rs). Both sides are ours;
// the Rust #[repr(C)] declaration must match this layout exactly.
struct markos_sampler_cfg {
    int32_t  top_k;
    float    top_p;
    float    min_p;
    float    temperature;
    float    repeat_penalty;
    float    frequency_penalty;
    float    presence_penalty;
    int32_t  repeat_last_n;
    int32_t  mirostat;
    float    mirostat_tau;
    float    mirostat_eta;
    uint32_t seed;
};

void * markos_llama_sampler_build(const struct markos_sampler_cfg * cfg, int n_vocab) {
    struct llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    struct llama_sampler * ch = llama_sampler_chain_init(sp);
    uint32_t seed = cfg->seed == 0 ? UINT32_MAX : cfg->seed;
    if (cfg->mirostat == 1) {
        llama_sampler_chain_add(ch, llama_sampler_init_mirostat(
            (int32_t) n_vocab, seed, cfg->mirostat_tau, cfg->mirostat_eta, 1));
        return ch;
    }
    if (cfg->mirostat == 2) {
        llama_sampler_chain_add(ch, llama_sampler_init_mirostat_v2(
            seed, cfg->mirostat_tau, cfg->mirostat_eta));
        return ch;
    }
    llama_sampler_chain_add(ch, llama_sampler_init_penalties(
        (int32_t) n_vocab, cfg->repeat_last_n, cfg->repeat_penalty,
        cfg->frequency_penalty, cfg->presence_penalty));
    llama_sampler_chain_add(ch, llama_sampler_init_top_k(cfg->top_k));
    llama_sampler_chain_add(ch, llama_sampler_init_top_p(cfg->top_p, 1));
    llama_sampler_chain_add(ch, llama_sampler_init_min_p(cfg->min_p, 1));
    if (cfg->temperature <= 0.0f) {
        llama_sampler_chain_add(ch, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(ch, llama_sampler_init_temp(cfg->temperature));
        llama_sampler_chain_add(ch, llama_sampler_init_dist(seed));
    }
    return ch;
}

void markos_llama_sampler_free(void * s) {
    llama_sampler_free((struct llama_sampler *) s);
}

int markos_llama_sampler_sample(void * s, void * c, int idx) {
    return llama_sampler_sample((struct llama_sampler *) s, (struct llama_context *) c,
                                (int32_t) idx);
}

void markos_llama_sampler_accept(void * s, int token) {
    llama_sampler_accept((struct llama_sampler *) s, (llama_token) token);
}

// ---- detokenize ----

// writes at most cap-1 bytes + NUL; returns piece length (may exceed cap)
int markos_llama_token_to_piece(void * m, int token, char * buf, int cap) {
    const struct llama_vocab * v = llama_model_get_vocab((const struct llama_model *) m);
    int n = llama_token_to_piece(v, (llama_token) token, buf, (int32_t) cap, 0, true);
    if (n < 0) return n; // too small: |n| is the needed size
    if (n < cap) buf[n] = '\0';
    return n;
}

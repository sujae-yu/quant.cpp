/* Quick diagnostic: print tokens our engine produces for a prompt. */
#define QUANT_IMPLEMENTATION
#include "../../quant.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf \"text\"\n", argv[0]); return 1; }
    tq_model_t* m = tq_load_gguf(argv[1]);
    if (!m) { fprintf(stderr, "load failed\n"); return 1; }
    tq_tokenizer_t t = {0};
    tq_load_tokenizer_from_gguf(argv[1], &t);

    int toks[128];
    int n = tq_encode(&t, argv[2], toks, 128, 0);  /* no BOS */
    printf("input: %s\n", argv[2]);
    printf("tokens (%d):", n);
    for (int i = 0; i < n; i++) printf(" %d", toks[i]);
    printf("\n");
    /* Decode each token back to show content */
    for (int i = 0; i < n; i++) {
        char buf[64];
        int len = tq_decode_token(&t, toks[i], buf, sizeof(buf));
        (void)len;
        printf("  [%d] = %d -> %s\n", i, toks[i], buf);
    }
    return 0;
}

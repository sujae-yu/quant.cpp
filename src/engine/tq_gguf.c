/**
 * tq_gguf.c — GGUF v3 format parser for TurboQuant
 *
 * Implements mmap-based zero-copy loading of GGUF files.
 * Supports GGUF versions 2 and 3 with all GGML quant types.
 *
 * SPDX-License-Identifier: MIT
 */

#include "turboquant/tq_gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

/* ============================================================
 * Helpers
 * ============================================================ */

static inline uint64_t align_up(uint64_t x, uint64_t align) {
    return (x + align - 1) & ~(align - 1);
}

/* Simple cursor-based reader over the mmap'd buffer */
typedef struct {
    const uint8_t* base;
    size_t         size;
    size_t         pos;
} tq_reader_t;

static bool reader_has(const tq_reader_t* r, size_t n) {
    return r->pos + n <= r->size;
}

static uint8_t read_u8(tq_reader_t* r) {
    uint8_t v = 0;
    if (reader_has(r, 1)) {
        v = r->base[r->pos];
        r->pos += 1;
    }
    return v;
}

static uint16_t read_u16(tq_reader_t* r) {
    uint16_t v = 0;
    if (reader_has(r, 2)) {
        memcpy(&v, r->base + r->pos, 2);
        r->pos += 2;
    }
    return v;
}

static uint32_t read_u32(tq_reader_t* r) {
    uint32_t v = 0;
    if (reader_has(r, 4)) {
        memcpy(&v, r->base + r->pos, 4);
        r->pos += 4;
    }
    return v;
}

static uint64_t read_u64(tq_reader_t* r) {
    uint64_t v = 0;
    if (reader_has(r, 8)) {
        memcpy(&v, r->base + r->pos, 8);
        r->pos += 8;
    }
    return v;
}

static float read_f32(tq_reader_t* r) {
    float v = 0.0f;
    if (reader_has(r, 4)) {
        memcpy(&v, r->base + r->pos, 4);
        r->pos += 4;
    }
    return v;
}

static double read_f64(tq_reader_t* r) {
    double v = 0.0;
    if (reader_has(r, 8)) {
        memcpy(&v, r->base + r->pos, 8);
        r->pos += 8;
    }
    return v;
}

/* Read a GGUF string: uint64 len + len bytes. Caller must free the returned str. */
static char* read_string(tq_reader_t* r, uint64_t* out_len) {
    uint64_t len = read_u64(r);
    if (out_len) *out_len = len;
    if (!reader_has(r, len)) {
        return NULL;
    }
    char* s = (char*)malloc(len + 1);
    if (!s) return NULL;
    memcpy(s, r->base + r->pos, len);
    s[len] = '\0';
    r->pos += len;
    return s;
}

/* Return the byte size of a single GGUF metadata scalar type */
static size_t gguf_type_scalar_size(tq_gguf_type type) {
    switch (type) {
        case TQ_GGUF_TYPE_UINT8:   return 1;
        case TQ_GGUF_TYPE_INT8:    return 1;
        case TQ_GGUF_TYPE_BOOL:    return 1;
        case TQ_GGUF_TYPE_UINT16:  return 2;
        case TQ_GGUF_TYPE_INT16:   return 2;
        case TQ_GGUF_TYPE_UINT32:  return 4;
        case TQ_GGUF_TYPE_INT32:   return 4;
        case TQ_GGUF_TYPE_FLOAT32: return 4;
        case TQ_GGUF_TYPE_UINT64:  return 8;
        case TQ_GGUF_TYPE_INT64:   return 8;
        case TQ_GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

/* ============================================================
 * GGML type tables
 * ============================================================ */

size_t tq_ggml_type_size(tq_ggml_dtype type) {
    switch (type) {
        case TQ_GGML_TYPE_F32:      return 4;
        case TQ_GGML_TYPE_F16:      return 2;
        case TQ_GGML_TYPE_BF16:     return 2;
        case TQ_GGML_TYPE_Q4_0:     return 18;
        case TQ_GGML_TYPE_Q4_1:     return 20;
        case TQ_GGML_TYPE_Q5_0:     return 22;
        case TQ_GGML_TYPE_Q5_1:     return 24;
        case TQ_GGML_TYPE_Q8_0:     return 34;
        case TQ_GGML_TYPE_Q8_1:     return 36;
        case TQ_GGML_TYPE_Q2_K:     return 84;
        case TQ_GGML_TYPE_Q3_K:     return 110;
        case TQ_GGML_TYPE_Q4_K:     return 144;
        case TQ_GGML_TYPE_Q5_K:     return 176;
        case TQ_GGML_TYPE_Q6_K:     return 210;
        case TQ_GGML_TYPE_Q8_K:     return 292;
        case TQ_GGML_TYPE_IQ2_XXS:  return 66;
        case TQ_GGML_TYPE_IQ2_XS:   return 74;
        case TQ_GGML_TYPE_IQ2_S:    return 82;
        case TQ_GGML_TYPE_IQ3_XXS:  return 98;
        case TQ_GGML_TYPE_IQ3_S:    return 110;
        case TQ_GGML_TYPE_IQ4_NL:   return 18;
        case TQ_GGML_TYPE_IQ4_XS:   return 136;
        case TQ_GGML_TYPE_IQ1_S:    return 50;
        default:                     return 0;
    }
}

int tq_ggml_type_blck(tq_ggml_dtype type) {
    switch (type) {
        case TQ_GGML_TYPE_F32:      return 1;
        case TQ_GGML_TYPE_F16:      return 1;
        case TQ_GGML_TYPE_BF16:     return 1;
        case TQ_GGML_TYPE_Q4_0:     return 32;
        case TQ_GGML_TYPE_Q4_1:     return 32;
        case TQ_GGML_TYPE_Q5_0:     return 32;
        case TQ_GGML_TYPE_Q5_1:     return 32;
        case TQ_GGML_TYPE_Q8_0:     return 32;
        case TQ_GGML_TYPE_Q8_1:     return 32;
        case TQ_GGML_TYPE_Q2_K:     return 256;
        case TQ_GGML_TYPE_Q3_K:     return 256;
        case TQ_GGML_TYPE_Q4_K:     return 256;
        case TQ_GGML_TYPE_Q5_K:     return 256;
        case TQ_GGML_TYPE_Q6_K:     return 256;
        case TQ_GGML_TYPE_Q8_K:     return 256;
        case TQ_GGML_TYPE_IQ2_XXS:  return 256;
        case TQ_GGML_TYPE_IQ2_XS:   return 256;
        case TQ_GGML_TYPE_IQ2_S:    return 256;
        case TQ_GGML_TYPE_IQ3_XXS:  return 256;
        case TQ_GGML_TYPE_IQ3_S:    return 256;
        case TQ_GGML_TYPE_IQ4_NL:   return 32;
        case TQ_GGML_TYPE_IQ4_XS:   return 256;
        case TQ_GGML_TYPE_IQ1_S:    return 256;
        default:                     return 0;
    }
}

const char* tq_ggml_type_name(tq_ggml_dtype type) {
    switch (type) {
        case TQ_GGML_TYPE_F32:      return "F32";
        case TQ_GGML_TYPE_F16:      return "F16";
        case TQ_GGML_TYPE_BF16:     return "BF16";
        case TQ_GGML_TYPE_Q4_0:     return "Q4_0";
        case TQ_GGML_TYPE_Q4_1:     return "Q4_1";
        case TQ_GGML_TYPE_Q5_0:     return "Q5_0";
        case TQ_GGML_TYPE_Q5_1:     return "Q5_1";
        case TQ_GGML_TYPE_Q8_0:     return "Q8_0";
        case TQ_GGML_TYPE_Q8_1:     return "Q8_1";
        case TQ_GGML_TYPE_Q2_K:     return "Q2_K";
        case TQ_GGML_TYPE_Q3_K:     return "Q3_K";
        case TQ_GGML_TYPE_Q4_K:     return "Q4_K";
        case TQ_GGML_TYPE_Q5_K:     return "Q5_K";
        case TQ_GGML_TYPE_Q6_K:     return "Q6_K";
        case TQ_GGML_TYPE_Q8_K:     return "Q8_K";
        case TQ_GGML_TYPE_IQ2_XXS:  return "IQ2_XXS";
        case TQ_GGML_TYPE_IQ2_XS:   return "IQ2_XS";
        case TQ_GGML_TYPE_IQ2_S:    return "IQ2_S";
        case TQ_GGML_TYPE_IQ3_XXS:  return "IQ3_XXS";
        case TQ_GGML_TYPE_IQ3_S:    return "IQ3_S";
        case TQ_GGML_TYPE_IQ4_NL:   return "IQ4_NL";
        case TQ_GGML_TYPE_IQ4_XS:   return "IQ4_XS";
        case TQ_GGML_TYPE_IQ1_S:    return "IQ1_S";
        default:                     return "unknown";
    }
}

/* ============================================================
 * Metadata value parsing
 * ============================================================ */

/* Parse a single metadata value from the reader into a kv struct.
 * Returns false on failure. */
static bool parse_kv_value(tq_reader_t* r, tq_gguf_kv_t* kv, tq_gguf_type type) {
    kv->type = type;

    switch (type) {
        case TQ_GGUF_TYPE_UINT8:
            kv->value.u8 = read_u8(r);
            break;
        case TQ_GGUF_TYPE_INT8:
            kv->value.i8 = (int8_t)read_u8(r);
            break;
        case TQ_GGUF_TYPE_BOOL:
            kv->value.bool_val = read_u8(r);
            break;
        case TQ_GGUF_TYPE_UINT16:
            kv->value.u16 = read_u16(r);
            break;
        case TQ_GGUF_TYPE_INT16:
            kv->value.i16 = (int16_t)read_u16(r);
            break;
        case TQ_GGUF_TYPE_UINT32:
            kv->value.u32 = read_u32(r);
            break;
        case TQ_GGUF_TYPE_INT32:
            kv->value.i32 = (int32_t)read_u32(r);
            break;
        case TQ_GGUF_TYPE_FLOAT32:
            kv->value.f32 = read_f32(r);
            break;
        case TQ_GGUF_TYPE_UINT64:
            kv->value.u64 = read_u64(r);
            break;
        case TQ_GGUF_TYPE_INT64:
            kv->value.i64 = (int64_t)read_u64(r);
            break;
        case TQ_GGUF_TYPE_FLOAT64:
            kv->value.f64 = read_f64(r);
            break;
        case TQ_GGUF_TYPE_STRING: {
            uint64_t slen = 0;
            char* s = read_string(r, &slen);
            if (!s) return false;
            kv->value.string.len = slen;
            kv->value.string.str = s;
            break;
        }
        case TQ_GGUF_TYPE_ARRAY: {
            uint32_t elem_type = read_u32(r);
            uint64_t count     = read_u64(r);

            kv->value.array.elem_type = (tq_gguf_type)elem_type;
            kv->value.array.count     = count;
            kv->value.array.data      = NULL;

            if (count == 0) break;

            if ((tq_gguf_type)elem_type == TQ_GGUF_TYPE_STRING) {
                /* Array of strings: allocate array of tq_gguf_string_t */
                tq_gguf_string_t* strs = (tq_gguf_string_t*)calloc(count, sizeof(tq_gguf_string_t));
                if (!strs) return false;
                for (uint64_t i = 0; i < count; i++) {
                    uint64_t slen = 0;
                    char* s = read_string(r, &slen);
                    if (!s) {
                        /* Clean up already-parsed strings */
                        for (uint64_t j = 0; j < i; j++) {
                            free(strs[j].str);
                        }
                        free(strs);
                        return false;
                    }
                    strs[i].len = slen;
                    strs[i].str = s;
                }
                kv->value.array.data = strs;
            } else if ((tq_gguf_type)elem_type == TQ_GGUF_TYPE_ARRAY) {
                /* Nested arrays not supported, skip */
                fprintf(stderr, "tq_gguf: nested arrays not supported\n");
                return false;
            } else {
                /* Scalar array: bulk copy */
                size_t elem_sz = gguf_type_scalar_size((tq_gguf_type)elem_type);
                if (elem_sz == 0) return false;
                size_t total = (size_t)count * elem_sz;
                if (!reader_has(r, total)) return false;
                void* buf = malloc(total);
                if (!buf) return false;
                memcpy(buf, r->base + r->pos, total);
                r->pos += total;
                kv->value.array.data = buf;
            }
            break;
        }
        default:
            fprintf(stderr, "tq_gguf: unknown metadata type %d\n", (int)type);
            return false;
    }
    return true;
}

/* Free heap memory inside a kv entry */
static void free_kv(tq_gguf_kv_t* kv) {
    if (kv->type == TQ_GGUF_TYPE_STRING) {
        free(kv->value.string.str);
        kv->value.string.str = NULL;
    } else if (kv->type == TQ_GGUF_TYPE_ARRAY) {
        if (kv->value.array.elem_type == TQ_GGUF_TYPE_STRING && kv->value.array.data) {
            tq_gguf_string_t* strs = (tq_gguf_string_t*)kv->value.array.data;
            for (uint64_t i = 0; i < kv->value.array.count; i++) {
                free(strs[i].str);
            }
        }
        free(kv->value.array.data);
        kv->value.array.data = NULL;
    }
}

/* ============================================================
 * Platform-specific mmap
 * ============================================================ */

#ifdef _WIN32

static void* platform_mmap(const char* path, size_t* out_size) {
    HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "tq_gguf: cannot open '%s'\n", path);
        return NULL;
    }

    LARGE_INTEGER fsize;
    if (!GetFileSizeEx(file, &fsize)) {
        CloseHandle(file);
        fprintf(stderr, "tq_gguf: cannot get size of '%s'\n", path);
        return NULL;
    }
    *out_size = (size_t)fsize.QuadPart;

    HANDLE mapping = CreateFileMappingA(file, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mapping) {
        CloseHandle(file);
        fprintf(stderr, "tq_gguf: CreateFileMapping failed for '%s'\n", path);
        return NULL;
    }

    void* data = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(mapping);
    CloseHandle(file);

    if (!data) {
        fprintf(stderr, "tq_gguf: MapViewOfFile failed for '%s'\n", path);
        return NULL;
    }
    return data;
}

static void platform_munmap(void* addr, size_t size) {
    (void)size;
    if (addr) UnmapViewOfFile(addr);
}

#else /* POSIX */

static void* platform_mmap(const char* path, size_t* out_size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "tq_gguf: cannot open '%s'\n", path);
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        fprintf(stderr, "tq_gguf: cannot stat '%s'\n", path);
        return NULL;
    }
    *out_size = (size_t)st.st_size;

    void* data = mmap(NULL, *out_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (data == MAP_FAILED) {
        fprintf(stderr, "tq_gguf: mmap failed for '%s'\n", path);
        return NULL;
    }
    return data;
}

static void platform_munmap(void* addr, size_t size) {
    if (addr) munmap(addr, size);
}

#endif

/* ============================================================
 * GGUF open / close
 * ============================================================ */

tq_gguf_ctx_t* tq_gguf_open(const char* path) {
    if (!path) {
        fprintf(stderr, "tq_gguf: NULL path\n");
        return NULL;
    }

    /* mmap the file */
    size_t file_size = 0;
    void* mmap_data = platform_mmap(path, &file_size);
    if (!mmap_data) return NULL;

    /* Minimum header: magic(4) + version(4) + n_tensors(8) + n_kv(8) = 24 */
    if (file_size < 24) {
        fprintf(stderr, "tq_gguf: file too small (%zu bytes)\n", file_size);
        platform_munmap(mmap_data, file_size);
        return NULL;
    }

    tq_reader_t reader = { .base = (const uint8_t*)mmap_data, .size = file_size, .pos = 0 };

    /* Parse header */
    uint32_t magic = read_u32(&reader);
    if (magic != TQ_GGUF_MAGIC) {
        fprintf(stderr, "tq_gguf: bad magic 0x%08x (expected 0x%08x)\n",
                magic, TQ_GGUF_MAGIC);
        platform_munmap(mmap_data, file_size);
        return NULL;
    }

    uint32_t version = read_u32(&reader);
    if (version < TQ_GGUF_VERSION_MIN || version > TQ_GGUF_VERSION_MAX) {
        fprintf(stderr, "tq_gguf: unsupported version %u (need %u-%u)\n",
                version, TQ_GGUF_VERSION_MIN, TQ_GGUF_VERSION_MAX);
        platform_munmap(mmap_data, file_size);
        return NULL;
    }

    uint64_t n_tensors = read_u64(&reader);
    uint64_t n_kv      = read_u64(&reader);

    /* Allocate context */
    tq_gguf_ctx_t* ctx = (tq_gguf_ctx_t*)calloc(1, sizeof(tq_gguf_ctx_t));
    if (!ctx) {
        fprintf(stderr, "tq_gguf: out of memory\n");
        platform_munmap(mmap_data, file_size);
        return NULL;
    }

    ctx->version    = version;
    ctx->n_tensors  = n_tensors;
    ctx->n_kv       = n_kv;
    ctx->alignment  = TQ_GGUF_DEFAULT_ALIGNMENT;
    ctx->mmap_data  = mmap_data;
    ctx->mmap_size  = file_size;
    ctx->arch[0]    = '\0';

    /* Allocate metadata KV array */
    if (n_kv > 0) {
        ctx->kv = (tq_gguf_kv_t*)calloc((size_t)n_kv, sizeof(tq_gguf_kv_t));
        if (!ctx->kv) {
            fprintf(stderr, "tq_gguf: out of memory (kv)\n");
            tq_gguf_close(ctx);
            return NULL;
        }
    }

    /* Parse metadata KV pairs */
    for (uint64_t i = 0; i < n_kv; i++) {
        tq_gguf_kv_t* kv = &ctx->kv[i];

        /* Read key */
        uint64_t key_len = 0;
        char* key = read_string(&reader, &key_len);
        if (!key) {
            fprintf(stderr, "tq_gguf: failed to read key %llu\n", (unsigned long long)i);
            tq_gguf_close(ctx);
            return NULL;
        }
        size_t copy_len = key_len < TQ_GGUF_MAX_NAME - 1 ? key_len : TQ_GGUF_MAX_NAME - 1;
        memcpy(kv->key, key, copy_len);
        kv->key[copy_len] = '\0';
        free(key);

        /* Read value type and value */
        uint32_t vtype = read_u32(&reader);
        if (!parse_kv_value(&reader, kv, (tq_gguf_type)vtype)) {
            fprintf(stderr, "tq_gguf: failed to parse value for key '%s'\n", kv->key);
            tq_gguf_close(ctx);
            return NULL;
        }

        /* Check for alignment override */
        if (strcmp(kv->key, "general.alignment") == 0 && kv->type == TQ_GGUF_TYPE_UINT32) {
            ctx->alignment = kv->value.u32;
        }

        /* Extract architecture string */
        if (strcmp(kv->key, "general.architecture") == 0 && kv->type == TQ_GGUF_TYPE_STRING) {
            size_t alen = kv->value.string.len;
            if (alen > sizeof(ctx->arch) - 1) alen = sizeof(ctx->arch) - 1;
            memcpy(ctx->arch, kv->value.string.str, alen);
            ctx->arch[alen] = '\0';
        }
    }

    /* Allocate tensor descriptors */
    if (n_tensors > 0) {
        ctx->tensors = (tq_gguf_tensor_t*)calloc((size_t)n_tensors, sizeof(tq_gguf_tensor_t));
        if (!ctx->tensors) {
            fprintf(stderr, "tq_gguf: out of memory (tensors)\n");
            tq_gguf_close(ctx);
            return NULL;
        }
    }

    /* Parse tensor descriptors */
    for (uint64_t i = 0; i < n_tensors; i++) {
        tq_gguf_tensor_t* t = &ctx->tensors[i];

        /* Read tensor name */
        uint64_t name_len = 0;
        char* name = read_string(&reader, &name_len);
        if (!name) {
            fprintf(stderr, "tq_gguf: failed to read tensor name %llu\n", (unsigned long long)i);
            tq_gguf_close(ctx);
            return NULL;
        }
        size_t copy_len = name_len < TQ_GGUF_MAX_NAME - 1 ? name_len : TQ_GGUF_MAX_NAME - 1;
        memcpy(t->name, name, copy_len);
        t->name[copy_len] = '\0';
        free(name);

        /* Number of dimensions */
        t->n_dims = read_u32(&reader);
        if (t->n_dims > 4) {
            fprintf(stderr, "tq_gguf: tensor '%s' has %u dims (max 4)\n", t->name, t->n_dims);
            tq_gguf_close(ctx);
            return NULL;
        }

        /* Shape */
        for (uint32_t d = 0; d < t->n_dims; d++) {
            t->shape[d] = (int64_t)read_u64(&reader);
        }
        for (uint32_t d = t->n_dims; d < 4; d++) {
            t->shape[d] = 1;
        }

        /* Type */
        t->type = (tq_ggml_dtype)read_u32(&reader);

        /* Offset within tensor data section */
        t->offset = read_u64(&reader);

        /* Compute total number of elements */
        int64_t n_elements = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) {
            n_elements *= t->shape[d];
        }

        /* Compute size in bytes */
        int blck = tq_ggml_type_blck(t->type);
        size_t type_sz = tq_ggml_type_size(t->type);
        if (blck > 0 && type_sz > 0) {
            t->size_bytes = (size_t)((n_elements + blck - 1) / blck) * type_sz;
        } else {
            t->size_bytes = 0;
        }
    }

    /* Compute data section offset: aligned up from current reader position */
    ctx->data_offset = (size_t)align_up((uint64_t)reader.pos, (uint64_t)ctx->alignment);

    /* Set data pointers for each tensor */
    for (uint64_t i = 0; i < n_tensors; i++) {
        tq_gguf_tensor_t* t = &ctx->tensors[i];
        size_t abs_offset = ctx->data_offset + (size_t)t->offset;
        if (abs_offset + t->size_bytes > file_size) {
            fprintf(stderr, "tq_gguf: tensor '%s' data out of bounds "
                    "(offset %zu + size %zu > file %zu)\n",
                    t->name, abs_offset, t->size_bytes, file_size);
            tq_gguf_close(ctx);
            return NULL;
        }
        t->data = (const uint8_t*)mmap_data + abs_offset;
    }

    return ctx;
}

void tq_gguf_close(tq_gguf_ctx_t* ctx) {
    if (!ctx) return;

    /* Free metadata heap allocations */
    if (ctx->kv) {
        for (uint64_t i = 0; i < ctx->n_kv; i++) {
            free_kv(&ctx->kv[i]);
        }
        free(ctx->kv);
    }

    /* Free tensor descriptors (no heap inside, just the array) */
    free(ctx->tensors);

    /* Unmap the file */
    platform_munmap(ctx->mmap_data, ctx->mmap_size);

    free(ctx);
}

/* ============================================================
 * Metadata lookup
 * ============================================================ */

int64_t tq_gguf_find_key(const tq_gguf_ctx_t* ctx, const char* key) {
    if (!ctx || !key) return -1;
    for (uint64_t i = 0; i < ctx->n_kv; i++) {
        if (strcmp(ctx->kv[i].key, key) == 0) {
            return (int64_t)i;
        }
    }
    return -1;
}

int32_t tq_gguf_get_i32(const tq_gguf_ctx_t* ctx, const char* key, int32_t fallback) {
    int64_t idx = tq_gguf_find_key(ctx, key);
    if (idx < 0) return fallback;
    const tq_gguf_kv_t* kv = &ctx->kv[idx];
    switch (kv->type) {
        case TQ_GGUF_TYPE_INT32:   return kv->value.i32;
        case TQ_GGUF_TYPE_UINT32:  return (int32_t)kv->value.u32;
        case TQ_GGUF_TYPE_INT16:   return (int32_t)kv->value.i16;
        case TQ_GGUF_TYPE_UINT16:  return (int32_t)kv->value.u16;
        case TQ_GGUF_TYPE_INT8:    return (int32_t)kv->value.i8;
        case TQ_GGUF_TYPE_UINT8:   return (int32_t)kv->value.u8;
        default:                   return fallback;
    }
}

uint32_t tq_gguf_get_u32(const tq_gguf_ctx_t* ctx, const char* key, uint32_t fallback) {
    int64_t idx = tq_gguf_find_key(ctx, key);
    if (idx < 0) return fallback;
    const tq_gguf_kv_t* kv = &ctx->kv[idx];
    switch (kv->type) {
        case TQ_GGUF_TYPE_UINT32:  return kv->value.u32;
        case TQ_GGUF_TYPE_INT32:   return (uint32_t)kv->value.i32;
        case TQ_GGUF_TYPE_UINT16:  return (uint32_t)kv->value.u16;
        case TQ_GGUF_TYPE_INT16:   return (uint32_t)kv->value.i16;
        case TQ_GGUF_TYPE_UINT8:   return (uint32_t)kv->value.u8;
        case TQ_GGUF_TYPE_INT8:    return (uint32_t)kv->value.i8;
        default:                   return fallback;
    }
}

float tq_gguf_get_f32(const tq_gguf_ctx_t* ctx, const char* key, float fallback) {
    int64_t idx = tq_gguf_find_key(ctx, key);
    if (idx < 0) return fallback;
    const tq_gguf_kv_t* kv = &ctx->kv[idx];
    switch (kv->type) {
        case TQ_GGUF_TYPE_FLOAT32: return kv->value.f32;
        case TQ_GGUF_TYPE_FLOAT64: return (float)kv->value.f64;
        default:                   return fallback;
    }
}

const char* tq_gguf_get_str(const tq_gguf_ctx_t* ctx, const char* key) {
    int64_t idx = tq_gguf_find_key(ctx, key);
    if (idx < 0) return NULL;
    const tq_gguf_kv_t* kv = &ctx->kv[idx];
    if (kv->type != TQ_GGUF_TYPE_STRING) return NULL;
    return kv->value.string.str;
}

/* ============================================================
 * Tensor lookup
 * ============================================================ */

const tq_gguf_tensor_t* tq_gguf_find_tensor(const tq_gguf_ctx_t* ctx, const char* name) {
    if (!ctx || !name) return NULL;
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        if (strcmp(ctx->tensors[i].name, name) == 0) {
            return &ctx->tensors[i];
        }
    }
    return NULL;
}

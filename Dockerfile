# quant.cpp — Multi-stage Docker build
# Final image: Alpine + static binary (~10MB)

# ---- Build stage ----
FROM alpine:3.20 AS builder

RUN apk add --no-cache cmake gcc g++ musl-dev make linux-headers

WORKDIR /src
COPY . .

RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-static" \
        -DCMAKE_EXE_LINKER_FLAGS="-static" \
        -DTQ_BUILD_TESTS=OFF \
        -DTQ_BUILD_BENCH=OFF \
        -DTQ_BUILD_SERVER=ON \
    && cmake --build build -j$(nproc) --target quant \
    && cmake --build build -j$(nproc) --target quant-server

# ---- Runtime stage ----
FROM alpine:3.20

# Labels
LABEL org.opencontainers.image.title="quant.cpp" \
      org.opencontainers.image.description="LLM inference with 7x longer context — pure C, zero dependencies" \
      org.opencontainers.image.source="https://github.com/quantumaikr/quant.cpp"

# Copy binaries
COPY --from=builder /src/build/quant /usr/local/bin/quant
COPY --from=builder /src/build/quant-server /usr/local/bin/quant-server

# Create model mount point
RUN mkdir -p /models

# OpenAI-compatible server port
EXPOSE 8080

# Volume for GGUF model files
VOLUME ["/models"]

ENTRYPOINT ["quant"]

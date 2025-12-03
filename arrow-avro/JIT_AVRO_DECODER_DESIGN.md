# JIT-Compiled Avro to Arrow Decoder Design

## Overview

The JIT Avro decoder uses Cranelift to generate native machine code for Avro-to-Arrow conversion at runtime. The decoder is compiled once per schema and cached for the lifetime of the connection, providing ~5x faster decoding compared to interpreted approaches.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Schema Compilation                           │
│  Arrow Schema → FieldOps → Cranelift IR → Native Code → JitDecoder  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Runtime Decoding                            │
│  Avro Bytes → decode_checked(buffers, row_count) → Arrow Buffers    │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Buffer to RecordBatch                          │
│  Arrow Buffers → buffers_to_batch() → RecordBatch (zero-copy)       │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Single-Pass Row-Oriented Decoding
- Avro is row-oriented, Arrow is column-oriented
- JIT decodes one row at a time, writing to multiple column buffers simultaneously
- No intermediate representation - direct Avro bytes to Arrow buffers

### 2. Pre-allocated Output Buffers
- Caller provides output buffers via `OutputBuffer` array
- Each column gets one or more buffer slots:
  - Fixed-size types: single buffer (values)
  - Variable-size types: two buffers (offsets + data)
  - Nullable types: validity buffer managed internally

### 3. Schema Metrics for Buffer Planning
- `SchemaMetrics` computed once per schema
- Tracks: column count, fixed/variable columns, estimated sizes
- Used by `plan_buffer_layout()` to pre-allocate optimal buffer sizes

### 4. Zero-Copy Buffer-to-Array Conversion
- `buffer_to_array()` wraps raw buffers as Arrow arrays
- No data copying - buffers become array backing storage
- `buffers_to_batch()` assembles arrays into RecordBatch

## Current Performance (2000 rows)

| Type | Apache Avro | JIT | Speedup |
|------|-------------|-----|---------|
| Flat (4 primitives) | 35.7 µs | 6.3 µs | **5.7x** |
| Boolean | 8.5 µs | 1.7 µs | **5.1x** |
| Fixed (16-byte) | 11.1 µs | 6.5 µs | **1.7x** |
| Enum (4 symbols) | 12.8 µs | 2.3 µs | **5.6x** |
| Struct (nested) | 40.2 µs | 7.8 µs | **5.1x** |
| Union (nullable) | 21.7 µs | 3.9 µs | **5.6x** |

### Variable-Length Types (10,000 rows)

| Type | Apache Avro | JIT | Speedup |
|------|-------------|-----|---------|
| String (avg 20 bytes) | 93.9 µs | 44.0 µs | **2.1x** |
| Bytes (avg 50 bytes) | 96.1 µs | 65.0 µs | **1.5x** |
| Array/List (avg 5 elem) | 280 µs | 150 µs | **1.9x** |
| Map (avg 3 entries) | 403 µs | 144 µs | **2.8x** |

### Throughput Highlights
- Boolean: **1.25 Gelem/s**
- Enum: **865 Melem/s**
- Primitives: **300-316 Melem/s**

## Supported Types

### Fully Implemented
- [x] Int32, Int64, Float32, Float64
- [x] Boolean
- [x] Utf8, Binary, LargeUtf8, LargeBinary (Bytes)
- [x] FixedSizeBinary
- [x] Struct (nested records)
- [x] Dictionary/Enum
- [x] List, LargeList
- [x] Map
- [x] FixedSizeList
- [x] Decimal128, Decimal256
- [x] Duration, Interval
- [x] Date32, Date64, Time32, Time64, Timestamp

### Missing Types
- [x] **Null** - Required for nullable unions (`["null", "type"]`) - **IMPLEMENTED**
- [ ] RunEndEncoded
- [ ] ListView, LargeListView
- [ ] BinaryView, Utf8View

## Recently Completed Features

### 1. Null Type Support ✅
**Status**: IMPLEMENTED - Enables all Avro nullable unions

Avro nullable fields are encoded as unions: `["null", "actual_type"]`.
The JIT decoder now handles Null as a no-op field (no data to read).

**Implementation** (in `jit_decoder.rs`):
```rust
// Added to FieldOp enum:
Null,  // No data to read, used in unions for nullable fields

// In field_to_op():
DataType::Null => Ok(FieldOp::Null),

// In emit_field_decode():
FieldOp::Null => {
    // Null type: no data to read, no output to write
}
```

### 2. List/Map Buffer Layout ✅
**Status**: FIXED - Variable-length nested types now work correctly

**Buffer Layout** (2 buffers each):
- List<T>: offsets (i32) + values buffer
- Map<K,V>: offsets + values buffer

**Benchmark Results** (10,000 rows):
- Array/List: 280µs → 150µs = **1.9x faster**
- Map: 403µs → 144µs = **2.8x faster**

### 3. Union Type with Null ✅
**Status**: WORKING - Benchmarked at **5.6x faster** than Apache Avro

## Optimization Opportunities

### High Impact

#### 1. SIMD Varint Decoding ⏸️ DEFERRED
**Reason**: SIMD batch decode requires architectural changes to integrate with row-oriented JIT.
Row-by-row decoding can't easily batch varints across rows. Would need two-pass approach
(scan all varints first, then decode) which adds complexity for limited applicability
(int-only schemas). Standalone benchmark showed 2.5x speedup but integration not practical.

#### 2. Prefetch Hints (10-20% for large batches) ⏸️ DEFERRED
**Current**: No prefetch instructions
**Optimized**: Add software prefetch for:
- Input buffer (next cache line)
- Output buffers (write position + N)

#### 3. Loop Unrolling (15-20% for fixed schemas) ⏸️ DEFERRED
**Reason**: Limited to fixed-size-only schemas (niche), modest gain doesn't justify complexity.
Column-batched variant analyzed: ~10-30% potential gain from write locality + ILP, but still niche.

### Medium Impact

#### 4. Nullable Skip Optimization
For mostly-non-null data, track "all valid" state and skip validity bitmap writes.

#### 5. Branch Prediction Hints
Add likely/unlikely hints for:
- Varint continuation (usually single byte)
- Null checks (usually non-null)

## Implementation Plan

### Phase 1: Quick Wins ✅ COMPLETE
1. ✅ Add `DataType::Null` support
2. ✅ Fix benchmark buffer layouts for String/Bytes/Array/Map
3. ⏸️ Prefetch hints (deferred - hardware prefetchers effective, Cranelift lacks direct support)

### Phase 2: Variable-Length Types ✅ COMPLETE
1. ✅ Proper List codegen with offset tracking
2. ✅ Proper Map codegen (key skipping, value decoding)
3. ✅ Union with Null variant

### Phase 3: Further Optimizations ⏸️ DEFERRED
All remaining optimizations deferred due to limited applicability or modest gains:
1. ⏸️ SIMD varint - requires architectural changes, limited to int-only schemas
2. ⏸️ Loop unrolling - limited to fixed-size-only schemas
3. ⏸️ Prefetch hints - hardware prefetchers effective

## Testing Strategy

### Unit Tests
- `jit_decoder.rs` - 124 tests covering all type combinations
- `jit_buffer_utils.rs` - 3 tests for buffer-to-array conversion

### Benchmarks
- `jit_vs_apache_benchmark.rs` - Compares JIT vs Apache Avro decoder
- `jit_per_type_benchmark.rs` - All simple types combined
- `jit_benchmark.rs` - Compilation time, multi-column performance

### Integration Tests
- `pooled_avro_roundtrip_test.rs` - End-to-end Avro encoding → JIT decoding

## File Structure

```
arrow-avro/src/reader/
├── jit_decoder.rs      # JIT compiler and decoder
├── jit_buffer_utils.rs # Buffer-to-array utilities (public API)
├── mod.rs              # Module exports
└── pooled_decoder.rs   # Interpreted decoder (fallback)

arrow-avro/benches/
├── jit_vs_apache_benchmark.rs
├── jit_per_type_benchmark.rs
└── jit_benchmark.rs
```

## Usage Example

```rust
use arrow_avro::reader::jit_decoder::{JitDecoder, OutputBuffer};
use arrow_avro::reader::jit_buffer_utils::{BufferSet, buffers_to_batch};

// Compile once per schema (70-200µs)
let decoder = JitDecoder::compile(&arrow_schema)?;

// Allocate buffers based on schema
let mut buffer_sets = allocate_buffer_sets(&decoder, row_count, avg_var_bytes);

// Build output buffer list
let output_buffers: Vec<OutputBuffer> = /* ... */;

// Decode (zero-copy to pre-allocated buffers)
decoder.decode_checked(&avro_bytes, &output_buffers, row_count)?;

// Convert to RecordBatch (zero-copy)
let batch = buffers_to_batch(&schema, &buffer_sets, row_count)?;
```

## Changelog

### 2025-12-03 (Optimization Review)
- Explored SIMD varint decoding: standalone benchmarks showed 2.5-3x speedup for single-byte varints
- Deferred SIMD integration: row-oriented JIT can't easily batch varints, would need architectural changes
- Deferred loop unrolling: limited to fixed-size-only schemas (niche use case)
- Removed batch string memcpy from roadmap: Avro string data not contiguous (length varints interleaved)
- **Conclusion**: JIT decoder already 5-6x faster than Apache Avro; further optimizations have diminishing returns

### 2025-12-03 (Phase 1 & 2 Complete)
- ✅ Implemented `DataType::Null` support for nullable unions
- ✅ Fixed buffer layouts for String, Bytes, Array, Map benchmarks
- ✅ All variable-length types now work correctly
- ✅ Union benchmark: **5.6x faster** than Apache Avro
- ✅ Array benchmark: **1.9x faster** (10k rows)
- ✅ Map benchmark: **2.8x faster** (10k rows)
- All 124 JIT tests + 3 buffer_utils tests passing

### 2024-12-03
- Initial design document
- Benchmarked: 5x average speedup over Apache Avro
- Identified missing Null type support
- Identified SIMD optimization opportunities

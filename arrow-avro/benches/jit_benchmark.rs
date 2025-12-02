//! Benchmark comparing JIT decoder vs interpreted decoder performance
//!
//! This benchmark measures:
//! - Interpreted (PooledRecordDecoder)
//! - JIT compiled (JitDecoder)
//!
//! Run with: cargo bench --features jit --bench jit_benchmark

use arrow_avro::buffer_provider::{BufferLayout, ColumnRegion, PooledBufferProvider};
use arrow_avro::reader::jit_decoder::JitDecoder;
use arrow_avro::reader::pooled_decoder::PooledRecordDecoder;
use arrow_schema::{DataType, Field, Schema};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::alloc::{alloc, Layout};
use std::hint::black_box;
use std::sync::Arc;

/// Allocate aligned buffer for Arrow (8-byte alignment for i64/f64)
fn alloc_aligned(size: usize) -> Vec<u8> {
    if size == 0 {
        return Vec::new();
    }
    let layout = Layout::from_size_align(size, 8).expect("valid layout");
    unsafe {
        let ptr = alloc(layout);
        if ptr.is_null() {
            panic!("allocation failed");
        }
        std::ptr::write_bytes(ptr, 0, size);
        Vec::from_raw_parts(ptr, size, size)
    }
}

/// Encode a varint (zig-zag + variable length)
fn encode_varint(buf: &mut Vec<u8>, value: i64) {
    let zigzag = ((value << 1) ^ (value >> 63)) as u64;
    let mut v = zigzag;
    loop {
        if v < 0x80 {
            buf.push(v as u8);
            break;
        }
        buf.push((v as u8) | 0x80);
        v >>= 7;
    }
}

/// Create layout for testing
fn create_test_layout(schema: &Schema, row_count: usize) -> BufferLayout {
    let mut columns = Vec::new();
    let mut offset = 0;

    for field in schema.fields() {
        let (data_cap, offsets_cap, null_cap) = estimate_column_size(field.data_type(), row_count);

        let region = ColumnRegion {
            data_offset: offset,
            data_capacity: data_cap,
            offsets_offset: if offsets_cap > 0 {
                Some(offset + data_cap)
            } else {
                None
            },
            offsets_capacity: if offsets_cap > 0 {
                Some(offsets_cap)
            } else {
                None
            },
            null_bitmap_offset: if field.is_nullable() {
                Some(offset + data_cap + offsets_cap)
            } else {
                None
            },
            null_bitmap_capacity: if field.is_nullable() {
                Some(null_cap)
            } else {
                None
            },
            child_regions: vec![],
        };

        offset += data_cap + offsets_cap + null_cap;
        columns.push(region);
    }

    BufferLayout {
        columns,
        total_size: offset.max(1024),
        row_count,
    }
}

fn estimate_column_size(dt: &DataType, rows: usize) -> (usize, usize, usize) {
    let data = match dt {
        DataType::Boolean => (rows + 7) / 8,
        DataType::Int32 | DataType::Float32 => rows * 4,
        DataType::Int64 | DataType::Float64 => rows * 8,
        DataType::Utf8 | DataType::Binary => rows * 128,
        DataType::List(_) | DataType::Map(_, _) => rows * 256,
        DataType::Struct(_) => rows * 128,
        _ => rows * 8,
    };
    let offsets = match dt {
        DataType::Utf8 | DataType::Binary | DataType::List(_) => (rows + 1) * 4,
        _ => 0,
    };
    let null_bitmap = (rows + 7) / 8;
    (data, offsets, null_bitmap)
}

/// Generate test data with small varints (0-127 encode to single byte)
fn generate_single_byte_int64_data(row_count: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(row_count);
    for i in 0..row_count {
        // Values 0-63 zigzag encode to 0-126, which are single-byte varints
        let value = (i % 64) as i64;
        encode_varint(&mut data, value);
    }
    data
}

fn bench_simple_int64_schema(c: &mut Criterion) {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));

    let row_counts = [100, 1000, 10000];

    let mut group = c.benchmark_group("jit_simple_int64");

    for &row_count in &row_counts {
        // Generate test data with single-byte varints (current JIT limitation)
        let test_data = generate_single_byte_int64_data(row_count);

        group.throughput(Throughput::Elements(row_count as u64));

        // Benchmark JIT decoder
        let jit_decoder = JitDecoder::compile(&schema).expect("JIT compile");

        group.bench_with_input(BenchmarkId::new("jit", row_count), &row_count, |b, _| {
            b.iter(|| {
                let mut output = vec![0u8; row_count * 8];
                let mut output_ptrs = vec![output.as_mut_ptr()];
                let result = jit_decoder
                    .decode(&test_data, &mut output_ptrs, row_count)
                    .expect("decode");
                black_box(result);
            });
        });

        // Benchmark interpreted decoder (PooledRecordDecoder)
        let layout = create_test_layout(&schema, row_count);

        group.bench_with_input(
            BenchmarkId::new("interpreted", row_count),
            &row_count,
            |b, _| {
                b.iter(|| {
                    let buffer = alloc_aligned(layout.total_size);
                    let provider = PooledBufferProvider::new(buffer, layout.clone());
                    let mut decoder =
                        PooledRecordDecoder::try_new(schema.clone(), provider).expect("decoder");
                    decoder.decode(&test_data, row_count).expect("decode");
                    let batch = decoder.flush().expect("flush");
                    black_box(batch);
                });
            },
        );
    }

    group.finish();
}

fn bench_multi_column_schema(c: &mut Criterion) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Int32, false),
        Field::new("score", DataType::Float64, false),
    ]));

    let row_count = 10000;

    // Generate multi-column test data with single-byte varints
    let mut test_data = Vec::new();
    for i in 0..row_count {
        // Int64 id (single-byte varint)
        let id = (i % 64) as i64;
        encode_varint(&mut test_data, id);

        // Int32 value (single-byte varint)
        let value = ((i * 2) % 64) as i64;
        encode_varint(&mut test_data, value);

        // Float64 score (8 bytes little-endian)
        let score = i as f64 * 1.5;
        test_data.extend_from_slice(&score.to_le_bytes());
    }

    let mut group = c.benchmark_group("jit_multi_column");
    group.throughput(Throughput::Elements(row_count as u64));

    // JIT decoder
    let jit_decoder = JitDecoder::compile(&schema).expect("JIT compile");

    group.bench_function("jit", |b| {
        b.iter(|| {
            let mut output0 = vec![0u8; row_count * 8]; // Int64
            let mut output1 = vec![0u8; row_count * 4]; // Int32
            let mut output2 = vec![0u8; row_count * 8]; // Float64
            let mut output_ptrs = vec![
                output0.as_mut_ptr(),
                output1.as_mut_ptr(),
                output2.as_mut_ptr(),
            ];
            let result = jit_decoder
                .decode(&test_data, &mut output_ptrs, row_count)
                .expect("decode");
            black_box(result);
        });
    });

    // Interpreted decoder
    let layout = create_test_layout(&schema, row_count);

    group.bench_function("interpreted", |b| {
        b.iter(|| {
            let buffer = alloc_aligned(layout.total_size);
            let provider = PooledBufferProvider::new(buffer, layout.clone());
            let mut decoder =
                PooledRecordDecoder::try_new(schema.clone(), provider).expect("decoder");
            decoder.decode(&test_data, row_count).expect("decode");
            let batch = decoder.flush().expect("flush");
            black_box(batch);
        });
    });

    group.finish();
}

fn bench_jit_compilation(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_compilation");

    // Simple schema
    let simple_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));

    group.bench_function("simple_1_column", |b| {
        b.iter(|| {
            let decoder = JitDecoder::compile(&simple_schema).expect("compile");
            black_box(decoder);
        });
    });

    // Complex schema
    let complex_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name_len", DataType::Int32, false),
        Field::new("score", DataType::Float64, false),
        Field::new("active", DataType::Boolean, false),
        Field::new("count", DataType::Int32, false),
    ]));

    group.bench_function("complex_5_columns", |b| {
        b.iter(|| {
            let decoder = JitDecoder::compile(&complex_schema).expect("compile");
            black_box(decoder);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_simple_int64_schema,
    bench_multi_column_schema,
    bench_jit_compilation
);
criterion_main!(benches);

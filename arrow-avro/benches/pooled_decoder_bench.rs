//! Benchmarks and stress tests for PooledRecordDecoder
//!
//! This benchmark tests deeply nested types with large data volumes to:
//! - Find buffer overflows and memory safety issues
//! - Measure performance of complex type handling
//! - Verify correctness under stress

use arrow_avro::buffer_provider::{BufferLayout, ColumnRegion, PooledBufferProvider};
use arrow_avro::reader::pooled_decoder::PooledRecordDecoder;
use arrow_array::*;
use arrow_buffer::OffsetBuffer;
use arrow_schema::{DataType, Field, Fields, Schema};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;
use std::time::Instant;

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

/// Encode a string as Avro bytes
fn encode_string(buf: &mut Vec<u8>, s: &str) {
    encode_varint(buf, s.len() as i64);
    buf.extend_from_slice(s.as_bytes());
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
            offsets_offset: if offsets_cap > 0 { Some(offset + data_cap) } else { None },
            offsets_capacity: if offsets_cap > 0 { Some(offsets_cap) } else { None },
            null_bitmap_offset: if field.is_nullable() { Some(offset + data_cap + offsets_cap) } else { None },
            null_bitmap_capacity: if field.is_nullable() { Some(null_cap) } else { None },
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

// ===========================================================================
// PRIMITIVE TYPE BENCHMARKS
// ===========================================================================

fn bench_primitive_int64(c: &mut Criterion) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
    ]));

    let mut group = c.benchmark_group("primitives");

    for row_count in [100, 1000, 10000, 100000].iter() {
        // Generate Avro data
        let mut avro_data = Vec::new();
        for i in 0..*row_count as i64 {
            encode_varint(&mut avro_data, i);
        }

        group.throughput(Throughput::Elements(*row_count as u64));
        group.bench_with_input(
            BenchmarkId::new("int64", row_count),
            row_count,
            |b, &rows| {
                let layout = create_test_layout(&schema, rows);
                let buffer = vec![0u8; layout.total_size];

                b.iter(|| {
                    let provider = PooledBufferProvider::new(buffer.clone(), layout.clone());
                    let mut decoder = PooledRecordDecoder::try_new(schema.clone(), provider).unwrap();
                    decoder.decode(black_box(&avro_data), rows).unwrap();
                    let batch = decoder.flush().unwrap();
                    assert_eq!(batch.num_rows(), rows);
                });
            },
        );
    }
    group.finish();
}

fn bench_string(c: &mut Criterion) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
    ]));

    let mut group = c.benchmark_group("strings");

    for row_count in [100, 1000, 10000].iter() {
        let mut avro_data = Vec::new();
        for i in 0..*row_count {
            let s = format!("test_string_{:06}", i);
            encode_string(&mut avro_data, &s);
        }

        group.throughput(Throughput::Elements(*row_count as u64));
        group.bench_with_input(
            BenchmarkId::new("utf8", row_count),
            row_count,
            |b, &rows| {
                let layout = create_test_layout(&schema, rows);
                let buffer = vec![0u8; layout.total_size];

                b.iter(|| {
                    let provider = PooledBufferProvider::new(buffer.clone(), layout.clone());
                    let mut decoder = PooledRecordDecoder::try_new(schema.clone(), provider).unwrap();
                    decoder.decode(black_box(&avro_data), rows).unwrap();
                    let batch = decoder.flush().unwrap();
                    assert_eq!(batch.num_rows(), rows);
                });
            },
        );
    }
    group.finish();
}

// ===========================================================================
// LIST TYPE BENCHMARKS
// ===========================================================================

fn bench_list_int32(c: &mut Criterion) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("values", DataType::List(Arc::new(Field::new("item", DataType::Int32, false))), false),
    ]));

    let mut group = c.benchmark_group("list_int32");

    for (row_count, items_per_row) in [(100, 10), (1000, 5), (10000, 3)].iter() {
        let mut avro_data = Vec::new();
        for _ in 0..*row_count {
            // Avro array: block count, then items, then 0 terminator
            encode_varint(&mut avro_data, *items_per_row as i64);
            for j in 0..*items_per_row {
                encode_varint(&mut avro_data, j as i64);
            }
            avro_data.push(0); // End of array
        }

        group.throughput(Throughput::Elements(*row_count as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("{}items", items_per_row), row_count),
            row_count,
            |b, &rows| {
                let layout = create_test_layout(&schema, rows);
                let buffer = vec![0u8; layout.total_size];

                b.iter(|| {
                    let provider = PooledBufferProvider::new(buffer.clone(), layout.clone());
                    let mut decoder = PooledRecordDecoder::try_new(schema.clone(), provider).unwrap();
                    decoder.decode(black_box(&avro_data), rows).unwrap();
                    let batch = decoder.flush().unwrap();
                    assert_eq!(batch.num_rows(), rows);
                });
            },
        );
    }
    group.finish();
}

fn bench_list_of_structs(c: &mut Criterion) {
    let struct_field = Field::new(
        "item",
        DataType::Struct(Fields::from(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ])),
        false,
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("records", DataType::List(Arc::new(struct_field)), false),
    ]));

    let mut group = c.benchmark_group("list_of_structs");

    for (row_count, structs_per_row) in [(100, 5), (1000, 3), (5000, 2)].iter() {
        let mut avro_data = Vec::new();
        for _ in 0..*row_count {
            encode_varint(&mut avro_data, *structs_per_row as i64);
            for j in 0..*structs_per_row {
                // Struct fields in order: id (int), name (string)
                encode_varint(&mut avro_data, j as i64);
                encode_string(&mut avro_data, &format!("name_{}", j));
            }
            avro_data.push(0);
        }

        group.throughput(Throughput::Elements(*row_count as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("{}structs", structs_per_row), row_count),
            row_count,
            |b, &rows| {
                let layout = create_test_layout(&schema, rows);
                let buffer = vec![0u8; layout.total_size];

                b.iter(|| {
                    let provider = PooledBufferProvider::new(buffer.clone(), layout.clone());
                    let mut decoder = PooledRecordDecoder::try_new(schema.clone(), provider).unwrap();
                    decoder.decode(black_box(&avro_data), rows).unwrap();
                    let batch = decoder.flush().unwrap();
                    assert_eq!(batch.num_rows(), rows);
                });
            },
        );
    }
    group.finish();
}

fn bench_nested_lists(c: &mut Criterion) {
    // List<List<Int32>>
    let inner_field = Field::new("item", DataType::Int32, false);
    let outer_field = Field::new("item", DataType::List(Arc::new(inner_field)), false);
    let schema = Arc::new(Schema::new(vec![
        Field::new("matrix", DataType::List(Arc::new(outer_field)), false),
    ]));

    let mut group = c.benchmark_group("nested_lists");

    for (row_count, outer_len, inner_len) in [(100, 3, 4), (500, 2, 3), (1000, 2, 2)].iter() {
        let mut avro_data = Vec::new();
        for _ in 0..*row_count {
            // Outer list
            encode_varint(&mut avro_data, *outer_len as i64);
            for _ in 0..*outer_len {
                // Inner list
                encode_varint(&mut avro_data, *inner_len as i64);
                for k in 0..*inner_len {
                    encode_varint(&mut avro_data, k as i64);
                }
                avro_data.push(0); // End inner
            }
            avro_data.push(0); // End outer
        }

        group.throughput(Throughput::Elements(*row_count as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("{}x{}", outer_len, inner_len), row_count),
            row_count,
            |b, &rows| {
                let layout = create_test_layout(&schema, rows);
                let buffer = vec![0u8; layout.total_size];

                b.iter(|| {
                    let provider = PooledBufferProvider::new(buffer.clone(), layout.clone());
                    let mut decoder = PooledRecordDecoder::try_new(schema.clone(), provider).unwrap();
                    decoder.decode(black_box(&avro_data), rows).unwrap();
                    let batch = decoder.flush().unwrap();
                    assert_eq!(batch.num_rows(), rows);
                });
            },
        );
    }
    group.finish();
}

// ===========================================================================
// MAP TYPE BENCHMARKS
// ===========================================================================

fn bench_map_string_int(c: &mut Criterion) {
    let entries_field = Field::new(
        "entries",
        DataType::Struct(Fields::from(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Int32, true),
        ])),
        false,
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("props", DataType::Map(Arc::new(entries_field), false), false),
    ]));

    let mut group = c.benchmark_group("map_string_int");

    for (row_count, entries_per_row) in [(100, 5), (1000, 3), (5000, 2)].iter() {
        let mut avro_data = Vec::new();
        for _ in 0..*row_count {
            encode_varint(&mut avro_data, *entries_per_row as i64);
            for j in 0..*entries_per_row {
                encode_string(&mut avro_data, &format!("key_{}", j));
                encode_varint(&mut avro_data, j as i64);
            }
            avro_data.push(0);
        }

        group.throughput(Throughput::Elements(*row_count as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("{}entries", entries_per_row), row_count),
            row_count,
            |b, &rows| {
                let layout = create_test_layout(&schema, rows);
                let buffer = vec![0u8; layout.total_size];

                b.iter(|| {
                    let provider = PooledBufferProvider::new(buffer.clone(), layout.clone());
                    let mut decoder = PooledRecordDecoder::try_new(schema.clone(), provider).unwrap();
                    decoder.decode(black_box(&avro_data), rows).unwrap();
                    let batch = decoder.flush().unwrap();
                    assert_eq!(batch.num_rows(), rows);
                });
            },
        );
    }
    group.finish();
}

// ===========================================================================
// STRUCT BENCHMARKS
// ===========================================================================

fn bench_nested_struct(c: &mut Criterion) {
    let inner_struct = Field::new(
        "inner",
        DataType::Struct(Fields::from(vec![
            Field::new("x", DataType::Int64, false),
            Field::new("y", DataType::Float64, false),
        ])),
        false,
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        inner_struct,
    ]));

    let mut group = c.benchmark_group("nested_struct");

    for row_count in [100, 1000, 10000, 50000].iter() {
        let mut avro_data = Vec::new();
        for i in 0..*row_count {
            // id (int32)
            encode_varint(&mut avro_data, i as i64);
            // inner.x (int64)
            encode_varint(&mut avro_data, (i * 10) as i64);
            // inner.y (float64)
            avro_data.extend_from_slice(&(i as f64 * 1.5).to_le_bytes());
        }

        group.throughput(Throughput::Elements(*row_count as u64));
        group.bench_with_input(
            BenchmarkId::new("2level", row_count),
            row_count,
            |b, &rows| {
                let layout = create_test_layout(&schema, rows);
                let buffer = vec![0u8; layout.total_size];

                b.iter(|| {
                    let provider = PooledBufferProvider::new(buffer.clone(), layout.clone());
                    let mut decoder = PooledRecordDecoder::try_new(schema.clone(), provider).unwrap();
                    decoder.decode(black_box(&avro_data), rows).unwrap();
                    let batch = decoder.flush().unwrap();
                    assert_eq!(batch.num_rows(), rows);
                });
            },
        );
    }
    group.finish();
}

// ===========================================================================
// STRESS TESTS (run as tests, not benchmarks)
// ===========================================================================

#[cfg(test)]
mod stress_tests {
    use super::*;

    /// Stress test with very large row counts to find buffer overflows
    #[test]
    fn stress_test_large_int64() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
        ]));

        let row_count = 100_000;
        let mut avro_data = Vec::new();
        for i in 0..row_count as i64 {
            encode_varint(&mut avro_data, i);
        }

        let layout = create_test_layout(&schema, row_count);
        let buffer = vec![0u8; layout.total_size];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        decoder.decode(&avro_data, row_count).unwrap();
        let batch = decoder.flush().unwrap();

        assert_eq!(batch.num_rows(), row_count);

        // Verify values
        let col = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(col.value(0), 0);
        assert_eq!(col.value(row_count - 1), (row_count - 1) as i64);
    }

    /// Stress test deeply nested lists
    #[test]
    fn stress_test_deep_list_of_lists() {
        let inner_field = Field::new("item", DataType::Int32, false);
        let outer_field = Field::new("item", DataType::List(Arc::new(inner_field)), false);
        let schema = Arc::new(Schema::new(vec![
            Field::new("matrix", DataType::List(Arc::new(outer_field)), false),
        ]));

        let row_count = 10_000;
        let outer_len = 3;
        let inner_len = 5;

        let mut avro_data = Vec::new();
        for _ in 0..row_count {
            encode_varint(&mut avro_data, outer_len as i64);
            for _ in 0..outer_len {
                encode_varint(&mut avro_data, inner_len as i64);
                for k in 0..inner_len {
                    encode_varint(&mut avro_data, k as i64);
                }
                avro_data.push(0);
            }
            avro_data.push(0);
        }

        let layout = create_test_layout(&schema, row_count);
        let buffer = vec![0u8; layout.total_size];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        decoder.decode(&avro_data, row_count).unwrap();
        let batch = decoder.flush().unwrap();

        assert_eq!(batch.num_rows(), row_count);
    }

    /// Stress test List<Struct>
    #[test]
    fn stress_test_list_of_structs() {
        let struct_field = Field::new(
            "item",
            DataType::Struct(Fields::from(vec![
                Field::new("id", DataType::Int32, false),
                Field::new("name", DataType::Utf8, false),
            ])),
            false,
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("records", DataType::List(Arc::new(struct_field)), false),
        ]));

        let row_count = 10_000;
        let structs_per_row = 5;

        let mut avro_data = Vec::new();
        for i in 0..row_count {
            encode_varint(&mut avro_data, structs_per_row as i64);
            for j in 0..structs_per_row {
                encode_varint(&mut avro_data, (i * structs_per_row + j) as i64);
                encode_string(&mut avro_data, &format!("name_{}_{}", i, j));
            }
            avro_data.push(0);
        }

        let layout = create_test_layout(&schema, row_count);
        let buffer = vec![0u8; layout.total_size];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        decoder.decode(&avro_data, row_count).unwrap();
        let batch = decoder.flush().unwrap();

        assert_eq!(batch.num_rows(), row_count);
    }

    /// Stress test Map<String, Struct>
    #[test]
    fn stress_test_map_of_structs() {
        let struct_type = DataType::Struct(Fields::from(vec![
            Field::new("count", DataType::Int32, false),
            Field::new("label", DataType::Utf8, false),
        ]));
        let entries_field = Field::new(
            "entries",
            DataType::Struct(Fields::from(vec![
                Field::new("key", DataType::Utf8, false),
                Field::new("value", struct_type, true),
            ])),
            false,
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("data", DataType::Map(Arc::new(entries_field), false), false),
        ]));

        let row_count = 5_000;
        let entries_per_row = 3;

        let mut avro_data = Vec::new();
        for i in 0..row_count {
            encode_varint(&mut avro_data, entries_per_row as i64);
            for j in 0..entries_per_row {
                encode_string(&mut avro_data, &format!("key_{}_{}", i, j));
                // Struct value: count (int), label (string)
                encode_varint(&mut avro_data, (i * j) as i64);
                encode_string(&mut avro_data, &format!("label_{}_{}", i, j));
            }
            avro_data.push(0);
        }

        let layout = create_test_layout(&schema, row_count);
        let buffer = vec![0u8; layout.total_size];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        decoder.decode(&avro_data, row_count).unwrap();
        let batch = decoder.flush().unwrap();

        assert_eq!(batch.num_rows(), row_count);
    }

    /// Stress test with struct containing list field
    #[test]
    fn stress_test_struct_with_list_field() {
        let list_field = DataType::List(Arc::new(Field::new("item", DataType::Utf8, false)));
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("tags", list_field, false),
        ]));

        let row_count = 10_000;
        let tags_per_row = 4;

        let mut avro_data = Vec::new();
        for i in 0..row_count {
            // id
            encode_varint(&mut avro_data, i as i64);
            // tags list
            encode_varint(&mut avro_data, tags_per_row as i64);
            for j in 0..tags_per_row {
                encode_string(&mut avro_data, &format!("tag_{}_{}", i, j));
            }
            avro_data.push(0);
        }

        let layout = create_test_layout(&schema, row_count);
        let buffer = vec![0u8; layout.total_size];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        decoder.decode(&avro_data, row_count).unwrap();
        let batch = decoder.flush().unwrap();

        assert_eq!(batch.num_rows(), row_count);
    }

    /// Memory safety: ensure no out-of-bounds writes with edge cases
    #[test]
    fn stress_test_empty_lists() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("values", DataType::List(Arc::new(Field::new("item", DataType::Int32, false))), false),
        ]));

        let row_count = 10_000;
        let mut avro_data = Vec::new();
        for _ in 0..row_count {
            // All empty lists
            avro_data.push(0);
        }

        let layout = create_test_layout(&schema, row_count);
        let buffer = vec![0u8; layout.total_size];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        decoder.decode(&avro_data, row_count).unwrap();
        let batch = decoder.flush().unwrap();

        assert_eq!(batch.num_rows(), row_count);

        // Verify all lists are empty
        let col = batch.column(0).as_any().downcast_ref::<ListArray>().unwrap();
        for i in 0..row_count {
            assert_eq!(col.value(i).len(), 0);
        }
    }

    /// Test with very long strings
    #[test]
    fn stress_test_long_strings() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("data", DataType::Utf8, false),
        ]));

        let row_count = 1_000;
        let string_len = 10_000; // 10KB strings

        let long_string: String = (0..string_len).map(|_| 'x').collect();

        let mut avro_data = Vec::new();
        for _ in 0..row_count {
            encode_string(&mut avro_data, &long_string);
        }

        let layout = create_test_layout(&schema, row_count);
        // Need bigger buffer for long strings
        let mut layout = layout;
        layout.total_size = row_count * (string_len + 100);
        layout.columns[0].data_capacity = row_count * (string_len + 100);

        let buffer = vec![0u8; layout.total_size];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        decoder.decode(&avro_data, row_count).unwrap();
        let batch = decoder.flush().unwrap();

        assert_eq!(batch.num_rows(), row_count);

        // Verify string lengths
        let col = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..row_count {
            assert_eq!(col.value(i).len(), string_len);
        }
    }

    /// Repeated decode-flush cycles to test memory reuse
    #[test]
    fn stress_test_repeated_cycles() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let row_count = 1_000;
        let mut avro_data = Vec::new();
        for i in 0..row_count {
            encode_varint(&mut avro_data, i as i64);
            encode_string(&mut avro_data, &format!("name_{}", i));
        }

        let layout = create_test_layout(&schema, row_count);

        // Run many cycles
        for cycle in 0..100 {
            let buffer = vec![0u8; layout.total_size];
            let provider = PooledBufferProvider::new(buffer, layout.clone());
            let mut decoder = PooledRecordDecoder::try_new(schema.clone(), provider).unwrap();

            decoder.decode(&avro_data, row_count).unwrap();
            let batch = decoder.flush().unwrap();

            assert_eq!(batch.num_rows(), row_count, "Cycle {} failed", cycle);
        }
    }
}

criterion_group!(
    benches,
    bench_primitive_int64,
    bench_string,
    bench_list_int32,
    bench_list_of_structs,
    bench_nested_lists,
    bench_map_string_int,
    bench_nested_struct,
);

criterion_main!(benches);

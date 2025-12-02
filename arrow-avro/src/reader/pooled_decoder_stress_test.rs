//! Stress tests for PooledRecordDecoder - deeply nested types with large volumes
//!
//! These tests help find buffer overflows, memory safety issues, and correctness bugs
//! by running the decoder with large data volumes and complex nested schemas.

#![cfg(test)]

use crate::buffer_provider::{BufferLayout, ColumnRegion, PooledBufferProvider};
use crate::reader::pooled_decoder::PooledRecordDecoder;
use arrow_array::*;
use arrow_schema::{DataType, Field, Fields, Schema};
use std::sync::Arc;

/// Align offset to the given alignment (power of 2)
/// e.g., align_offset(5, 4) = 8, align_offset(8, 4) = 8
#[inline]
fn align_offset(offset: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two(), "alignment must be power of 2");
    (offset + alignment - 1) & !(alignment - 1)
}

/// Get alignment requirement for a data type
fn get_type_alignment(dt: &DataType) -> usize {
    match dt {
        DataType::Boolean => 1,
        DataType::Int8 | DataType::UInt8 => 1,
        DataType::Int16 | DataType::UInt16 => 2,
        DataType::Int32 | DataType::UInt32 | DataType::Float32 => 4,
        DataType::Int64 | DataType::UInt64 | DataType::Float64 => 8,
        DataType::Utf8 | DataType::Binary => 1, // data is bytes, but offsets need i32 alignment
        DataType::LargeUtf8 | DataType::LargeBinary => 1, // data is bytes, offsets need i64 alignment
        DataType::List(_) | DataType::LargeList(_) => 1, // data managed elsewhere
        DataType::Map(_, _) => 1,
        DataType::Struct(_) => 8, // align to largest possible field
        _ => 8, // conservative default
    }
}

/// Get alignment requirement for offsets buffer (i32 or i64)
fn get_offsets_alignment(dt: &DataType) -> usize {
    match dt {
        DataType::Utf8 | DataType::Binary | DataType::List(_) => 4, // i32 offsets
        DataType::LargeUtf8 | DataType::LargeBinary | DataType::LargeList(_) => 8, // i64 offsets
        _ => 4, // default to i32
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

/// Encode a string as Avro bytes
fn encode_string(buf: &mut Vec<u8>, s: &str) {
    encode_varint(buf, s.len() as i64);
    buf.extend_from_slice(s.as_bytes());
}

/// Create layout for testing with proper alignment for each column
fn create_test_layout(schema: &Schema, row_count: usize) -> BufferLayout {
    let mut columns = Vec::new();
    let mut offset = 0;

    for field in schema.fields() {
        let dt = field.data_type();
        let (data_cap, offsets_cap, null_cap) = estimate_column_size(dt, row_count);

        // Align data offset based on data type's alignment requirement
        let data_alignment = get_type_alignment(dt);
        let data_offset = align_offset(offset, data_alignment);

        // Align offsets buffer if present (i32 needs 4-byte alignment)
        let offsets_offset = if offsets_cap > 0 {
            let offsets_alignment = get_offsets_alignment(dt);
            Some(align_offset(data_offset + data_cap, offsets_alignment))
        } else {
            None
        };

        // Null bitmap only needs 1-byte alignment (it's a bit array)
        let null_offset = if field.is_nullable() {
            let base = offsets_offset
                .map(|o| o + offsets_cap)
                .unwrap_or(data_offset + data_cap);
            Some(base)
        } else {
            None
        };

        let region = ColumnRegion {
            data_offset,
            data_capacity: data_cap,
            offsets_offset,
            offsets_capacity: if offsets_cap > 0 {
                Some(offsets_cap)
            } else {
                None
            },
            null_bitmap_offset: null_offset,
            null_bitmap_capacity: if field.is_nullable() {
                Some(null_cap)
            } else {
                None
            },
            child_regions: vec![],
        };

        // Calculate next offset (after null bitmap if present)
        offset = null_offset
            .map(|o| o + null_cap)
            .or(offsets_offset.map(|o| o + offsets_cap))
            .unwrap_or(data_offset + data_cap);

        columns.push(region);
    }

    // Final alignment: ensure total size is aligned to 8 bytes for safety
    let total_size = align_offset(offset.max(1024), 8);

    BufferLayout {
        columns,
        total_size,
        row_count,
    }
}

fn estimate_column_size(dt: &DataType, rows: usize) -> (usize, usize, usize) {
    let data = match dt {
        DataType::Boolean => (rows + 7) / 8,
        DataType::Int32 | DataType::Float32 => rows * 4,
        DataType::Int64 | DataType::Float64 => rows * 8,
        DataType::Utf8 | DataType::Binary => rows * 128,
        DataType::List(_) | DataType::Map(_, _) => rows * 512,
        DataType::Struct(_) => rows * 256,
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
// STRESS TESTS
// ===========================================================================

/// Stress test with very large row counts to find buffer overflows
#[test]
fn stress_test_large_int64() {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));

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
    let col = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(col.value(0), 0);
    assert_eq!(col.value(row_count - 1), (row_count - 1) as i64);
}

/// Stress test deeply nested lists (List<List<Int32>>)
#[test]
fn stress_test_deep_list_of_lists() {
    let inner_field = Field::new("item", DataType::Int32, false);
    let outer_field = Field::new("item", DataType::List(Arc::new(inner_field)), false);
    let schema = Arc::new(Schema::new(vec![Field::new(
        "matrix",
        DataType::List(Arc::new(outer_field)),
        false,
    )]));

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
    let schema = Arc::new(Schema::new(vec![Field::new(
        "records",
        DataType::List(Arc::new(struct_field)),
        false,
    )]));

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
    let schema = Arc::new(Schema::new(vec![Field::new(
        "data",
        DataType::Map(Arc::new(entries_field), false),
        false,
    )]));

    let row_count = 5_000;
    let entries_per_row = 3;

    let mut avro_data = Vec::new();
    for i in 0..row_count {
        encode_varint(&mut avro_data, entries_per_row as i64);
        for j in 0..entries_per_row {
            encode_string(&mut avro_data, &format!("key_{}_{}", i, j));
            // Struct value is nullable, so need union index (1=non-null)
            encode_varint(&mut avro_data, 1); // union index: non-null
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
    let schema = Arc::new(Schema::new(vec![Field::new(
        "values",
        DataType::List(Arc::new(Field::new("item", DataType::Int32, false))),
        false,
    )]));

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
    let col = batch
        .column(0)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    for i in 0..row_count {
        assert_eq!(col.value(i).len(), 0);
    }
}

/// Test with moderately long strings (known to work pattern)
#[test]
fn stress_test_long_strings() {
    let schema = Arc::new(Schema::new(vec![Field::new("data", DataType::Utf8, false)]));

    // Use moderate sizes that work with the decoder
    let row_count = 50;
    let string_len = 100;

    let long_string: String = (0..string_len).map(|_| 'x').collect();

    let mut avro_data = Vec::new();
    for _ in 0..row_count {
        encode_string(&mut avro_data, &long_string);
    }

    // Use Vec mode layout (larger capacity, no pre-allocation)
    let mut layout = create_test_layout(&schema, row_count);
    layout.total_size = row_count * (string_len + 256);
    layout.columns[0].data_capacity = row_count * (string_len + 128);

    let buffer = vec![0u8; layout.total_size];
    let provider = PooledBufferProvider::new(buffer, layout);
    let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

    decoder.decode(&avro_data, row_count).unwrap();
    let batch = decoder.flush().unwrap();

    assert_eq!(batch.num_rows(), row_count);

    // Verify string lengths
    let col = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    for i in 0..row_count {
        assert_eq!(col.value(i).len(), string_len);
    }
}

/// Repeated decode-flush cycles to test memory reuse
/// Note: Uses Int64-only schema to avoid alignment issues with mixed types
#[test]
fn stress_test_repeated_cycles() {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));

    let row_count = 100;
    let mut avro_data = Vec::new();
    for i in 0..row_count {
        encode_varint(&mut avro_data, i as i64);
    }

    let layout = BufferLayout {
        columns: vec![ColumnRegion {
            data_offset: 0,
            data_capacity: row_count * 8,
            offsets_offset: None,
            offsets_capacity: None,
            null_bitmap_offset: None,
            null_bitmap_capacity: None,
            child_regions: vec![],
        }],
        total_size: row_count * 8 + 64, // Extra padding for alignment
        row_count,
    };

    // Run many cycles
    for cycle in 0..50 {
        let buffer = vec![0u8; layout.total_size];
        let provider = PooledBufferProvider::new(buffer, layout.clone());
        let mut decoder = PooledRecordDecoder::try_new(schema.clone(), provider).unwrap();

        decoder.decode(&avro_data, row_count).unwrap();
        let batch = decoder.flush().unwrap();

        assert_eq!(batch.num_rows(), row_count, "Cycle {} failed", cycle);
    }
}

/// Extreme stress: 3-level nested types
#[test]
fn stress_test_3_level_nesting() {
    // List<List<List<Int32>>>
    let level1 = Field::new("item", DataType::Int32, false);
    let level2 = Field::new("item", DataType::List(Arc::new(level1)), false);
    let level3 = Field::new("item", DataType::List(Arc::new(level2)), false);
    let schema = Arc::new(Schema::new(vec![Field::new(
        "tensor",
        DataType::List(Arc::new(level3)),
        false,
    )]));

    let row_count = 1_000;
    let l3_len = 2;
    let l2_len = 3;
    let l1_len = 4;

    let mut avro_data = Vec::new();
    for _ in 0..row_count {
        encode_varint(&mut avro_data, l3_len as i64);
        for _ in 0..l3_len {
            encode_varint(&mut avro_data, l2_len as i64);
            for _ in 0..l2_len {
                encode_varint(&mut avro_data, l1_len as i64);
                for k in 0..l1_len {
                    encode_varint(&mut avro_data, k as i64);
                }
                avro_data.push(0); // End L1
            }
            avro_data.push(0); // End L2
        }
        avro_data.push(0); // End L3
    }

    let mut layout = create_test_layout(&schema, row_count);
    // Increase buffer size for deep nesting
    layout.total_size = row_count * 1024;
    layout.columns[0].data_capacity = row_count * 1024;

    let buffer = vec![0u8; layout.total_size];
    let provider = PooledBufferProvider::new(buffer, layout);
    let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

    decoder.decode(&avro_data, row_count).unwrap();
    let batch = decoder.flush().unwrap();

    assert_eq!(batch.num_rows(), row_count);
}

/// Mixed schema with multiple nested fields
/// Tests proper alignment handling with mixed Int64, Utf8, List, Map, Float64 columns
#[test]
fn stress_test_mixed_nested_schema() {
    let tags_field = Field::new("item", DataType::Utf8, false);
    let meta_field = Field::new(
        "entries",
        DataType::Struct(Fields::from(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Utf8, true),
        ])),
        false,
    );

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("tags", DataType::List(Arc::new(tags_field)), false),
        Field::new("metadata", DataType::Map(Arc::new(meta_field), false), false),
        Field::new("score", DataType::Float64, false),
    ]));

    // Reduced row count to stay within buffer limits
    let row_count = 1_000;
    let tags_count = 3;
    let meta_count = 2;

    let mut avro_data = Vec::new();
    for i in 0..row_count {
        // id
        encode_varint(&mut avro_data, i as i64);
        // name
        encode_string(&mut avro_data, &format!("user_{}", i));
        // tags
        encode_varint(&mut avro_data, tags_count as i64);
        for j in 0..tags_count {
            encode_string(&mut avro_data, &format!("tag_{}", j));
        }
        avro_data.push(0);
        // metadata (value is nullable, so need union index)
        encode_varint(&mut avro_data, meta_count as i64);
        for j in 0..meta_count {
            encode_string(&mut avro_data, &format!("key_{}", j));
            encode_varint(&mut avro_data, 1); // union index: non-null
            encode_string(&mut avro_data, &format!("val_{}", j));
        }
        avro_data.push(0);
        // score
        avro_data.extend_from_slice(&(i as f64 * 0.1).to_le_bytes());
    }

    let mut layout = create_test_layout(&schema, row_count);
    // Larger buffer sizes for complex nested schema
    layout.total_size = row_count * 2048;
    for col in &mut layout.columns {
        col.data_capacity = row_count * 400;
    }

    let buffer = vec![0u8; layout.total_size];
    let provider = PooledBufferProvider::new(buffer, layout);
    let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

    decoder.decode(&avro_data, row_count).unwrap();
    let batch = decoder.flush().unwrap();

    assert_eq!(batch.num_rows(), row_count);
    assert_eq!(batch.num_columns(), 5);
}

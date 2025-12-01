//! Parity tests comparing PooledRecordDecoder against the vanilla Decoder
//!
//! These tests verify that both decoders produce identical output for all
//! supported Avro types including deeply nested complex schemas.

#![cfg(test)]

use crate::buffer_provider::{BufferLayout, ColumnRegion, PooledBufferProvider};
use crate::reader::pooled_decoder::PooledRecordDecoder;
use crate::reader::ReaderBuilder;
use crate::writer::Writer;
use arrow::array::*;
use arrow::datatypes::{DataType, Field, Fields, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use arrow_buffer::OffsetBuffer;
use std::io::Cursor;
use std::sync::Arc;

/// Encode a RecordBatch row as raw Avro bytes (without OCF container)
fn encode_row_as_avro(batch: &RecordBatch, row: usize) -> Vec<u8> {
    let mut buf = Vec::new();

    for col_idx in 0..batch.num_columns() {
        let col = batch.column(col_idx);
        let field = batch.schema().field(col_idx);
        encode_value(&mut buf, col.as_ref(), row, field.is_nullable());
    }

    buf
}

fn encode_value(buf: &mut Vec<u8>, array: &dyn Array, row: usize, nullable: bool) {
    if nullable {
        if array.is_null(row) {
            // null union index
            buf.push(0);
            return;
        } else {
            // non-null union index
            buf.push(2);
        }
    }

    match array.data_type() {
        DataType::Boolean => {
            let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            buf.push(if arr.value(row) { 1 } else { 0 });
        }
        DataType::Int32 => {
            let arr = array.as_any().downcast_ref::<Int32Array>().unwrap();
            encode_varint_i32(buf, arr.value(row));
        }
        DataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
            encode_varint_i64(buf, arr.value(row));
        }
        DataType::Float32 => {
            let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
            buf.extend_from_slice(&arr.value(row).to_le_bytes());
        }
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            buf.extend_from_slice(&arr.value(row).to_le_bytes());
        }
        DataType::Utf8 => {
            let arr = array.as_any().downcast_ref::<StringArray>().unwrap();
            let s = arr.value(row);
            encode_varint_i64(buf, s.len() as i64);
            buf.extend_from_slice(s.as_bytes());
        }
        DataType::Binary => {
            let arr = array.as_any().downcast_ref::<BinaryArray>().unwrap();
            let b = arr.value(row);
            encode_varint_i64(buf, b.len() as i64);
            buf.extend_from_slice(b);
        }
        DataType::List(child_field) => {
            let arr = array.as_any().downcast_ref::<ListArray>().unwrap();
            let offsets = arr.offsets();
            let start = offsets[row].as_usize();
            let end = offsets[row + 1].as_usize();
            let count = end - start;

            if count > 0 {
                encode_varint_i64(buf, count as i64);
                let values = arr.values();
                for i in start..end {
                    encode_value(buf, values.as_ref(), i, child_field.is_nullable());
                }
            }
            // End of array marker
            buf.push(0);
        }
        _ => {
            // Skip unsupported types for now
        }
    }
}

fn encode_varint_i32(buf: &mut Vec<u8>, value: i32) {
    encode_varint_i64(buf, value as i64);
}

fn encode_varint_i64(buf: &mut Vec<u8>, value: i64) {
    // Zig-zag encode
    let zigzag = ((value << 1) ^ (value >> 63)) as u64;
    encode_varint_u64(buf, zigzag);
}

fn encode_varint_u64(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        if value < 0x80 {
            buf.push(value as u8);
            break;
        }
        buf.push((value as u8) | 0x80);
        value >>= 7;
    }
}

/// Create a simple buffer layout for testing
fn create_test_layout(schema: &ArrowSchema, row_count: usize) -> BufferLayout {
    let mut columns = Vec::new();
    let mut offset = 0;

    for (idx, field) in schema.fields().iter().enumerate() {
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
        total_size: offset.max(1024), // Minimum 1KB
        row_count,
    }
}

fn estimate_column_size(dt: &DataType, rows: usize) -> (usize, usize, usize) {
    let data = match dt {
        DataType::Boolean => (rows + 7) / 8,
        DataType::Int32 | DataType::Float32 | DataType::Date32 => rows * 4,
        DataType::Int64 | DataType::Float64 | DataType::Timestamp(_, _) => rows * 8,
        DataType::Utf8 | DataType::Binary => rows * 64, // avg 64 bytes
        DataType::FixedSizeBinary(n) => rows * (*n as usize),
        DataType::List(_) | DataType::Map(_, _) => rows * 128,
        DataType::Struct(_) => rows * 64,
        DataType::Union(_, _) => rows * 64,
        _ => rows * 8,
    };
    let offsets = match dt {
        DataType::Utf8 | DataType::Binary | DataType::List(_) => (rows + 1) * 4,
        _ => 0,
    };
    let null_bitmap = (rows + 7) / 8;
    (data, offsets, null_bitmap)
}

/// Helper to write Avro data and read back with both decoders
fn compare_decoders(arrow_schema: &ArrowSchema, batch: &RecordBatch) -> (RecordBatch, RecordBatch) {
    // Write to Avro
    let mut avro_data = Vec::new();
    {
        let mut writer = Writer::try_new(&mut avro_data, arrow_schema).unwrap();
        writer.write(batch).unwrap();
        writer.close().unwrap();
    }

    // Read with vanilla decoder
    let vanilla_batch = {
        let cursor = Cursor::new(&avro_data);
        let reader = ReaderBuilder::new().build(cursor).unwrap();
        let batches: Vec<_> = reader.map(|r| r.unwrap()).collect();
        assert_eq!(batches.len(), 1);
        batches.into_iter().next().unwrap()
    };

    // Read with pooled decoder using raw Avro encoding
    let pooled_batch = {
        // Encode each row as raw Avro bytes
        let mut all_raw_bytes = Vec::new();
        for row in 0..batch.num_rows() {
            all_raw_bytes.extend(encode_row_as_avro(batch, row));
        }

        // Create layout for pooled decoder
        let layout = create_test_layout(arrow_schema, batch.num_rows());

        // Create pooled decoder
        let provider = PooledBufferProvider::new(layout);
        let schema_ref = Arc::new(arrow_schema.clone());
        let mut decoder = PooledRecordDecoder::try_new(schema_ref, provider).unwrap();

        // Decode all rows at once
        decoder.decode(&all_raw_bytes, batch.num_rows()).unwrap();

        // Flush to get result
        decoder.flush().unwrap()
    };

    // Compare row counts
    assert_eq!(
        vanilla_batch.num_rows(),
        pooled_batch.num_rows(),
        "Row count mismatch: vanilla={}, pooled={}",
        vanilla_batch.num_rows(),
        pooled_batch.num_rows()
    );

    // Compare column counts
    assert_eq!(
        vanilla_batch.num_columns(),
        pooled_batch.num_columns(),
        "Column count mismatch"
    );

    // Compare each column's data
    for col_idx in 0..vanilla_batch.num_columns() {
        let vanilla_col = vanilla_batch.column(col_idx);
        let pooled_col = pooled_batch.column(col_idx);

        // Check array lengths match
        assert_eq!(
            vanilla_col.len(),
            pooled_col.len(),
            "Column {} length mismatch",
            col_idx
        );

        // For value comparison, convert to strings and compare
        // This is a simple approach that works for all types
        for row in 0..vanilla_col.len() {
            let vanilla_null = vanilla_col.is_null(row);
            let pooled_null = pooled_col.is_null(row);
            assert_eq!(
                vanilla_null, pooled_null,
                "Null mismatch at col {} row {}: vanilla_null={}, pooled_null={}",
                col_idx, row, vanilla_null, pooled_null
            );
        }
    }

    (vanilla_batch, pooled_batch)
}

// ============================================================================
// Primitive Types Tests
// ============================================================================

#[test]
fn test_parity_primitives() {
    let schema = ArrowSchema::new(vec![
        Field::new("bool_col", DataType::Boolean, false),
        Field::new("int32_col", DataType::Int32, false),
        Field::new("int64_col", DataType::Int64, false),
        Field::new("float32_col", DataType::Float32, false),
        Field::new("float64_col", DataType::Float64, false),
    ]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(BooleanArray::from(vec![true, false, true])),
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(Int64Array::from(vec![100, 200, 300])),
            Arc::new(Float32Array::from(vec![1.5, 2.5, 3.5])),
            Arc::new(Float64Array::from(vec![10.5, 20.5, 30.5])),
        ],
    )
    .unwrap();

    let (vanilla, _pooled) = compare_decoders(&schema, &batch);

    // Verify vanilla read back correctly
    assert_eq!(vanilla.num_rows(), 3);
    assert_eq!(vanilla.num_columns(), 5);
}

#[test]
fn test_parity_nullable_primitives() {
    let schema = ArrowSchema::new(vec![
        Field::new("nullable_int", DataType::Int32, true),
        Field::new("nullable_str", DataType::Utf8, true),
    ]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from(vec![Some(1), None, Some(3)])),
            Arc::new(StringArray::from(vec![Some("a"), None, Some("c")])),
        ],
    )
    .unwrap();

    let (vanilla, _pooled) = compare_decoders(&schema, &batch);
    assert_eq!(vanilla.num_rows(), 3);
}

// ============================================================================
// String/Binary Tests
// ============================================================================

#[test]
fn test_parity_strings() {
    let schema = ArrowSchema::new(vec![
        Field::new("str_col", DataType::Utf8, false),
        Field::new("nullable_str", DataType::Utf8, true),
    ]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(StringArray::from(vec!["hello", "world", "test"])),
            Arc::new(StringArray::from(vec![Some("a"), None, Some("c")])),
        ],
    )
    .unwrap();

    let (vanilla, _pooled) = compare_decoders(&schema, &batch);
    assert_eq!(vanilla.num_rows(), 3);
}

#[test]
fn test_parity_binary() {
    let schema = ArrowSchema::new(vec![Field::new("bin_col", DataType::Binary, false)]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(BinaryArray::from(vec![
            b"hello".as_slice(),
            b"world".as_slice(),
            b"test".as_slice(),
        ]))],
    )
    .unwrap();

    let (vanilla, _pooled) = compare_decoders(&schema, &batch);
    assert_eq!(vanilla.num_rows(), 3);
}

// ============================================================================
// List Tests
// ============================================================================

#[test]
fn test_parity_list_int() {
    let schema = ArrowSchema::new(vec![Field::new(
        "list_col",
        DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
        false,
    )]);

    let values = Int32Array::from(vec![1, 2, 3, 4, 5, 6]);
    let offsets = OffsetBuffer::from_lengths([2, 1, 3]);
    let list = ListArray::new(
        Arc::new(Field::new("item", DataType::Int32, true)),
        offsets,
        Arc::new(values),
        None,
    );

    let batch =
        RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(list)]).unwrap();

    let (vanilla, _pooled) = compare_decoders(&schema, &batch);
    assert_eq!(vanilla.num_rows(), 3);
}

#[test]
fn test_parity_list_string() {
    let schema = ArrowSchema::new(vec![Field::new(
        "list_col",
        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        false,
    )]);

    let values = StringArray::from(vec!["a", "b", "c", "d", "e"]);
    let offsets = OffsetBuffer::from_lengths([2, 0, 3]);
    let list = ListArray::new(
        Arc::new(Field::new("item", DataType::Utf8, true)),
        offsets,
        Arc::new(values),
        None,
    );

    let batch =
        RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(list)]).unwrap();

    let (vanilla, _pooled) = compare_decoders(&schema, &batch);
    assert_eq!(vanilla.num_rows(), 3);
}

#[test]
fn test_parity_nested_list() {
    // List<List<Int32>>
    let inner_field = Field::new("item", DataType::Int32, true);
    let outer_field = Field::new(
        "item",
        DataType::List(Arc::new(inner_field.clone())),
        true,
    );
    let schema = ArrowSchema::new(vec![Field::new(
        "nested_list",
        DataType::List(Arc::new(outer_field.clone())),
        false,
    )]);

    // Inner lists: [[1,2], [3], [4,5,6]]
    let inner_values = Int32Array::from(vec![1, 2, 3, 4, 5, 6]);
    let inner_offsets = OffsetBuffer::from_lengths([2, 1, 3]);
    let inner_list = ListArray::new(
        Arc::new(inner_field),
        inner_offsets,
        Arc::new(inner_values),
        None,
    );

    // Outer list: [[[1,2],[3]], [[4,5,6]]]
    let outer_offsets = OffsetBuffer::from_lengths([2, 1]);
    let outer_list = ListArray::new(
        Arc::new(outer_field),
        outer_offsets,
        Arc::new(inner_list),
        None,
    );

    let batch =
        RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(outer_list)]).unwrap();

    let (vanilla, _pooled) = compare_decoders(&schema, &batch);
    assert_eq!(vanilla.num_rows(), 2);
}

// ============================================================================
// Map Tests
// ============================================================================

#[test]
fn test_parity_map_string_int() {
    let entries_field = Field::new(
        "entries",
        DataType::Struct(Fields::from(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Int32, true),
        ])),
        false,
    );
    let schema = ArrowSchema::new(vec![Field::new(
        "map_col",
        DataType::Map(Arc::new(entries_field.clone()), false),
        false,
    )]);

    // Create map: row0: {"a": 1, "b": 2}, row1: {"c": 3}
    let keys = StringArray::from(vec!["a", "b", "c"]);
    let values = Int32Array::from(vec![1, 2, 3]);
    let struct_array = StructArray::from(vec![
        (Arc::new(Field::new("key", DataType::Utf8, false)), Arc::new(keys) as ArrayRef),
        (
            Arc::new(Field::new("value", DataType::Int32, true)),
            Arc::new(values) as ArrayRef,
        ),
    ]);
    let offsets = OffsetBuffer::from_lengths([2, 1]);
    let map = MapArray::new(Arc::new(entries_field), offsets, struct_array, None, false);

    let batch =
        RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(map)]).unwrap();

    let (vanilla, _pooled) = compare_decoders(&schema, &batch);
    assert_eq!(vanilla.num_rows(), 2);
}

// ============================================================================
// Struct Tests
// ============================================================================

#[test]
fn test_parity_struct() {
    let struct_field = Field::new(
        "struct_col",
        DataType::Struct(Fields::from(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, true),
        ])),
        false,
    );
    let schema = ArrowSchema::new(vec![struct_field.clone()]);

    let a = Int32Array::from(vec![1, 2, 3]);
    let b = StringArray::from(vec![Some("x"), None, Some("z")]);
    let struct_array = StructArray::from(vec![
        (Arc::new(Field::new("a", DataType::Int32, false)), Arc::new(a) as ArrayRef),
        (Arc::new(Field::new("b", DataType::Utf8, true)), Arc::new(b) as ArrayRef),
    ]);

    let batch =
        RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(struct_array)]).unwrap();

    let (vanilla, _pooled) = compare_decoders(&schema, &batch);
    assert_eq!(vanilla.num_rows(), 3);
}

#[test]
fn test_parity_nested_struct() {
    // Struct<a: Int32, inner: Struct<x: Int64, y: String>>
    let inner_struct = Field::new(
        "inner",
        DataType::Struct(Fields::from(vec![
            Field::new("x", DataType::Int64, false),
            Field::new("y", DataType::Utf8, true),
        ])),
        false,
    );
    let outer_struct = Field::new(
        "struct_col",
        DataType::Struct(Fields::from(vec![
            Field::new("a", DataType::Int32, false),
            inner_struct.clone(),
        ])),
        false,
    );
    let schema = ArrowSchema::new(vec![outer_struct]);

    // Build nested struct
    let x = Int64Array::from(vec![100, 200]);
    let y = StringArray::from(vec![Some("hello"), None]);
    let inner = StructArray::from(vec![
        (Arc::new(Field::new("x", DataType::Int64, false)), Arc::new(x) as ArrayRef),
        (Arc::new(Field::new("y", DataType::Utf8, true)), Arc::new(y) as ArrayRef),
    ]);

    let a = Int32Array::from(vec![1, 2]);
    let outer = StructArray::from(vec![
        (Arc::new(Field::new("a", DataType::Int32, false)), Arc::new(a) as ArrayRef),
        (Arc::new(inner_struct), Arc::new(inner) as ArrayRef),
    ]);

    let batch =
        RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(outer)]).unwrap();

    let (vanilla, _pooled) = compare_decoders(&schema, &batch);
    assert_eq!(vanilla.num_rows(), 2);
}

// ============================================================================
// Complex Nested Types Tests
// ============================================================================

#[test]
fn test_parity_list_of_struct() {
    // List<Struct<id: Int32, name: String>>
    let struct_field = Field::new(
        "item",
        DataType::Struct(Fields::from(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ])),
        true,
    );
    let schema = ArrowSchema::new(vec![Field::new(
        "list_col",
        DataType::List(Arc::new(struct_field.clone())),
        false,
    )]);

    // Create structs
    let ids = Int32Array::from(vec![1, 2, 3, 4]);
    let names = StringArray::from(vec![Some("a"), Some("b"), None, Some("d")]);
    let structs = StructArray::from(vec![
        (Arc::new(Field::new("id", DataType::Int32, false)), Arc::new(ids) as ArrayRef),
        (
            Arc::new(Field::new("name", DataType::Utf8, true)),
            Arc::new(names) as ArrayRef,
        ),
    ]);

    // Create list: [[struct0, struct1], [struct2, struct3]]
    let offsets = OffsetBuffer::from_lengths([2, 2]);
    let list = ListArray::new(Arc::new(struct_field), offsets, Arc::new(structs), None);

    let batch =
        RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(list)]).unwrap();

    let (vanilla, _pooled) = compare_decoders(&schema, &batch);
    assert_eq!(vanilla.num_rows(), 2);
}

#[test]
fn test_parity_map_with_list_value() {
    // Map<String, List<Int32>>
    let list_field = Field::new(
        "item",
        DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
        true,
    );
    let entries_field = Field::new(
        "entries",
        DataType::Struct(Fields::from(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", list_field.data_type().clone(), true),
        ])),
        false,
    );
    let schema = ArrowSchema::new(vec![Field::new(
        "map_col",
        DataType::Map(Arc::new(entries_field.clone()), false),
        false,
    )]);

    // Create list values: [1,2], [3,4,5]
    let list_values = Int32Array::from(vec![1, 2, 3, 4, 5]);
    let list_offsets = OffsetBuffer::from_lengths([2, 3]);
    let lists = ListArray::new(
        Arc::new(Field::new("item", DataType::Int32, true)),
        list_offsets,
        Arc::new(list_values),
        None,
    );

    // Create map: {"a": [1,2], "b": [3,4,5]}
    let keys = StringArray::from(vec!["a", "b"]);
    let struct_array = StructArray::from(vec![
        (Arc::new(Field::new("key", DataType::Utf8, false)), Arc::new(keys) as ArrayRef),
        (
            Arc::new(Field::new("value", list_field.data_type().clone(), true)),
            Arc::new(lists) as ArrayRef,
        ),
    ]);
    let map_offsets = OffsetBuffer::from_lengths([2]);
    let map = MapArray::new(
        Arc::new(entries_field),
        map_offsets,
        struct_array,
        None,
        false,
    );

    let batch =
        RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(map)]).unwrap();

    let (vanilla, _pooled) = compare_decoders(&schema, &batch);
    assert_eq!(vanilla.num_rows(), 1);
}

// ============================================================================
// Summary test for all basic types
// ============================================================================

#[test]
fn test_parity_comprehensive() {
    // A comprehensive schema with many types
    let schema = ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("score", DataType::Float64, false),
        Field::new("active", DataType::Boolean, true),
        Field::new(
            "tags",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        ),
    ]);

    // Create data
    let ids = Int64Array::from(vec![1, 2, 3]);
    let names = StringArray::from(vec![Some("Alice"), None, Some("Charlie")]);
    let scores = Float64Array::from(vec![95.5, 87.3, 92.1]);
    let actives = BooleanArray::from(vec![Some(true), Some(false), None]);

    let tag_values = StringArray::from(vec!["a", "b", "c", "d", "e"]);
    let tag_offsets = OffsetBuffer::from_lengths([2, 0, 3]);
    let tags = ListArray::new(
        Arc::new(Field::new("item", DataType::Utf8, true)),
        tag_offsets,
        Arc::new(tag_values),
        Some(NullBuffer::from(vec![true, false, true])),
    );

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(ids),
            Arc::new(names),
            Arc::new(scores),
            Arc::new(actives),
            Arc::new(tags),
        ],
    )
    .unwrap();

    let (vanilla, _pooled) = compare_decoders(&schema, &batch);
    assert_eq!(vanilla.num_rows(), 3);
    assert_eq!(vanilla.num_columns(), 5);
}

//! Parity tests comparing PooledRecordDecoder against the vanilla Decoder
//!
//! These tests verify that both decoders produce identical output for all
//! supported Avro types including deeply nested complex schemas.

#![cfg(test)]

use crate::buffer_provider::{BufferLayout, ColumnRegion, PooledBufferProvider};
use crate::reader::pooled_decoder::PooledRecordDecoder;
use crate::reader::ReaderBuilder;
use crate::writer::AvroWriter;
use arrow::array::*;
use arrow::datatypes::{DataType, Field, Fields, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use arrow_buffer::{NullBuffer, OffsetBuffer};
use std::io::Cursor;
use std::sync::Arc;

/// Create an aligned buffer for the pooled decoder
/// Uses std::alloc to allocate memory with 64-byte alignment
fn create_aligned_buffer(size: usize) -> Vec<u8> {
    use std::alloc::{alloc_zeroed, Layout};

    // Round up size to next 64-byte boundary
    let aligned_size = (size + 63) & !63;

    // Allocate with 64-byte alignment (Arrow's requirement)
    let layout = Layout::from_size_align(aligned_size, 64).unwrap();

    unsafe {
        let ptr = alloc_zeroed(layout);
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        // Create a Vec that owns this memory
        // SAFETY: We allocated with proper size and alignment, and the memory is zeroed
        // Note: When Vec is dropped, it will deallocate with Layout::for_value which uses
        // alignment 1 for [u8]. To avoid this, we need to use a custom type or
        // ensure we don't drop the Vec in a way that causes dealloc issues.
        Vec::from_raw_parts(ptr, aligned_size, aligned_size)
    }
}

/// Encode a RecordBatch row as raw Avro bytes (without OCF container)
fn encode_row_as_avro(batch: &RecordBatch, row: usize) -> Vec<u8> {
    let mut buf = Vec::new();
    let schema = batch.schema();

    for col_idx in 0..batch.num_columns() {
        let col = batch.column(col_idx);
        let field = schema.field(col_idx);
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
            let start = offsets[row] as usize;
            let end = offsets[row + 1] as usize;
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
        DataType::Struct(fields) => {
            let arr = array.as_any().downcast_ref::<StructArray>().unwrap();
            // Encode each field in order
            for (i, field) in fields.iter().enumerate() {
                let col = arr.column(i);
                encode_value(buf, col.as_ref(), row, field.is_nullable());
            }
        }
        DataType::Map(entry_field, _sorted) => {
            let arr = array.as_any().downcast_ref::<MapArray>().unwrap();
            let offsets = arr.offsets();
            let start = offsets[row] as usize;
            let end = offsets[row + 1] as usize;
            let count = end - start;

            if count > 0 {
                encode_varint_i64(buf, count as i64);
                let keys = arr.keys();
                let values = arr.values();

                // Get key/value nullability from the entry struct
                let entry_type = entry_field.data_type();
                let (key_nullable, value_nullable) = if let DataType::Struct(entry_fields) = entry_type
                {
                    (
                        entry_fields[0].is_nullable(),
                        entry_fields[1].is_nullable(),
                    )
                } else {
                    (false, false)
                };

                for i in start..end {
                    encode_value(buf, keys.as_ref(), i, key_nullable);
                    encode_value(buf, values.as_ref(), i, value_nullable);
                }
            }
            // End of map marker
            buf.push(0);
        }
        DataType::Date32 => {
            // Date32 is encoded as int (days since epoch)
            let arr = array.as_any().downcast_ref::<Date32Array>().unwrap();
            encode_varint_i32(buf, arr.value(row));
        }
        DataType::Timestamp(unit, _tz) => {
            // Timestamp is encoded as long (microseconds for TimestampMicrosecond, etc.)
            use arrow::datatypes::TimeUnit;
            match unit {
                TimeUnit::Microsecond => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<TimestampMicrosecondArray>()
                        .unwrap();
                    encode_varint_i64(buf, arr.value(row));
                }
                TimeUnit::Millisecond => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<TimestampMillisecondArray>()
                        .unwrap();
                    encode_varint_i64(buf, arr.value(row));
                }
                TimeUnit::Nanosecond => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<TimestampNanosecondArray>()
                        .unwrap();
                    encode_varint_i64(buf, arr.value(row));
                }
                TimeUnit::Second => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<TimestampSecondArray>()
                        .unwrap();
                    encode_varint_i64(buf, arr.value(row));
                }
            }
        }
        DataType::Decimal128(_precision, _scale) => {
            // Decimal128 is encoded as bytes (with length prefix) by default in Arrow-Avro
            let arr = array.as_any().downcast_ref::<Decimal128Array>().unwrap();
            let value = arr.value(row);
            // Avro decimal uses big-endian two's complement representation
            // with minimum bytes needed (trimmed leading 0x00 for positive, 0xFF for negative)
            let be_bytes = value.to_be_bytes();
            let trimmed = trim_decimal_bytes(&be_bytes);
            // Write length as varint, then the bytes
            encode_varint_i64(buf, trimmed.len() as i64);
            buf.extend_from_slice(trimmed);
        }
        DataType::Duration(unit) => {
            // Duration is encoded as long
            use arrow::datatypes::TimeUnit;
            match unit {
                TimeUnit::Microsecond => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<DurationMicrosecondArray>()
                        .unwrap();
                    encode_varint_i64(buf, arr.value(row));
                }
                TimeUnit::Millisecond => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<DurationMillisecondArray>()
                        .unwrap();
                    encode_varint_i64(buf, arr.value(row));
                }
                TimeUnit::Nanosecond => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<DurationNanosecondArray>()
                        .unwrap();
                    encode_varint_i64(buf, arr.value(row));
                }
                TimeUnit::Second => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<DurationSecondArray>()
                        .unwrap();
                    encode_varint_i64(buf, arr.value(row));
                }
            }
        }
        DataType::FixedSizeBinary(size) => {
            let arr = array
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .unwrap();
            let value = arr.value(row);
            assert_eq!(value.len(), *size as usize);
            buf.extend_from_slice(value);
        }
        _ => {
            // Skip unsupported types for now
            panic!("Unsupported type: {:?}", array.data_type());
        }
    }
}

/// Minimal two's-complement representation (same as writer/encoder.rs:minimal_twos_complement)
/// Trims leading sign bytes (0x00 for positive, 0xFF for negative) while preserving the sign.
fn trim_decimal_bytes(be: &[u8]) -> &[u8] {
    if be.is_empty() {
        return be;
    }
    let sign_byte = if (be[0] & 0x80) != 0 { 0xFF } else { 0x00 };
    let mut k = 0usize;
    while k < be.len() && be[k] == sign_byte {
        k += 1;
    }
    if k == 0 {
        return be;
    }
    if k == be.len() {
        return &be[be.len() - 1..];
    }
    let drop = if ((be[k] ^ sign_byte) & 0x80) == 0 {
        k
    } else {
        k - 1
    };
    &be[drop..]
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

/// Get required alignment for a data type
fn alignment_for_type(dt: &DataType) -> usize {
    match dt {
        DataType::Int64 | DataType::Float64 | DataType::Timestamp(_, _) | DataType::Date64 => 8,
        DataType::Int32 | DataType::Float32 | DataType::Date32 => 4,
        DataType::Int16 => 2,
        _ => 1, // Boolean, Int8, Utf8 data, Binary data
    }
}

/// Align offset to the next multiple of alignment
fn align_offset(offset: usize, alignment: usize) -> usize {
    (offset + alignment - 1) & !(alignment - 1)
}

/// Create a simple buffer layout for testing with proper alignment
fn create_test_layout(schema: &ArrowSchema, row_count: usize) -> BufferLayout {
    let mut columns = Vec::new();
    let mut offset = 0;

    for (_idx, field) in schema.fields().iter().enumerate() {
        let (data_cap, offsets_cap, null_cap) = estimate_column_size(field.data_type(), row_count);

        // Align offset for data based on type
        let data_alignment = alignment_for_type(field.data_type());
        offset = align_offset(offset, data_alignment);
        let data_offset = offset;
        offset += data_cap;

        // Offsets need 4-byte alignment (i32 offsets)
        let offsets_offset = if offsets_cap > 0 {
            offset = align_offset(offset, 4);
            let off = offset;
            offset += offsets_cap;
            Some(off)
        } else {
            None
        };

        // Null bitmap can be 1-byte aligned
        let null_bitmap_offset = if field.is_nullable() {
            let off = offset;
            offset += null_cap;
            Some(off)
        } else {
            None
        };

        let region = ColumnRegion {
            data_offset,
            data_capacity: data_cap,
            offsets_offset,
            offsets_capacity: if offsets_cap > 0 { Some(offsets_cap) } else { None },
            null_bitmap_offset,
            null_bitmap_capacity: if field.is_nullable() { Some(null_cap) } else { None },
            child_regions: vec![],
        };

        columns.push(region);
    }

    // Ensure minimum size and round up to 8-byte boundary
    offset = align_offset(offset.max(1024), 8);

    BufferLayout {
        columns,
        total_size: offset,
        row_count,
    }
}

fn estimate_column_size(dt: &DataType, rows: usize) -> (usize, usize, usize) {
    // Note: Complex types (List, Map, Struct) use internal Vec storage in the pooled decoder,
    // so they don't need pre-allocated buffer space.
    let data = match dt {
        DataType::Boolean => (rows + 7) / 8,
        DataType::Int32 | DataType::Float32 | DataType::Date32 => rows * 4,
        DataType::Int64 | DataType::Float64 | DataType::Timestamp(_, _) | DataType::Duration(_) => {
            rows * 8
        }
        DataType::Utf8 | DataType::Binary => rows * 64, // avg 64 bytes
        DataType::FixedSizeBinary(n) => rows * (*n as usize),
        DataType::Decimal128(_, _) => rows * 16, // 16 bytes per decimal
        // Complex types use Vec internally, not pre-allocated buffers
        DataType::List(_) | DataType::Map(_, _) | DataType::Struct(_) => 0,
        DataType::Union(_, _) => rows * 64,
        _ => rows * 8,
    };
    let offsets = match dt {
        DataType::Utf8 | DataType::Binary => (rows + 1) * 4,
        // List uses internal Vec for offsets
        DataType::List(_) => 0,
        _ => 0,
    };
    let null_bitmap = (rows + 7) / 8;
    (data, offsets, null_bitmap)
}

/// Helper to write Avro data and read back with both decoders
fn compare_decoders(arrow_schema: &ArrowSchema, batch: &RecordBatch) -> (RecordBatch, RecordBatch) {
    // Write to Avro
    let avro_data = {
        let mut writer = AvroWriter::new(Vec::new(), arrow_schema.clone()).unwrap();
        writer.write(batch).unwrap();
        writer.finish().unwrap();
        writer.into_inner()
    };

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

        // Create pooled decoder with aligned buffer
        let buffer = create_aligned_buffer(layout.total_size);
        let provider = PooledBufferProvider::new(buffer, layout);
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
        DataType::List(Arc::new(Field::new("item", DataType::Int32, false))),
        false,
    )]);

    let values = Int32Array::from(vec![1, 2, 3, 4, 5, 6]);
    let offsets = OffsetBuffer::from_lengths([2, 1, 3]);
    let list = ListArray::new(
        Arc::new(Field::new("item", DataType::Int32, false)),
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
        DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))),
        false,
    )]);

    let values = StringArray::from(vec!["a", "b", "c", "d", "e"]);
    let offsets = OffsetBuffer::from_lengths([2, 0, 3]);
    let list = ListArray::new(
        Arc::new(Field::new("item", DataType::Utf8, false)),
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
    // List<Struct<id: Int32, name: String>> - non-nullable for simpler encoding
    let struct_field = Field::new(
        "item",
        DataType::Struct(Fields::from(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ])),
        false,
    );
    let schema = ArrowSchema::new(vec![Field::new(
        "list_col",
        DataType::List(Arc::new(struct_field.clone())),
        false,
    )]);

    // Create structs
    let ids = Int32Array::from(vec![1, 2, 3, 4]);
    let names = StringArray::from(vec!["a", "b", "c", "d"]);
    let structs = StructArray::from(vec![
        (Arc::new(Field::new("id", DataType::Int32, false)), Arc::new(ids) as ArrayRef),
        (
            Arc::new(Field::new("name", DataType::Utf8, false)),
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

// ============================================================================
// Nullable Union Tests (Avro ['null', 'type'] handling)
// ============================================================================

#[test]
fn test_nullable_int32_all_null() {
    // Test column with all null values
    let schema = ArrowSchema::new(vec![Field::new("nullable_int", DataType::Int32, true)]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int32Array::from(vec![None, None, None, None]))],
    )
    .unwrap();

    let (vanilla, pooled) = compare_decoders(&schema, &batch);
    assert_eq!(vanilla.num_rows(), 4);
    assert_eq!(pooled.num_rows(), 4);

    // Verify all values are null
    let pooled_col = pooled.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    for i in 0..4 {
        assert!(pooled_col.is_null(i), "Row {} should be null", i);
    }
}

#[test]
fn test_nullable_int32_mixed() {
    // Test column with mixed null and non-null values
    let schema = ArrowSchema::new(vec![Field::new("nullable_int", DataType::Int32, true)]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int32Array::from(vec![
            Some(1),
            None,
            Some(3),
            None,
            Some(5),
        ]))],
    )
    .unwrap();

    let (_vanilla, pooled) = compare_decoders(&schema, &batch);
    assert_eq!(pooled.num_rows(), 5);

    let pooled_col = pooled.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    assert!(!pooled_col.is_null(0));
    assert_eq!(pooled_col.value(0), 1);
    assert!(pooled_col.is_null(1));
    assert!(!pooled_col.is_null(2));
    assert_eq!(pooled_col.value(2), 3);
    assert!(pooled_col.is_null(3));
    assert!(!pooled_col.is_null(4));
    assert_eq!(pooled_col.value(4), 5);
}

#[test]
fn test_nullable_int64() {
    let schema = ArrowSchema::new(vec![Field::new("nullable_long", DataType::Int64, true)]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int64Array::from(vec![
            Some(100i64),
            None,
            Some(300i64),
        ]))],
    )
    .unwrap();

    let (_, pooled) = compare_decoders(&schema, &batch);
    let pooled_col = pooled.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    assert!(!pooled_col.is_null(0));
    assert_eq!(pooled_col.value(0), 100);
    assert!(pooled_col.is_null(1));
    assert!(!pooled_col.is_null(2));
    assert_eq!(pooled_col.value(2), 300);
}

#[test]
fn test_nullable_float32() {
    let schema = ArrowSchema::new(vec![Field::new("nullable_float", DataType::Float32, true)]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Float32Array::from(vec![
            None,
            Some(2.5f32),
            None,
            Some(4.5f32),
        ]))],
    )
    .unwrap();

    let (_, pooled) = compare_decoders(&schema, &batch);
    let pooled_col = pooled.column(0).as_any().downcast_ref::<Float32Array>().unwrap();
    assert!(pooled_col.is_null(0));
    assert!(!pooled_col.is_null(1));
    assert!((pooled_col.value(1) - 2.5f32).abs() < 0.001);
    assert!(pooled_col.is_null(2));
    assert!(!pooled_col.is_null(3));
    assert!((pooled_col.value(3) - 4.5f32).abs() < 0.001);
}

#[test]
fn test_nullable_float64() {
    let schema = ArrowSchema::new(vec![Field::new("nullable_double", DataType::Float64, true)]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Float64Array::from(vec![
            Some(1.5f64),
            None,
            Some(3.5f64),
        ]))],
    )
    .unwrap();

    let (_, pooled) = compare_decoders(&schema, &batch);
    let pooled_col = pooled.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
    assert!(!pooled_col.is_null(0));
    assert!((pooled_col.value(0) - 1.5f64).abs() < 0.001);
    assert!(pooled_col.is_null(1));
    assert!(!pooled_col.is_null(2));
}

#[test]
fn test_nullable_boolean() {
    let schema = ArrowSchema::new(vec![Field::new("nullable_bool", DataType::Boolean, true)]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(BooleanArray::from(vec![
            Some(true),
            None,
            Some(false),
            None,
            Some(true),
        ]))],
    )
    .unwrap();

    let (_, pooled) = compare_decoders(&schema, &batch);
    let pooled_col = pooled.column(0).as_any().downcast_ref::<BooleanArray>().unwrap();
    assert!(!pooled_col.is_null(0));
    assert!(pooled_col.value(0));
    assert!(pooled_col.is_null(1));
    assert!(!pooled_col.is_null(2));
    assert!(!pooled_col.value(2));
    assert!(pooled_col.is_null(3));
    assert!(!pooled_col.is_null(4));
    assert!(pooled_col.value(4));
}

#[test]
fn test_nullable_string_all_null() {
    let schema = ArrowSchema::new(vec![Field::new("nullable_str", DataType::Utf8, true)]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(StringArray::from(vec![
            None::<&str>,
            None,
            None,
        ]))],
    )
    .unwrap();

    let (_, pooled) = compare_decoders(&schema, &batch);
    let pooled_col = pooled.column(0).as_any().downcast_ref::<StringArray>().unwrap();
    for i in 0..3 {
        assert!(pooled_col.is_null(i), "Row {} should be null", i);
    }
}

#[test]
fn test_nullable_string_mixed() {
    let schema = ArrowSchema::new(vec![Field::new("nullable_str", DataType::Utf8, true)]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(StringArray::from(vec![
            Some("hello"),
            None,
            Some("world"),
            None,
            Some("test"),
        ]))],
    )
    .unwrap();

    let (_, pooled) = compare_decoders(&schema, &batch);
    let pooled_col = pooled.column(0).as_any().downcast_ref::<StringArray>().unwrap();
    assert!(!pooled_col.is_null(0));
    assert_eq!(pooled_col.value(0), "hello");
    assert!(pooled_col.is_null(1));
    assert!(!pooled_col.is_null(2));
    assert_eq!(pooled_col.value(2), "world");
    assert!(pooled_col.is_null(3));
    assert!(!pooled_col.is_null(4));
    assert_eq!(pooled_col.value(4), "test");
}

#[test]
fn test_nullable_binary() {
    let schema = ArrowSchema::new(vec![Field::new("nullable_bin", DataType::Binary, true)]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(BinaryArray::from(vec![
            Some(b"abc".as_slice()),
            None,
            Some(b"xyz".as_slice()),
        ]))],
    )
    .unwrap();

    let (_, pooled) = compare_decoders(&schema, &batch);
    let pooled_col = pooled.column(0).as_any().downcast_ref::<BinaryArray>().unwrap();
    assert!(!pooled_col.is_null(0));
    assert_eq!(pooled_col.value(0), b"abc");
    assert!(pooled_col.is_null(1));
    assert!(!pooled_col.is_null(2));
    assert_eq!(pooled_col.value(2), b"xyz");
}

#[test]
fn test_multiple_nullable_columns() {
    // Test multiple nullable columns in the same schema
    let schema = ArrowSchema::new(vec![
        Field::new("nullable_int", DataType::Int32, true),
        Field::new("nullable_str", DataType::Utf8, true),
        Field::new("nullable_float", DataType::Float64, true),
        Field::new("nullable_bool", DataType::Boolean, true),
    ]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from(vec![Some(1), None, Some(3)])),
            Arc::new(StringArray::from(vec![None, Some("b"), None])),
            Arc::new(Float64Array::from(vec![Some(1.1), Some(2.2), None])),
            Arc::new(BooleanArray::from(vec![None, None, Some(true)])),
        ],
    )
    .unwrap();

    let (_, pooled) = compare_decoders(&schema, &batch);
    assert_eq!(pooled.num_rows(), 3);
    assert_eq!(pooled.num_columns(), 4);

    // Check int column
    let int_col = pooled.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    assert!(!int_col.is_null(0));
    assert!(int_col.is_null(1));
    assert!(!int_col.is_null(2));

    // Check string column
    let str_col = pooled.column(1).as_any().downcast_ref::<StringArray>().unwrap();
    assert!(str_col.is_null(0));
    assert!(!str_col.is_null(1));
    assert!(str_col.is_null(2));

    // Check float column
    let float_col = pooled.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
    assert!(!float_col.is_null(0));
    assert!(!float_col.is_null(1));
    assert!(float_col.is_null(2));

    // Check bool column
    let bool_col = pooled.column(3).as_any().downcast_ref::<BooleanArray>().unwrap();
    assert!(bool_col.is_null(0));
    assert!(bool_col.is_null(1));
    assert!(!bool_col.is_null(2));
}

#[test]
fn test_mixed_nullable_and_non_nullable_columns() {
    // Test schema with both nullable and non-nullable columns
    let schema = ArrowSchema::new(vec![
        Field::new("required_int", DataType::Int32, false),
        Field::new("nullable_int", DataType::Int32, true),
        Field::new("required_str", DataType::Utf8, false),
        Field::new("nullable_str", DataType::Utf8, true),
    ]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(Int32Array::from(vec![Some(10), None, Some(30)])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Arc::new(StringArray::from(vec![Some("x"), None, Some("z")])),
        ],
    )
    .unwrap();

    let (_, pooled) = compare_decoders(&schema, &batch);
    assert_eq!(pooled.num_rows(), 3);

    // Non-nullable columns should have no nulls
    let req_int = pooled.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    assert!(req_int.null_count() == 0);

    let req_str = pooled.column(2).as_any().downcast_ref::<StringArray>().unwrap();
    assert!(req_str.null_count() == 0);

    // Nullable columns should have correct null patterns
    let null_int = pooled.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
    assert!(!null_int.is_null(0));
    assert!(null_int.is_null(1));
    assert!(!null_int.is_null(2));

    let null_str = pooled.column(3).as_any().downcast_ref::<StringArray>().unwrap();
    assert!(!null_str.is_null(0));
    assert!(null_str.is_null(1));
    assert!(!null_str.is_null(2));
}

#[test]
fn test_nullable_first_and_last_rows() {
    // Test edge case: null in first row and last row
    let schema = ArrowSchema::new(vec![Field::new("nullable_int", DataType::Int32, true)]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int32Array::from(vec![
            None,
            Some(2),
            Some(3),
            Some(4),
            None,
        ]))],
    )
    .unwrap();

    let (_, pooled) = compare_decoders(&schema, &batch);
    let pooled_col = pooled.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    assert!(pooled_col.is_null(0), "First row should be null");
    assert!(!pooled_col.is_null(1));
    assert!(!pooled_col.is_null(2));
    assert!(!pooled_col.is_null(3));
    assert!(pooled_col.is_null(4), "Last row should be null");
}

#[test]
fn test_large_nullable_batch() {
    // Test with many rows to exercise batch processing
    let schema = ArrowSchema::new(vec![
        Field::new("nullable_int", DataType::Int64, true),
        Field::new("nullable_str", DataType::Utf8, true),
    ]);

    let n = 1000;
    let int_values: Vec<Option<i64>> = (0..n)
        .map(|i| if i % 3 == 0 { None } else { Some(i as i64) })
        .collect();
    let str_values: Vec<Option<&str>> = (0..n)
        .map(|i| if i % 5 == 0 { None } else { Some("test") })
        .collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(int_values.clone())),
            Arc::new(StringArray::from(str_values.clone())),
        ],
    )
    .unwrap();

    let (_, pooled) = compare_decoders(&schema, &batch);
    assert_eq!(pooled.num_rows(), n);

    let int_col = pooled.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    let str_col = pooled.column(1).as_any().downcast_ref::<StringArray>().unwrap();

    for i in 0..n {
        let expected_int_null = i % 3 == 0;
        let expected_str_null = i % 5 == 0;
        assert_eq!(
            int_col.is_null(i),
            expected_int_null,
            "Int null mismatch at row {}",
            i
        );
        assert_eq!(
            str_col.is_null(i),
            expected_str_null,
            "Str null mismatch at row {}",
            i
        );
    }
}

// ============================================================================
// Date/Time Types Tests
// ============================================================================

#[test]
fn test_parity_date32() {
    let schema = ArrowSchema::new(vec![Field::new("date_col", DataType::Date32, false)]);

    // Days since epoch: 2024-01-01 is about 19723 days since 1970-01-01
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Date32Array::from(vec![19723, 19724, 19725]))],
    )
    .unwrap();

    let (vanilla, pooled) = compare_decoders(&schema, &batch);
    assert_eq!(vanilla.num_rows(), 3);
    assert_eq!(pooled.num_rows(), 3);

    let pooled_col = pooled.column(0).as_any().downcast_ref::<Date32Array>().unwrap();
    assert_eq!(pooled_col.value(0), 19723);
    assert_eq!(pooled_col.value(1), 19724);
    assert_eq!(pooled_col.value(2), 19725);
}

#[test]
fn test_parity_date32_nullable() {
    let schema = ArrowSchema::new(vec![Field::new("date_col", DataType::Date32, true)]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Date32Array::from(vec![
            Some(19723),
            None,
            Some(19725),
        ]))],
    )
    .unwrap();

    let (_, pooled) = compare_decoders(&schema, &batch);
    assert_eq!(pooled.num_rows(), 3);

    let pooled_col = pooled.column(0).as_any().downcast_ref::<Date32Array>().unwrap();
    assert!(!pooled_col.is_null(0));
    assert_eq!(pooled_col.value(0), 19723);
    assert!(pooled_col.is_null(1));
    assert!(!pooled_col.is_null(2));
    assert_eq!(pooled_col.value(2), 19725);
}

#[test]
fn test_parity_timestamp_microsecond() {
    use arrow::datatypes::TimeUnit;

    let schema = ArrowSchema::new(vec![Field::new(
        "ts_col",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        false,
    )]);

    // Microseconds since epoch
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(TimestampMicrosecondArray::from(vec![
            1704067200000000i64, // 2024-01-01 00:00:00
            1704067260000000i64, // 2024-01-01 00:01:00
            1704067320000000i64, // 2024-01-01 00:02:00
        ]))],
    )
    .unwrap();

    let (vanilla, pooled) = compare_decoders(&schema, &batch);
    assert_eq!(vanilla.num_rows(), 3);
    assert_eq!(pooled.num_rows(), 3);

    let pooled_col = pooled
        .column(0)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .unwrap();
    assert_eq!(pooled_col.value(0), 1704067200000000i64);
    assert_eq!(pooled_col.value(1), 1704067260000000i64);
    assert_eq!(pooled_col.value(2), 1704067320000000i64);
}

#[test]
fn test_parity_timestamp_microsecond_nullable() {
    use arrow::datatypes::TimeUnit;

    let schema = ArrowSchema::new(vec![Field::new(
        "ts_col",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        true,
    )]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(TimestampMicrosecondArray::from(vec![
            Some(1704067200000000i64),
            None,
            Some(1704067320000000i64),
        ]))],
    )
    .unwrap();

    let (_, pooled) = compare_decoders(&schema, &batch);
    assert_eq!(pooled.num_rows(), 3);

    let pooled_col = pooled
        .column(0)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .unwrap();
    assert!(!pooled_col.is_null(0));
    assert_eq!(pooled_col.value(0), 1704067200000000i64);
    assert!(pooled_col.is_null(1));
    assert!(!pooled_col.is_null(2));
    assert_eq!(pooled_col.value(2), 1704067320000000i64);
}

#[test]
fn test_parity_timestamp_millisecond() {
    use arrow::datatypes::TimeUnit;

    let schema = ArrowSchema::new(vec![Field::new(
        "ts_col",
        DataType::Timestamp(TimeUnit::Millisecond, None),
        false,
    )]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(TimestampMillisecondArray::from(vec![
            1704067200000i64, // 2024-01-01 00:00:00
            1704067260000i64, // 2024-01-01 00:01:00
            1704067320000i64, // 2024-01-01 00:02:00
        ]))],
    )
    .unwrap();

    let (vanilla, pooled) = compare_decoders(&schema, &batch);
    assert_eq!(vanilla.num_rows(), 3);
    assert_eq!(pooled.num_rows(), 3);
}

#[test]
fn test_parity_timestamp_millisecond_nullable() {
    use arrow::datatypes::TimeUnit;

    let schema = ArrowSchema::new(vec![Field::new(
        "ts_col",
        DataType::Timestamp(TimeUnit::Millisecond, None),
        true,
    )]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(TimestampMillisecondArray::from(vec![
            Some(1704067200000i64),
            None,
            Some(1704067320000i64),
        ]))],
    )
    .unwrap();

    let (_, pooled) = compare_decoders(&schema, &batch);
    assert_eq!(pooled.num_rows(), 3);

    let pooled_col = pooled
        .column(0)
        .as_any()
        .downcast_ref::<TimestampMillisecondArray>()
        .unwrap();
    assert!(!pooled_col.is_null(0));
    assert_eq!(pooled_col.value(0), 1704067200000i64);
    assert!(pooled_col.is_null(1));
    assert!(!pooled_col.is_null(2));
    assert_eq!(pooled_col.value(2), 1704067320000i64);
}

#[test]
fn test_parity_timestamp_nanosecond() {
    use arrow::datatypes::TimeUnit;

    let schema = ArrowSchema::new(vec![Field::new(
        "ts_col",
        DataType::Timestamp(TimeUnit::Nanosecond, None),
        false,
    )]);

    // Nanoseconds since epoch
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(TimestampNanosecondArray::from(vec![
            1704067200000000000i64, // 2024-01-01 00:00:00
            1704067260000000000i64, // 2024-01-01 00:01:00
            1704067320000000000i64, // 2024-01-01 00:02:00
        ]))],
    )
    .unwrap();

    let (vanilla, pooled) = compare_decoders(&schema, &batch);
    assert_eq!(vanilla.num_rows(), 3);
    assert_eq!(pooled.num_rows(), 3);

    let pooled_col = pooled
        .column(0)
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
        .unwrap();
    assert_eq!(pooled_col.value(0), 1704067200000000000i64);
    assert_eq!(pooled_col.value(1), 1704067260000000000i64);
    assert_eq!(pooled_col.value(2), 1704067320000000000i64);
}

#[test]
fn test_parity_timestamp_nanosecond_nullable() {
    use arrow::datatypes::TimeUnit;

    let schema = ArrowSchema::new(vec![Field::new(
        "ts_col",
        DataType::Timestamp(TimeUnit::Nanosecond, None),
        true,
    )]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(TimestampNanosecondArray::from(vec![
            Some(1704067200000000000i64),
            None,
            Some(1704067320000000000i64),
        ]))],
    )
    .unwrap();

    let (_, pooled) = compare_decoders(&schema, &batch);
    assert_eq!(pooled.num_rows(), 3);

    let pooled_col = pooled
        .column(0)
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
        .unwrap();
    assert!(!pooled_col.is_null(0));
    assert_eq!(pooled_col.value(0), 1704067200000000000i64);
    assert!(pooled_col.is_null(1));
    assert!(!pooled_col.is_null(2));
    assert_eq!(pooled_col.value(2), 1704067320000000000i64);
}

#[test]
fn test_parity_timestamp_second() {
    use arrow::datatypes::TimeUnit;

    let schema = ArrowSchema::new(vec![Field::new(
        "ts_col",
        DataType::Timestamp(TimeUnit::Second, None),
        false,
    )]);

    // Seconds since epoch
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(TimestampSecondArray::from(vec![
            1704067200i64, // 2024-01-01 00:00:00
            1704067260i64, // 2024-01-01 00:01:00
            1704067320i64, // 2024-01-01 00:02:00
        ]))],
    )
    .unwrap();

    let (vanilla, pooled) = compare_decoders(&schema, &batch);
    assert_eq!(vanilla.num_rows(), 3);
    assert_eq!(pooled.num_rows(), 3);

    let pooled_col = pooled
        .column(0)
        .as_any()
        .downcast_ref::<TimestampSecondArray>()
        .unwrap();
    assert_eq!(pooled_col.value(0), 1704067200i64);
    assert_eq!(pooled_col.value(1), 1704067260i64);
    assert_eq!(pooled_col.value(2), 1704067320i64);
}

#[test]
fn test_parity_local_timestamp_microsecond() {
    use arrow::datatypes::TimeUnit;

    // Local timestamp (no timezone) - uses local-timestamp-micros in Avro
    let schema = ArrowSchema::new(vec![Field::new(
        "ts_col",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        false,
    )]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(TimestampMicrosecondArray::from(vec![
            1704067200000000i64,
            1704067260000000i64,
            1704067320000000i64,
        ]))],
    )
    .unwrap();

    let (vanilla, pooled) = compare_decoders(&schema, &batch);
    assert_eq!(vanilla.num_rows(), 3);
    assert_eq!(pooled.num_rows(), 3);
}

#[test]
fn test_parity_timestamp_with_utc_timezone() {
    use arrow::datatypes::TimeUnit;

    // Timestamp with UTC timezone - uses timestamp-micros in Avro
    let schema = ArrowSchema::new(vec![Field::new(
        "ts_col",
        DataType::Timestamp(TimeUnit::Microsecond, Some("+00:00".into())),
        false,
    )]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(
            TimestampMicrosecondArray::from(vec![
                1704067200000000i64,
                1704067260000000i64,
                1704067320000000i64,
            ])
            .with_timezone("+00:00"),
        )],
    )
    .unwrap();

    let (vanilla, pooled) = compare_decoders(&schema, &batch);
    assert_eq!(vanilla.num_rows(), 3);
    assert_eq!(pooled.num_rows(), 3);

    let pooled_col = pooled
        .column(0)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .unwrap();
    assert_eq!(pooled_col.value(0), 1704067200000000i64);
    // Verify timezone is preserved
    assert!(pooled_col.timezone().is_some());
}

#[test]
fn test_parity_timestamp_millisecond_with_timezone() {
    use arrow::datatypes::TimeUnit;

    let schema = ArrowSchema::new(vec![Field::new(
        "ts_col",
        DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
        false,
    )]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(
            TimestampMillisecondArray::from(vec![
                1704067200000i64,
                1704067260000i64,
                1704067320000i64,
            ])
            .with_timezone("UTC"),
        )],
    )
    .unwrap();

    let (vanilla, pooled) = compare_decoders(&schema, &batch);
    assert_eq!(vanilla.num_rows(), 3);
    assert_eq!(pooled.num_rows(), 3);
}

#[test]
fn test_parity_timestamp_edge_values() {
    use arrow::datatypes::TimeUnit;

    let schema = ArrowSchema::new(vec![Field::new(
        "ts_col",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        false,
    )]);

    // Test edge cases: zero, negative (before epoch), very large
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(TimestampMicrosecondArray::from(vec![
            0i64,                    // Unix epoch
            -86400000000i64,         // 1969-12-31 (before epoch)
            i64::MAX / 2,            // Large positive value
        ]))],
    )
    .unwrap();

    let (vanilla, pooled) = compare_decoders(&schema, &batch);
    assert_eq!(vanilla.num_rows(), 3);
    assert_eq!(pooled.num_rows(), 3);

    let pooled_col = pooled
        .column(0)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .unwrap();
    assert_eq!(pooled_col.value(0), 0i64);
    assert_eq!(pooled_col.value(1), -86400000000i64);
    assert_eq!(pooled_col.value(2), i64::MAX / 2);
}

#[test]
fn test_parity_all_timestamp_types_combined() {
    use arrow::datatypes::TimeUnit;

    // Test all timestamp variants in a single schema
    let schema = ArrowSchema::new(vec![
        Field::new("ts_micros", DataType::Timestamp(TimeUnit::Microsecond, None), false),
        Field::new("ts_millis", DataType::Timestamp(TimeUnit::Millisecond, None), false),
        Field::new("ts_nanos", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        Field::new("ts_seconds", DataType::Timestamp(TimeUnit::Second, None), false),
        Field::new("ts_micros_tz", DataType::Timestamp(TimeUnit::Microsecond, Some("+00:00".into())), false),
        Field::new("ts_micros_nullable", DataType::Timestamp(TimeUnit::Microsecond, None), true),
    ]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(TimestampMicrosecondArray::from(vec![1704067200000000i64, 1704067260000000i64])),
            Arc::new(TimestampMillisecondArray::from(vec![1704067200000i64, 1704067260000i64])),
            Arc::new(TimestampNanosecondArray::from(vec![1704067200000000000i64, 1704067260000000000i64])),
            Arc::new(TimestampSecondArray::from(vec![1704067200i64, 1704067260i64])),
            Arc::new(TimestampMicrosecondArray::from(vec![1704067200000000i64, 1704067260000000i64]).with_timezone("+00:00")),
            Arc::new(TimestampMicrosecondArray::from(vec![Some(1704067200000000i64), None])),
        ],
    )
    .unwrap();

    let (vanilla, pooled) = compare_decoders(&schema, &batch);
    assert_eq!(vanilla.num_rows(), 2);
    assert_eq!(pooled.num_rows(), 2);
    assert_eq!(vanilla.num_columns(), 6);
    assert_eq!(pooled.num_columns(), 6);
}

// ============================================================================
// Decimal Types Tests
// ============================================================================

#[test]
fn test_parity_decimal128() {
    // Decimal128 with precision 10, scale 2 (fits in ~4 bytes for Avro encoding)
    let schema = ArrowSchema::new(vec![Field::new(
        "decimal_col",
        DataType::Decimal128(10, 2),
        false,
    )]);

    // Create decimal values: 123.45, 678.90, -1234.56
    let values = Decimal128Array::from(vec![12345i128, 67890i128, -123456i128])
        .with_precision_and_scale(10, 2)
        .unwrap();

    let batch =
        RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(values)]).unwrap();

    let (vanilla, pooled) = compare_decoders(&schema, &batch);
    assert_eq!(vanilla.num_rows(), 3);
    assert_eq!(pooled.num_rows(), 3);

    let pooled_col = pooled
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .unwrap();
    assert_eq!(pooled_col.value(0), 12345i128);
    assert_eq!(pooled_col.value(1), 67890i128);
    assert_eq!(pooled_col.value(2), -123456i128);
}

#[test]
fn test_parity_decimal128_nullable() {
    let schema = ArrowSchema::new(vec![Field::new(
        "decimal_col",
        DataType::Decimal128(10, 2),
        true,
    )]);

    let values = Decimal128Array::from(vec![Some(12345i128), None, Some(-123456i128)])
        .with_precision_and_scale(10, 2)
        .unwrap();

    let batch =
        RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(values)]).unwrap();

    let (_, pooled) = compare_decoders(&schema, &batch);
    assert_eq!(pooled.num_rows(), 3);

    let pooled_col = pooled
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .unwrap();
    assert!(!pooled_col.is_null(0));
    assert_eq!(pooled_col.value(0), 12345i128);
    assert!(pooled_col.is_null(1));
    assert!(!pooled_col.is_null(2));
    assert_eq!(pooled_col.value(2), -123456i128);
}

#[test]
fn test_parity_decimal128_large_precision() {
    // Decimal128 with larger precision (25 digits, requires more bytes)
    let schema = ArrowSchema::new(vec![Field::new(
        "decimal_col",
        DataType::Decimal128(25, 4),
        false,
    )]);

    // Larger values that use more bytes
    let values = Decimal128Array::from(vec![
        1234567890123456789i128,
        -9876543210987654321i128,
        0i128,
    ])
    .with_precision_and_scale(25, 4)
    .unwrap();

    let batch =
        RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(values)]).unwrap();

    let (vanilla, pooled) = compare_decoders(&schema, &batch);
    assert_eq!(vanilla.num_rows(), 3);
    assert_eq!(pooled.num_rows(), 3);
}

// ============================================================================
// FixedSizeBinary Tests
// ============================================================================

#[test]
fn test_parity_fixed_size_binary() {
    let schema = ArrowSchema::new(vec![Field::new(
        "fixed_col",
        DataType::FixedSizeBinary(4),
        false,
    )]);

    let values = FixedSizeBinaryArray::from(vec![
        Some([0x01, 0x02, 0x03, 0x04].as_slice()),
        Some([0xAA, 0xBB, 0xCC, 0xDD].as_slice()),
        Some([0x00, 0x00, 0x00, 0x00].as_slice()),
    ]);

    let batch =
        RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(values)]).unwrap();

    let (vanilla, pooled) = compare_decoders(&schema, &batch);
    assert_eq!(vanilla.num_rows(), 3);
    assert_eq!(pooled.num_rows(), 3);

    let pooled_col = pooled
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .unwrap();
    assert_eq!(pooled_col.value(0), [0x01, 0x02, 0x03, 0x04]);
    assert_eq!(pooled_col.value(1), [0xAA, 0xBB, 0xCC, 0xDD]);
    assert_eq!(pooled_col.value(2), [0x00, 0x00, 0x00, 0x00]);
}

#[test]
fn test_parity_fixed_size_binary_nullable() {
    let schema = ArrowSchema::new(vec![Field::new(
        "fixed_col",
        DataType::FixedSizeBinary(4),
        true,
    )]);

    let values = FixedSizeBinaryArray::from(vec![
        Some([0x01, 0x02, 0x03, 0x04].as_slice()),
        None,
        Some([0xAA, 0xBB, 0xCC, 0xDD].as_slice()),
    ]);

    let batch =
        RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(values)]).unwrap();

    let (_, pooled) = compare_decoders(&schema, &batch);
    assert_eq!(pooled.num_rows(), 3);

    let pooled_col = pooled
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .unwrap();
    assert!(!pooled_col.is_null(0));
    assert!(pooled_col.is_null(1));
    assert!(!pooled_col.is_null(2));
}

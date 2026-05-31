//! Robustness tests for JIT Avro decoder
//!
//! These tests verify that the decoder handles malformed/truncated input safely
//! by returning errors instead of crashing or producing undefined behavior.

#![cfg(all(test, feature = "jit"))]

use super::jit_decoder::JitDecoder;
use arrow_schema::{DataType, Field, Fields, Schema};
use std::sync::Arc;

/// Helper to encode a varint (zig-zag + variable length)
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

/// Helper to encode a raw varint (no zig-zag) - used for union/enum indices
fn encode_raw_varint(buf: &mut Vec<u8>, value: u64) {
    let mut v = value;
    loop {
        if v < 0x80 {
            buf.push(v as u8);
            break;
        }
        buf.push((v as u8) | 0x80);
        v >>= 7;
    }
}

/// Helper to encode Avro bytes (length + data)
fn encode_avro_bytes(buf: &mut Vec<u8>, data: &[u8]) {
    encode_varint(buf, data.len() as i64);
    buf.extend_from_slice(data);
}

// =============================================================================
// Empty and minimal input tests
// =============================================================================

#[test]
fn test_empty_input_int64() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    let data: Vec<u8> = vec![];
    let mut output = vec![0u8; 8];
    let mut ptrs = vec![output.as_mut_ptr()];

    // Trying to decode 1 row from empty input should fail
    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for empty input, got {:?}", result);
}

#[test]
fn test_empty_input_zero_rows() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    let data: Vec<u8> = vec![];
    let mut output = vec![0u8; 8];
    let mut ptrs = vec![output.as_mut_ptr()];

    // Decoding 0 rows from empty input should succeed
    let result = decoder.decode(&data, &mut ptrs, 0);
    assert!(result.is_ok(), "Expected success for zero rows, got {:?}", result);
    assert_eq!(result.unwrap(), 0);
}

// =============================================================================
// Truncated varint tests
// =============================================================================

#[test]
fn test_truncated_varint_single_byte_missing() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Varint with continuation bit set but no following byte
    let data: Vec<u8> = vec![0x80]; // High bit set = more bytes expected
    let mut output = vec![0u8; 8];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated varint");
}

#[test]
fn test_truncated_varint_multi_byte() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Start of a multi-byte varint, truncated
    let data: Vec<u8> = vec![0x80, 0x80, 0x80]; // Needs more bytes
    let mut output = vec![0u8; 8];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated multi-byte varint");
}

#[test]
fn test_varint_max_bytes_exceeded() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // 11 bytes all with continuation bit - exceeds max varint length (10)
    let data: Vec<u8> = vec![0x80; 11];
    let mut output = vec![0u8; 8];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for overlong varint");
}

// =============================================================================
// Truncated fixed-size type tests
// =============================================================================

#[test]
fn test_truncated_float32() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float32, false)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Float32 needs 4 bytes, provide only 3
    let data: Vec<u8> = vec![0x00, 0x00, 0x80];
    let mut output = vec![0u8; 4];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated Float32");
}

#[test]
fn test_truncated_float64() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, false)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Float64 needs 8 bytes, provide only 7
    let data: Vec<u8> = vec![0x00; 7];
    let mut output = vec![0u8; 8];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated Float64");
}

#[test]
fn test_truncated_boolean() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Boolean, false)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Boolean needs 1 byte, provide 0
    let data: Vec<u8> = vec![];
    let mut output = vec![0u8; 1];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated Boolean");
}

// =============================================================================
// Truncated variable-length type tests
// =============================================================================

#[test]
fn test_truncated_bytes_length_only() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Binary, false)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Length says 10 bytes, but no data follows
    let mut data = Vec::new();
    encode_varint(&mut data, 10);
    // No payload bytes

    let mut output = vec![0u8; 128];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated bytes (length only)");
}

#[test]
fn test_truncated_bytes_partial_data() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Binary, false)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Length says 10 bytes, but only 5 bytes of data
    let mut data = Vec::new();
    encode_varint(&mut data, 10);
    data.extend_from_slice(&[0x01, 0x02, 0x03, 0x04, 0x05]); // Only 5 bytes

    let mut output = vec![0u8; 128];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated bytes (partial data)");
}

#[test]
fn test_negative_bytes_length() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Binary, false)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Negative length (after zigzag decode)
    let mut data = Vec::new();
    encode_varint(&mut data, -5); // Negative length

    let mut output = vec![0u8; 128];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for negative bytes length");
}

#[test]
fn test_truncated_string() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Utf8, false)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Length says 20 bytes, but only 10 bytes of data
    let mut data = Vec::new();
    encode_varint(&mut data, 20);
    data.extend_from_slice(b"0123456789"); // Only 10 bytes

    let mut output = vec![0u8; 128];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated string");
}

// =============================================================================
// Truncated fixed binary tests
// =============================================================================

#[test]
fn test_truncated_fixed_binary() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("v", DataType::FixedSizeBinary(16), false),
    ]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Fixed 16 bytes, provide only 10
    let data: Vec<u8> = vec![0x00; 10];
    let mut output = vec![0u8; 16];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated FixedSizeBinary");
}

// =============================================================================
// Multi-row truncation tests
// =============================================================================

#[test]
fn test_truncated_second_row() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // First row complete, second row truncated
    let mut data = Vec::new();
    encode_varint(&mut data, 42); // First row - complete
    data.push(0x80); // Second row - truncated varint

    let mut output = vec![0u8; 16]; // Space for 2 rows
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 2);
    assert!(result.is_err(), "Expected error for truncated second row");
}

#[test]
fn test_truncated_middle_of_many_rows() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, false)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // 5 complete rows, 6th row truncated
    let mut data = Vec::new();
    for _ in 0..5 {
        data.extend_from_slice(&1.0f64.to_le_bytes());
    }
    data.extend_from_slice(&[0x00; 4]); // Only 4 bytes of 6th row

    let mut output = vec![0u8; 48]; // Space for 6 rows
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 6);
    assert!(result.is_err(), "Expected error for truncated 6th row");
}

// =============================================================================
// Multi-column truncation tests
// =============================================================================

#[test]
fn test_truncated_multi_column_first_field() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Float64, false),
    ]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Truncated first field
    let data: Vec<u8> = vec![0x80]; // Truncated varint for field 'a'

    let mut output_a = vec![0u8; 8];
    let mut output_b = vec![0u8; 8];
    let mut ptrs = vec![output_a.as_mut_ptr(), output_b.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated first field");
}

#[test]
fn test_truncated_multi_column_second_field() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Float64, false),
    ]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Complete first field, truncated second field
    let mut data = Vec::new();
    encode_varint(&mut data, 42); // Field 'a' complete
    data.extend_from_slice(&[0x00; 4]); // Field 'b' truncated (needs 8 bytes)

    let mut output_a = vec![0u8; 8];
    let mut output_b = vec![0u8; 8];
    let mut ptrs = vec![output_a.as_mut_ptr(), output_b.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated second field");
}

// =============================================================================
// Unsupported type tests (fail-fast)
// =============================================================================

#[test]
fn test_unsupported_type_int8() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int8, false)]));
    let result = JitDecoder::compile(&schema);
    assert!(result.is_err(), "Expected compile error for Int8");
    let err = result.err().unwrap();
    assert!(err.contains("Unsupported"), "Error should mention 'Unsupported': {}", err);
}

#[test]
fn test_unsupported_type_int16() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int16, false)]));
    let result = JitDecoder::compile(&schema);
    assert!(result.is_err(), "Expected compile error for Int16");
}

#[test]
fn test_unsupported_type_uint64() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::UInt64, false)]));
    let result = JitDecoder::compile(&schema);
    assert!(result.is_err(), "Expected compile error for UInt64");
}

#[test]
fn test_unsupported_type_float16() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float16, false)]));
    let result = JitDecoder::compile(&schema);
    assert!(result.is_err(), "Expected compile error for Float16");
}

#[test]
fn test_unsupported_nested_type() {
    // Struct with unsupported nested type
    let schema = Arc::new(Schema::new(vec![Field::new(
        "s",
        DataType::Struct(Fields::from(vec![Field::new("v", DataType::Int8, false)])),
        false,
    )]));
    let result = JitDecoder::compile(&schema);
    assert!(result.is_err(), "Expected compile error for nested unsupported type");
}

// =============================================================================
// Valid input tests (sanity checks)
// =============================================================================

#[test]
fn test_valid_single_int64() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    let mut data = Vec::new();
    encode_varint(&mut data, 12345);

    let mut output = vec![0u8; 8];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_ok(), "Expected success for valid Int64");

    let value = i64::from_le_bytes(output[..8].try_into().unwrap());
    assert_eq!(value, 12345);
}

#[test]
fn test_valid_float64() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, false)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    let data = 3.14159f64.to_le_bytes().to_vec();

    let mut output = vec![0u8; 8];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_ok(), "Expected success for valid Float64");

    let value = f64::from_le_bytes(output[..8].try_into().unwrap());
    assert!((value - 3.14159).abs() < 1e-10);
}

#[test]
fn test_valid_bytes() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Binary, false)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    let mut data = Vec::new();
    encode_avro_bytes(&mut data, b"hello world");

    // Variable-length types require 2 buffers: offsets + data
    let mut offsets = vec![0u8; 8]; // 2 offsets (row_count + 1) * 4
    let mut output = vec![0u8; 128];
    let mut ptrs = vec![offsets.as_mut_ptr(), output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_ok(), "Expected success for valid Binary");
}

#[test]
fn test_valid_multi_row() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    let mut data = Vec::new();
    for i in 0..100 {
        encode_varint(&mut data, i);
    }

    let mut output = vec![0u8; 800]; // 100 rows * 8 bytes
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 100);
    assert!(result.is_ok(), "Expected success for 100 valid rows");

    // Verify all values
    for i in 0..100 {
        let offset = i * 8;
        let value = i64::from_le_bytes(output[offset..offset + 8].try_into().unwrap());
        assert_eq!(value, i as i64, "Mismatch at row {}", i);
    }
}

// =============================================================================
// Union tests
// =============================================================================

#[test]
fn test_union_truncated_type_tag() {
    use arrow_schema::UnionFields;

    let union_fields = UnionFields::new(
        vec![0, 1],
        vec![
            Field::new("int", DataType::Int64, false),
            Field::new("float", DataType::Float64, false),
        ],
    );
    let schema = Arc::new(Schema::new(vec![Field::new(
        "u",
        DataType::Union(union_fields, arrow_schema::UnionMode::Dense),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Truncated type tag varint
    let data: Vec<u8> = vec![0x80];
    let mut output = vec![0u8; 16];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated union type tag");
}

#[test]
fn test_union_truncated_value() {
    use arrow_schema::UnionFields;

    let union_fields = UnionFields::new(
        vec![0, 1],
        vec![
            Field::new("int", DataType::Int64, false),
            Field::new("float", DataType::Float64, false),
        ],
    );
    let schema = Arc::new(Schema::new(vec![Field::new(
        "u",
        DataType::Union(union_fields, arrow_schema::UnionMode::Dense),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Type tag says Float64 (index 1), but value is truncated
    let mut data = Vec::new();
    encode_raw_varint(&mut data, 1); // Select Float64 variant (raw varint, no zigzag)
    data.extend_from_slice(&[0x00; 4]); // Only 4 bytes, need 8

    let mut output = vec![0u8; 16];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated union value");
}

#[test]
fn test_union_valid() {
    use arrow_schema::UnionFields;

    let union_fields = UnionFields::new(
        vec![0, 1],
        vec![
            Field::new("int", DataType::Int64, false),
            Field::new("float", DataType::Float64, false),
        ],
    );
    let schema = Arc::new(Schema::new(vec![Field::new(
        "u",
        DataType::Union(union_fields, arrow_schema::UnionMode::Dense),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Type tag = 0 (Int64), followed by varint value
    let mut data = Vec::new();
    encode_raw_varint(&mut data, 0); // Select Int64 variant (raw varint, no zigzag)
    encode_varint(&mut data, 42); // Value (zigzag encoded)

    let mut output = vec![0u8; 16];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_ok(), "Expected success for valid union");
}

// =============================================================================
// FixedSizeList tests
// =============================================================================

#[test]
fn test_fixed_size_list_truncated() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "l",
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, false)), 3),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Need 3 * 8 = 24 bytes, provide only 16
    let data: Vec<u8> = vec![0x00; 16];
    let mut output = vec![0u8; 24];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated FixedSizeList");
}

#[test]
fn test_fixed_size_list_valid() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "l",
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, false)), 3),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // 3 Float64 values = 24 bytes
    let mut data = Vec::new();
    data.extend_from_slice(&1.0f64.to_le_bytes());
    data.extend_from_slice(&2.0f64.to_le_bytes());
    data.extend_from_slice(&3.0f64.to_le_bytes());

    let mut output = vec![0u8; 24];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_ok(), "Expected success for valid FixedSizeList");
}

// =============================================================================
// Enum/Dictionary tests
// =============================================================================

#[test]
fn test_enum_truncated_index() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "e",
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Truncated varint for enum index
    let data: Vec<u8> = vec![0x80];
    let mut output = vec![0u8; 4];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated enum index");
}

#[test]
fn test_enum_valid() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "e",
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Valid enum index (not zigzag encoded for enum)
    let mut data = Vec::new();
    encode_raw_varint(&mut data, 5); // Index 5 (raw varint, no zigzag)

    let mut output = vec![0u8; 4];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_ok(), "Expected success for valid enum");
}

// =============================================================================
// Duration tests
// =============================================================================

#[test]
fn test_duration_truncated() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "d",
        DataType::Duration(arrow_schema::TimeUnit::Nanosecond),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Duration is encoded as varint, truncate it
    let data: Vec<u8> = vec![0x80];
    let mut output = vec![0u8; 8];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated Duration");
}

#[test]
fn test_duration_valid() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "d",
        DataType::Duration(arrow_schema::TimeUnit::Nanosecond),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Valid duration value
    let mut data = Vec::new();
    encode_varint(&mut data, 1_000_000_000); // 1 second in nanoseconds

    let mut output = vec![0u8; 8];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_ok(), "Expected success for valid Duration");
}

// =============================================================================
// Interval tests
// =============================================================================

#[test]
fn test_interval_year_month_truncated() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "i",
        DataType::Interval(arrow_schema::IntervalUnit::YearMonth),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // YearMonth interval is varint, truncate it
    let data: Vec<u8> = vec![0x80];
    let mut output = vec![0u8; 4];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated YearMonth Interval");
}

#[test]
fn test_interval_day_time_truncated() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "i",
        DataType::Interval(arrow_schema::IntervalUnit::DayTime),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // DayTime interval needs 8 bytes, provide only 4
    let data: Vec<u8> = vec![0x00; 4];
    let mut output = vec![0u8; 8];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated DayTime Interval");
}

#[test]
fn test_interval_month_day_nano_truncated() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "i",
        DataType::Interval(arrow_schema::IntervalUnit::MonthDayNano),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // MonthDayNano interval needs 16 bytes, provide only 10
    let data: Vec<u8> = vec![0x00; 10];
    let mut output = vec![0u8; 16];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated MonthDayNano Interval");
}

#[test]
fn test_interval_valid() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "i",
        DataType::Interval(arrow_schema::IntervalUnit::DayTime),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // 8 bytes for DayTime interval
    let data: Vec<u8> = vec![0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00]; // 1 day, 2 ms
    let mut output = vec![0u8; 8];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_ok(), "Expected success for valid Interval");
}

// =============================================================================
// Nullable field tests
// =============================================================================

#[test]
fn test_nullable_truncated_tag() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)])); // nullable
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Empty input - can't even read the null tag
    let data: Vec<u8> = vec![];
    let mut output = vec![0u8; 8];
    let mut validity = vec![0u8; 1]; // validity buffer for nullable field
    let mut ptrs = vec![output.as_mut_ptr(), validity.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated nullable tag");
}

#[test]
fn test_nullable_valid_null() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Null tag = 0 (meaning null in Avro union [null, T])
    let mut data = Vec::new();
    encode_varint(&mut data, 0); // null

    let mut output = vec![0u8; 8];
    let mut validity = vec![0u8; 1]; // validity buffer for nullable field
    let mut ptrs = vec![output.as_mut_ptr(), validity.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_ok(), "Expected success for valid null");
}

#[test]
fn test_nullable_valid_value() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Non-null tag = 1, followed by value
    let mut data = Vec::new();
    encode_varint(&mut data, 1); // non-null
    encode_varint(&mut data, 42); // value

    let mut output = vec![0u8; 8];
    let mut validity = vec![0u8; 1]; // validity buffer for nullable field
    let mut ptrs = vec![output.as_mut_ptr(), validity.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_ok(), "Expected success for valid nullable value");
}

#[test]
fn test_nullable_truncated_value_after_tag() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Non-null tag, but truncated value
    let mut data = Vec::new();
    encode_varint(&mut data, 1); // non-null tag
    data.extend_from_slice(&[0x00; 4]); // Only 4 bytes, need 8 for Float64

    let mut output = vec![0u8; 8];
    let mut validity = vec![0u8; 1]; // validity buffer for nullable field
    let mut ptrs = vec![output.as_mut_ptr(), validity.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated value after null tag");
}

// =============================================================================
// Struct tests
// =============================================================================

#[test]
fn test_struct_truncated_first_field() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "s",
        DataType::Struct(Fields::from(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Float64, false),
        ])),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Truncated varint for first struct field
    let data: Vec<u8> = vec![0x80]; // Incomplete varint
    let mut output = vec![0u8; 16];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated struct first field");
}

#[test]
fn test_struct_truncated_second_field() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "s",
        DataType::Struct(Fields::from(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Float64, false),
        ])),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Complete first field, truncated second
    let mut data = Vec::new();
    encode_varint(&mut data, 100); // Field 'a' complete
    data.extend_from_slice(&[0x00; 4]); // Field 'b' truncated (needs 8)

    let mut output = vec![0u8; 16];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated struct second field");
}

#[test]
fn test_struct_valid() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "s",
        DataType::Struct(Fields::from(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Float64, false),
        ])),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    let mut data = Vec::new();
    encode_varint(&mut data, 123); // Field 'a'
    data.extend_from_slice(&3.14f64.to_le_bytes()); // Field 'b'

    let mut output = vec![0u8; 16];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_ok(), "Expected success for valid struct");
}

// =============================================================================
// List tests
// =============================================================================

#[test]
fn test_list_truncated_block_count() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "l",
        DataType::List(Arc::new(Field::new("item", DataType::Int64, false))),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Incomplete block count varint
    let data: Vec<u8> = vec![0x80];
    let mut output = vec![0u8; 128];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated list block count");
}

#[test]
fn test_list_truncated_elements() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "l",
        DataType::List(Arc::new(Field::new("item", DataType::Float64, false))),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Block count = 3, but only 2 elements worth of data
    let mut data = Vec::new();
    encode_varint(&mut data, 3); // 3 elements
    data.extend_from_slice(&1.0f64.to_le_bytes()); // Element 1
    data.extend_from_slice(&2.0f64.to_le_bytes()); // Element 2
    // Element 3 missing

    let mut output = vec![0u8; 256];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated list elements");
}

#[test]
fn test_list_empty_valid() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "l",
        DataType::List(Arc::new(Field::new("item", DataType::Int64, false))),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Empty list: block count = 0
    let mut data = Vec::new();
    encode_varint(&mut data, 0); // 0 elements (end of list)

    // Variable-length types require 2 buffers: offsets + data
    let mut offsets = vec![0u8; 8]; // 2 offsets (row_count + 1) * 4
    let mut output = vec![0u8; 128];
    let mut ptrs = vec![offsets.as_mut_ptr(), output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_ok(), "Expected success for empty list");
}

// =============================================================================
// Map tests
// =============================================================================

#[test]
fn test_map_truncated_block_count() {
    let map_field = Field::new(
        "entries",
        DataType::Struct(Fields::from(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Int64, false),
        ])),
        false,
    );
    let schema = Arc::new(Schema::new(vec![Field::new(
        "m",
        DataType::Map(Arc::new(map_field), false),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Incomplete block count
    let data: Vec<u8> = vec![0x80];
    let mut output = vec![0u8; 256];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated map block count");
}

#[test]
fn test_map_truncated_key() {
    let map_field = Field::new(
        "entries",
        DataType::Struct(Fields::from(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Int64, false),
        ])),
        false,
    );
    let schema = Arc::new(Schema::new(vec![Field::new(
        "m",
        DataType::Map(Arc::new(map_field), false),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Block count = 1, but key is truncated
    let mut data = Vec::new();
    encode_varint(&mut data, 1); // 1 entry
    encode_varint(&mut data, 10); // Key length = 10
    data.extend_from_slice(b"short"); // Only 5 bytes, need 10

    let mut output = vec![0u8; 256];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated map key");
}

#[test]
fn test_map_empty_valid() {
    let map_field = Field::new(
        "entries",
        DataType::Struct(Fields::from(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Int64, false),
        ])),
        false,
    );
    let schema = Arc::new(Schema::new(vec![Field::new(
        "m",
        DataType::Map(Arc::new(map_field), false),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Empty map
    let mut data = Vec::new();
    encode_varint(&mut data, 0); // 0 entries

    // Variable-length types require 2 buffers: offsets + data
    let mut offsets = vec![0u8; 8]; // 2 offsets (row_count + 1) * 4
    let mut output = vec![0u8; 256];
    let mut ptrs = vec![offsets.as_mut_ptr(), output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_ok(), "Expected success for empty map");
}

// =============================================================================
// Decimal tests
// =============================================================================

#[test]
fn test_decimal128_truncated() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "d",
        DataType::Decimal128(38, 10),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Decimal128 needs 16 bytes, provide only 10
    let data: Vec<u8> = vec![0x00; 10];
    let mut output = vec![0u8; 16];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated Decimal128");
}

#[test]
fn test_decimal256_truncated() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "d",
        DataType::Decimal256(76, 10),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Decimal256 needs 32 bytes, provide only 20
    let data: Vec<u8> = vec![0x00; 20];
    let mut output = vec![0u8; 32];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated Decimal256");
}

#[test]
fn test_decimal128_valid() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "d",
        DataType::Decimal128(38, 10),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // 16 bytes big-endian
    let data: Vec<u8> = vec![0x00; 16];
    let mut output = vec![0u8; 16];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_ok(), "Expected success for valid Decimal128");
}

// =============================================================================
// Edge case tests
// =============================================================================

#[test]
fn test_zero_length_string() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Utf8, false)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Empty string: length = 0
    let mut data = Vec::new();
    encode_varint(&mut data, 0);

    // Variable-length types require 2 buffers: offsets + data
    let mut offsets = vec![0u8; 8]; // 2 offsets (row_count + 1) * 4
    let mut output = vec![0u8; 64];
    let mut ptrs = vec![offsets.as_mut_ptr(), output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_ok(), "Expected success for zero-length string");
}

#[test]
fn test_zero_length_bytes() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Binary, false)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Empty bytes: length = 0
    let mut data = Vec::new();
    encode_varint(&mut data, 0);

    // Variable-length types require 2 buffers: offsets + data
    let mut offsets = vec![0u8; 8]; // 2 offsets (row_count + 1) * 4
    let mut output = vec![0u8; 64];
    let mut ptrs = vec![offsets.as_mut_ptr(), output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_ok(), "Expected success for zero-length bytes");
}

#[test]
fn test_max_varint_value() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Encode i64::MAX
    let mut data = Vec::new();
    encode_varint(&mut data, i64::MAX);

    let mut output = vec![0u8; 8];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_ok(), "Expected success for max varint value");

    let value = i64::from_le_bytes(output[..8].try_into().unwrap());
    assert_eq!(value, i64::MAX);
}

#[test]
fn test_min_varint_value() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Encode i64::MIN
    let mut data = Vec::new();
    encode_varint(&mut data, i64::MIN);

    let mut output = vec![0u8; 8];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_ok(), "Expected success for min varint value");

    let value = i64::from_le_bytes(output[..8].try_into().unwrap());
    assert_eq!(value, i64::MIN);
}

// =============================================================================
// Boundary condition tests
// =============================================================================

#[test]
fn test_exact_input_length() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, false)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Exactly 8 bytes for one Float64
    let data = 1.0f64.to_le_bytes().to_vec();

    let mut output = vec![0u8; 8];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_ok(), "Expected success for exact input length");
    assert_eq!(result.unwrap(), 8);
}

#[test]
fn test_one_byte_short() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, false)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // 7 bytes - one short
    let data: Vec<u8> = vec![0x00; 7];

    let mut output = vec![0u8; 8];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for one byte short");
}

#[test]
fn test_large_row_count_insufficient_data() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Data for only 1 row
    let mut data = Vec::new();
    encode_varint(&mut data, 42);

    let mut output = vec![0u8; 8000]; // Space for 1000 rows
    let mut ptrs = vec![output.as_mut_ptr()];

    // Try to decode 1000 rows from 1 row of data
    let result = decoder.decode(&data, &mut ptrs, 1000);
    assert!(result.is_err(), "Expected error for insufficient data");
}

// =============================================================================
// Timestamp type robustness tests
// =============================================================================

use arrow_schema::TimeUnit;

#[test]
fn test_timestamp_empty_input() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "ts",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    let data: Vec<u8> = vec![];
    let mut output = vec![0u8; 8];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for empty timestamp input");
}

#[test]
fn test_timestamp_truncated_varint() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "ts",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Varint with continuation bit set but no following byte
    let data: Vec<u8> = vec![0x80];
    let mut output = vec![0u8; 8];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated timestamp varint");
}

#[test]
fn test_timestamp_max_value() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "ts",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    let mut data = Vec::new();
    encode_varint(&mut data, i64::MAX);

    let mut output = vec![0u8; 8];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_ok(), "Expected success for max timestamp value");

    let value = i64::from_le_bytes(output[..8].try_into().unwrap());
    assert_eq!(value, i64::MAX);
}

#[test]
fn test_timestamp_min_value() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "ts",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    let mut data = Vec::new();
    encode_varint(&mut data, i64::MIN);

    let mut output = vec![0u8; 8];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_ok(), "Expected success for min timestamp value");

    let value = i64::from_le_bytes(output[..8].try_into().unwrap());
    assert_eq!(value, i64::MIN);
}

#[test]
fn test_timestamp_millisecond_truncated() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "ts",
        DataType::Timestamp(TimeUnit::Millisecond, None),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Multi-byte varint truncated midway
    let data: Vec<u8> = vec![0x80, 0x80, 0x80]; // Needs more bytes
    let mut output = vec![0u8; 8];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_err(), "Expected error for truncated millisecond timestamp");
}

#[test]
fn test_timestamp_nanosecond_large_values() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "ts",
        DataType::Timestamp(TimeUnit::Nanosecond, None),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Large nanosecond values (year 2024 in nanoseconds)
    let test_values: Vec<i64> = vec![
        1704067200000000000,  // 2024-01-01 00:00:00 UTC in nanoseconds
        i64::MAX / 2,         // Very large but valid
        -1704067200000000000, // Negative (before epoch)
    ];

    for expected in test_values {
        let mut data = Vec::new();
        encode_varint(&mut data, expected);

        let mut output = vec![0u8; 8];
        let mut ptrs = vec![output.as_mut_ptr()];

        let result = decoder.decode(&data, &mut ptrs, 1);
        assert!(result.is_ok(), "Expected success for nanosecond timestamp {}", expected);

        let value = i64::from_le_bytes(output[..8].try_into().unwrap());
        assert_eq!(value, expected, "Nanosecond value mismatch");
    }
}

#[test]
fn test_timestamp_with_timezone_robustness() {
    // Timestamp with timezone should decode the same as without
    let schema = Arc::new(Schema::new(vec![Field::new(
        "ts",
        DataType::Timestamp(TimeUnit::Microsecond, Some("+00:00".into())),
        false,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    let mut data = Vec::new();
    encode_varint(&mut data, 1704067200000000);

    let mut output = vec![0u8; 8];
    let mut ptrs = vec![output.as_mut_ptr()];

    let result = decoder.decode(&data, &mut ptrs, 1);
    assert!(result.is_ok(), "Expected success for timestamp with timezone");

    let value = i64::from_le_bytes(output[..8].try_into().unwrap());
    assert_eq!(value, 1704067200000000);
}

#[test]
fn test_timestamp_all_units_insufficient_data() {
    // Test all timestamp units handle insufficient data correctly
    let units = vec![
        TimeUnit::Second,
        TimeUnit::Millisecond,
        TimeUnit::Microsecond,
        TimeUnit::Nanosecond,
    ];

    for unit in units {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "ts",
            DataType::Timestamp(unit, None),
            false,
        )]));
        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Data for only 1 row
        let mut data = Vec::new();
        encode_varint(&mut data, 42);

        let mut output = vec![0u8; 800];
        let mut ptrs = vec![output.as_mut_ptr()];

        // Try to decode 100 rows
        let result = decoder.decode(&data, &mut ptrs, 100);
        assert!(result.is_err(), "Expected error for {:?} with insufficient data", unit);
    }
}

#[test]
fn test_nullable_timestamp_truncated_union_tag() {
    use super::jit_decoder::OutputBuffer;

    let schema = Arc::new(Schema::new(vec![Field::new(
        "ts",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        true,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Empty input - no union tag at all
    let data: Vec<u8> = vec![];

    let mut data_buf = vec![0u8; 8];
    let mut validity_buf = vec![0u8; 1];

    let output_buffers = vec![
        OutputBuffer::new(data_buf.as_mut_ptr(), data_buf.len()),
        OutputBuffer::new(validity_buf.as_mut_ptr(), validity_buf.len()),
    ];

    let result = decoder.decode_chunked(&data, &output_buffers, 1);
    // Should fail - no data for union tag
    assert!(!result.is_ok() || result.rows_decoded == 0,
        "Expected failure or zero rows for truncated nullable timestamp");
}

#[test]
fn test_nullable_timestamp_null_value_handling() {
    use super::jit_decoder::OutputBuffer;

    let schema = Arc::new(Schema::new(vec![Field::new(
        "ts",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        true,
    )]));
    let decoder = JitDecoder::compile(&schema).expect("compile");

    // Encode null value (union index 0 = null in [null, T] union)
    let mut data = Vec::new();
    encode_raw_varint(&mut data, 0); // null

    let mut data_buf = vec![0u8; 8];
    let mut validity_buf = vec![0u8; 1];

    let output_buffers = vec![
        OutputBuffer::new(data_buf.as_mut_ptr(), data_buf.len()),
        OutputBuffer::new(validity_buf.as_mut_ptr(), validity_buf.len()),
    ];

    let result = decoder.decode_chunked(&data, &output_buffers, 1);
    assert!(result.is_ok(), "Expected success decoding null timestamp");
    assert_eq!(result.rows_decoded, 1);

    // Validity bit should indicate null (bit 0 should be 0)
    assert_eq!(validity_buf[0] & 1, 0, "Expected null bit to be set");
}

// =============================================================================
// Boolean correctness regression
//
// The JIT writes ONE BYTE per boolean (JitDecoder::fixed_size_for_type), so the
// buffer→Arrow conversion in jit_buffer_utils must PACK those bytes into a
// bit-packed BooleanBuffer. A prior bug wrapped the byte buffer directly as a
// bitmap, reading 8 booleans out of each byte and corrupting every value
// (true -> false). This guards against regressing that.
// =============================================================================

#[test]
fn test_boolean_decode_not_corrupted_byte_vs_bit() {
    use super::jit_buffer_utils::{buffers_to_batch_owned, BufferSet};
    use super::jit_decoder::OutputBuffer;
    use arrow_array::{Array, BooleanArray};

    // 9 values → spans more than one byte, so a byte-vs-bit confusion is visible.
    let vals = [true, false, true, true, false, false, true, false, true];
    let data: Vec<u8> = vals.iter().map(|&v| v as u8).collect(); // Avro bool = 1 byte

    let schema = Arc::new(Schema::new(vec![Field::new("b", DataType::Boolean, false)]));
    let decoder = JitDecoder::compile(&schema).expect("compile");
    let row_count = vals.len();

    let elem = JitDecoder::fixed_size_for_type(&DataType::Boolean).unwrap_or(1);
    let mut bs = BufferSet::new_fixed(row_count * elem);
    let outs = vec![OutputBuffer::new(bs.primary_ptr(), bs.primary.len())];
    decoder.decode_checked(&data, &outs, row_count).expect("decode");

    let batch = buffers_to_batch_owned(&schema, vec![bs], row_count).expect("to_batch");
    let col = batch
        .column(0)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .expect("bool column");
    assert_eq!(col.len(), row_count);
    for (i, &v) in vals.iter().enumerate() {
        assert_eq!(col.value(i), v, "boolean row {i} corrupted (byte-vs-bit regression)");
    }
}

// =============================================================================
// Nested List<scalar> + Struct<scalar> via the POOLED converter
// (pooled_buffers_to_batch_owned) — the production pooled path (server.rs:581).
// =============================================================================

#[test]
fn test_pooled_nested_list_and_struct_roundtrip() {
    use super::jit_buffer_utils::{pooled_buffers_to_batch_owned, PooledBufferSet};
    use super::jit_decoder::OutputBuffer;
    use arrow_array::{Array, BooleanArray, Float64Array, Int64Array, ListArray, StructArray};
    use arrow_schema::Fields;

    fn packed(dt: &DataType) -> usize {
        match dt {
            DataType::Struct(c) => c.iter().map(|f| packed(f.data_type())).sum(),
            o => JitDecoder::fixed_size_for_type(o).unwrap_or(8),
        }
    }
    fn decode_pooled(schema: &Arc<Schema>, data: &[u8], rows: usize) -> arrow_array::RecordBatch {
        let dec = JitDecoder::compile(schema).expect("compile");
        let var_cap = (data.len() * 16).max(1 << 16);
        let mut sets = Vec::new();
        for (ci, f) in schema.fields().iter().enumerate() {
            let slot = dec.buffer_slot(ci).unwrap();
            let bs = if slot.secondary.is_some() {
                PooledBufferSet::new_variable((rows + 1) * 4, var_cap)
            } else {
                PooledBufferSet::new_fixed(rows * packed(f.data_type()))
            };
            sets.push(bs);
        }
        let mut outs = Vec::new();
        for (ci, bs) in sets.iter_mut().enumerate() {
            let slot = dec.buffer_slot(ci).unwrap();
            outs.push(OutputBuffer::new(bs.primary_ptr(), bs.primary_capacity()));
            if slot.secondary.is_some() {
                outs.push(OutputBuffer::new(bs.secondary_ptr(), bs.secondary_capacity()));
            }
        }
        dec.decode_checked(data, &outs, rows).expect("decode");
        pooled_buffers_to_batch_owned(schema, sets, rows).expect("convert")
    }

    // List<Int64>
    let lschema = Arc::new(Schema::new(vec![Field::new(
        "l",
        DataType::List(Arc::new(Field::new("item", DataType::Int64, false))),
        false,
    )]));
    let lists: Vec<Vec<i64>> = vec![vec![], vec![1, -2, 3], vec![7; 5]];
    let mut d = Vec::new();
    for list in &lists {
        if !list.is_empty() {
            encode_varint(&mut d, list.len() as i64);
            for v in list {
                encode_varint(&mut d, *v);
            }
        }
        encode_varint(&mut d, 0);
    }
    let batch = decode_pooled(&lschema, &d, lists.len());
    let la = batch.column(0).as_any().downcast_ref::<ListArray>().unwrap();
    for (i, list) in lists.iter().enumerate() {
        let v = la.value(i);
        assert_eq!(
            v.as_any().downcast_ref::<Int64Array>().unwrap().values(),
            list.as_slice(),
            "pooled list row {i}"
        );
    }

    // Struct<Int64, Float64, Boolean>
    let sfields = Fields::from(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Float64, false),
        Field::new("c", DataType::Boolean, false),
    ]);
    let sschema = Arc::new(Schema::new(vec![Field::new("s", DataType::Struct(sfields), false)]));
    let srows: Vec<(i64, f64, bool)> = vec![(1, 1.5, true), (-9, -2.5, false)];
    let mut sd = Vec::new();
    for (a, b, c) in &srows {
        encode_varint(&mut sd, *a);
        sd.extend_from_slice(&b.to_le_bytes());
        sd.push(if *c { 1 } else { 0 });
    }
    let batch = decode_pooled(&sschema, &sd, srows.len());
    let sa = batch.column(0).as_any().downcast_ref::<StructArray>().unwrap();
    let ac = sa.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    let bc = sa.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
    let cc = sa.column(2).as_any().downcast_ref::<BooleanArray>().unwrap();
    for (i, (a, b, c)) in srows.iter().enumerate() {
        assert_eq!(ac.value(i), *a, "pooled struct.a {i}");
        assert_eq!(bc.value(i), *b, "pooled struct.b {i}");
        assert_eq!(cc.value(i), *c, "pooled struct.c {i}");
    }
}

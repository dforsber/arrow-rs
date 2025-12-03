// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Pooled Avro decoder for zero-allocation decoding into pre-allocated buffers
//!
//! This module provides [`PooledRecordDecoder`] which decodes Avro data directly
//! into pre-allocated memory buffers, eliminating memory allocation during the
//! decoding hot path.
//!
//! ## Partial Row Rollback
//!
//! When decoding complex nested types, the buffer may run out of capacity mid-row.
//! The decoder implements a checkpoint mechanism:
//! 1. Save checkpoint before each row (cursor position + column positions)
//! 2. On capacity exceeded → restore checkpoint
//! 3. Flush complete rows only
//! 4. Re-decode the incomplete row into fresh buffer

use crate::buffer_provider::BufferProvider;
use crate::codec::{AvroDataType, Codec};
use super::cursor::AvroCursor;
use super::record::Decoder as ApacheDecoder;
use arrow_array::*;
use arrow_buffer::{i256, Buffer, IntervalMonthDayNano, NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{ArrowError, DataType, Field, FieldRef, Fields, IntervalUnit, Schema, SchemaRef, UnionFields, UnionMode};
use std::cmp::Ordering;
use std::sync::Arc;

/// Error indicating a row decode could not complete due to capacity limits
#[derive(Debug)]
pub struct CapacityExceededError {
    /// Column that exceeded capacity
    pub column_index: usize,
    /// Bytes that would have been written
    pub bytes_needed: usize,
    /// Bytes available
    pub bytes_available: usize,
}

impl std::fmt::Display for CapacityExceededError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Capacity exceeded in column {}: needed {} bytes, only {} available",
            self.column_index, self.bytes_needed, self.bytes_available
        )
    }
}

/// Checkpoint state for partial row rollback
#[derive(Debug, Clone)]
struct RowCheckpoint {
    /// Cursor position at row start (reserved for future cursor rollback)
    #[allow(dead_code)]
    cursor_position: usize,
    /// Per-column data positions at row start
    data_positions: Vec<usize>,
    /// Per-column row counts at row start
    row_counts: Vec<usize>,
    /// Per-column nested state snapshots (simplified - just counts/offsets)
    nested_snapshots: Vec<NestedSnapshot>,
}

/// Snapshot of nested state for rollback
#[derive(Debug, Clone)]
enum NestedSnapshot {
    None,
    Map {
        map_offsets_len: usize,
        key_offsets_len: usize,
        key_data_len: usize,
        value_offsets_len: usize,
        value_data_len: usize,
        value_validity_len: usize,
        total_entries: usize,
    },
    Union {
        type_ids_len: usize,
        offsets_len: Option<usize>,
        child_data_lens: Vec<usize>,
        child_offsets_lens: Vec<usize>,
        child_counts: Vec<usize>,
    },
    List {
        offsets_len: usize,
        child_data_len: usize,
        child_validity_len: usize,
        child_data_position: usize,
        child_offsets_position: usize,
        total_elements: usize,
    },
    Dictionary {
        indices_len: usize,
    },
    Struct {
        field_data_lens: Vec<usize>,
        field_null_bitmap_lens: Vec<usize>,
        field_validity_lens: Vec<usize>,
        row_count: usize,
        field_positions: Vec<usize>,
        field_offsets_positions: Vec<usize>,
        /// Lengths for typed storage (row counts for each field)
        typed_storage_lens: Vec<usize>,
    },
}

/// Decode big-endian two's complement bytes to i128
fn decode_decimal_bytes_to_i128(bytes: &[u8]) -> i128 {
    if bytes.is_empty() {
        return 0;
    }
    // Sign extend: if high bit is set, value is negative
    let is_negative = bytes[0] & 0x80 != 0;
    let mut result = if is_negative { -1i128 } else { 0i128 };

    for &b in bytes {
        result = (result << 8) | (b as i128);
    }
    result
}

/// Decode big-endian two's complement bytes to i256 (as [u8; 32] in little-endian)
fn decode_decimal_bytes_to_i256(bytes: &[u8]) -> [u8; 32] {
    let mut result = [0u8; 32];
    if bytes.is_empty() {
        return result;
    }

    // Sign extend: if high bit is set, fill with 0xFF
    let fill_byte = if bytes[0] & 0x80 != 0 { 0xFF } else { 0x00 };
    result.fill(fill_byte);

    // Copy bytes in reverse order (big-endian to little-endian)
    for (i, &b) in bytes.iter().rev().enumerate() {
        if i < 32 {
            result[i] = b;
        }
    }
    result
}

/// Function pointer type for decoding a field to a slice
/// Returns bytes written on success
type DecodeToSliceFn = fn(&mut AvroCursor<'_>, &mut [u8]) -> Result<usize, ArrowError>;

/// Pre-compiled struct field specification for zero-overhead decoding
#[derive(Clone)]
struct StructFieldSpec {
    /// Fixed size of the field (0 for variable-length)
    size: usize,
    /// Decode function pointer (avoids enum match overhead)
    decode_fn: DecodeToSliceFn,
}

impl std::fmt::Debug for StructFieldSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StructFieldSpec")
            .field("size", &self.size)
            .field("decode_fn", &"<fn>")
            .finish()
    }
}

// Decode functions for each type - no match overhead, direct inline
#[inline(always)]
fn decode_int32_slice(cursor: &mut AvroCursor<'_>, output: &mut [u8]) -> Result<usize, ArrowError> {
    let value = cursor.get_int()?;
    output[..4].copy_from_slice(&value.to_le_bytes());
    Ok(4)
}

#[inline(always)]
fn decode_int64_slice(cursor: &mut AvroCursor<'_>, output: &mut [u8]) -> Result<usize, ArrowError> {
    let value = cursor.get_long()?;
    output[..8].copy_from_slice(&value.to_le_bytes());
    Ok(8)
}

#[inline(always)]
fn decode_float32_slice(cursor: &mut AvroCursor<'_>, output: &mut [u8]) -> Result<usize, ArrowError> {
    let value = cursor.get_float()?;
    output[..4].copy_from_slice(&value.to_le_bytes());
    Ok(4)
}

#[inline(always)]
fn decode_float64_slice(cursor: &mut AvroCursor<'_>, output: &mut [u8]) -> Result<usize, ArrowError> {
    let value = cursor.get_double()?;
    output[..8].copy_from_slice(&value.to_le_bytes());
    Ok(8)
}

#[inline(always)]
fn decode_boolean_slice(cursor: &mut AvroCursor<'_>, output: &mut [u8]) -> Result<usize, ArrowError> {
    let value = cursor.get_bool()?;
    output[0] = if value { 1 } else { 0 };
    Ok(1)
}

#[inline(always)]
fn decode_null_slice(_cursor: &mut AvroCursor<'_>, _output: &mut [u8]) -> Result<usize, ArrowError> {
    Ok(0)
}

impl StructFieldSpec {
    fn from_field_decoder(decoder: &FieldDecoder) -> Option<Self> {
        match decoder {
            FieldDecoder::Null => Some(Self { size: 0, decode_fn: decode_null_slice }),
            FieldDecoder::Boolean => Some(Self { size: 1, decode_fn: decode_boolean_slice }),
            FieldDecoder::Int32 => Some(Self { size: 4, decode_fn: decode_int32_slice }),
            FieldDecoder::Int64 => Some(Self { size: 8, decode_fn: decode_int64_slice }),
            FieldDecoder::Float32 => Some(Self { size: 4, decode_fn: decode_float32_slice }),
            FieldDecoder::Float64 => Some(Self { size: 8, decode_fn: decode_float64_slice }),
            // Variable-length and complex types not supported in fast path
            _ => None,
        }
    }
}

/// Pre-compiled field decoder for zero runtime type matching
///
/// Instead of runtime `match data_type` on every value, we build a tree of
/// typed decoders at construction time. Each variant knows exactly how to
/// decode its type, eliminating the overhead of runtime type dispatch.
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum FieldDecoder {
    // Primitives - each knows exact read/write size
    Null,
    Boolean,
    Int32,
    Int64,
    Float32,
    Float64,

    // Variable length
    String,
    Binary,
    FixedBytes { size: usize },

    // Decimals
    Decimal128,
    Decimal256,

    // Interval
    Interval,

    // Nested types with pre-compiled children
    Struct {
        fields: Vec<FieldDecoder>,
        nullable_flags: Vec<bool>,
    },

    List {
        child: Box<FieldDecoder>,
        child_nullable: bool,
    },

    Map {
        value: Box<FieldDecoder>,
        value_nullable: bool,
    },

    Union {
        variants: Vec<FieldDecoder>,
        type_ids: Vec<i8>,
    },
}

/// Typed storage for struct fields - uses typed Vecs like Apache's Decoder
/// This avoids the overhead of byte conversion (to_le_bytes + extend_from_slice)
/// by using direct typed push operations (Vec<i32>.push(value))
#[derive(Debug)]
enum TypedStorage {
    Null(usize),
    Boolean(Vec<u8>), // Apache uses BooleanBufferBuilder, we use bytes for simplicity
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
    /// String/Binary with offsets and data
    String { offsets: Vec<i32>, data: Vec<u8> },
    Binary { offsets: Vec<i32>, data: Vec<u8> },
    /// Fixed-size binary
    Fixed { size: usize, data: Vec<u8> },
    /// Decimal128 as i128 values
    Decimal128(Vec<i128>),
    /// Interval as raw bytes (16 bytes per value)
    Interval(Vec<[u8; 16]>),
    /// Fallback to raw bytes for unsupported types
    Bytes(Vec<u8>),
}

#[allow(dead_code)]
impl TypedStorage {
    fn from_data_type(data_type: &DataType, capacity: usize) -> Self {
        match data_type {
            DataType::Null => Self::Null(0),
            DataType::Boolean => Self::Boolean(Vec::with_capacity(capacity)),
            DataType::Int32 | DataType::Date32 | DataType::Time32(_) => {
                Self::Int32(Vec::with_capacity(capacity))
            }
            DataType::Int64 | DataType::Time64(_) | DataType::Timestamp(_, _) => {
                Self::Int64(Vec::with_capacity(capacity))
            }
            DataType::Float32 => Self::Float32(Vec::with_capacity(capacity)),
            DataType::Float64 => Self::Float64(Vec::with_capacity(capacity)),
            DataType::Utf8 => Self::String {
                offsets: Vec::with_capacity(capacity + 1),
                data: Vec::with_capacity(capacity * 32),
            },
            DataType::Binary => Self::Binary {
                offsets: Vec::with_capacity(capacity + 1),
                data: Vec::with_capacity(capacity * 32),
            },
            DataType::FixedSizeBinary(size) => Self::Fixed {
                size: *size as usize,
                data: Vec::with_capacity(capacity * (*size as usize)),
            },
            DataType::Decimal128(_, _) => Self::Decimal128(Vec::with_capacity(capacity)),
            DataType::Interval(IntervalUnit::MonthDayNano) => {
                Self::Interval(Vec::with_capacity(capacity))
            }
            _ => Self::Bytes(Vec::with_capacity(capacity * 8)),
        }
    }

    /// Decode a value from cursor using typed push (fast path)
    #[inline]
    fn decode(&mut self, cursor: &mut AvroCursor<'_>) -> Result<(), ArrowError> {
        match self {
            Self::Null(count) => *count += 1,
            Self::Boolean(vec) => vec.push(if cursor.get_bool()? { 1 } else { 0 }),
            Self::Int32(vec) => vec.push(cursor.get_int()?),
            Self::Int64(vec) => vec.push(cursor.get_long()?),
            Self::Float32(vec) => vec.push(cursor.get_float()?),
            Self::Float64(vec) => vec.push(cursor.get_double()?),
            Self::String { offsets, data } | Self::Binary { offsets, data } => {
                let bytes = cursor.get_bytes()?;
                if offsets.is_empty() {
                    offsets.push(0);
                }
                data.extend_from_slice(bytes);
                offsets.push(data.len() as i32);
            }
            Self::Fixed { size, data } => {
                let bytes = cursor.get_fixed(*size)?;
                data.extend_from_slice(bytes);
            }
            Self::Decimal128(vec) => {
                let bytes = cursor.get_bytes()?;
                vec.push(decode_decimal_bytes_to_i128(bytes));
            }
            Self::Interval(vec) => {
                let bytes = cursor.get_fixed(12)?;
                let months = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as i32;
                let days = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as i32;
                let millis = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
                let nanos = (millis as i64) * 1_000_000;
                let interval = IntervalMonthDayNano::new(months, days, nanos);
                let interval_bytes: [u8; 16] = unsafe { std::mem::transmute(interval) };
                vec.push(interval_bytes);
            }
            Self::Bytes(vec) => {
                // Fallback: store length-prefixed bytes
                let bytes = cursor.get_bytes()?;
                vec.extend_from_slice(&(bytes.len() as i32).to_le_bytes());
                vec.extend_from_slice(bytes);
            }
        }
        Ok(())
    }

    /// Clear for reuse (keeps allocated capacity)
    fn clear(&mut self) {
        match self {
            Self::Null(count) => *count = 0,
            Self::Boolean(vec) => vec.clear(),
            Self::Int32(vec) => vec.clear(),
            Self::Int64(vec) => vec.clear(),
            Self::Float32(vec) => vec.clear(),
            Self::Float64(vec) => vec.clear(),
            Self::String { offsets, data } | Self::Binary { offsets, data } => {
                offsets.clear();
                data.clear();
            }
            Self::Fixed { data, .. } => data.clear(),
            Self::Decimal128(vec) => vec.clear(),
            Self::Interval(vec) => vec.clear(),
            Self::Bytes(vec) => vec.clear(),
        }
    }

    /// Get row count
    fn len(&self) -> usize {
        match self {
            Self::Null(count) => *count,
            Self::Boolean(vec) => vec.len(),
            Self::Int32(vec) => vec.len(),
            Self::Int64(vec) => vec.len(),
            Self::Float32(vec) => vec.len(),
            Self::Float64(vec) => vec.len(),
            Self::String { offsets, .. } | Self::Binary { offsets, .. } => {
                if offsets.is_empty() { 0 } else { offsets.len() - 1 }
            }
            Self::Fixed { size, data } => if *size > 0 { data.len() / *size } else { 0 },
            Self::Decimal128(vec) => vec.len(),
            Self::Interval(vec) => vec.len(),
            Self::Bytes(_) => 0, // Can't determine easily
        }
    }

    /// Truncate to specified row count (for snapshot rollback)
    fn truncate(&mut self, len: usize) {
        match self {
            Self::Null(count) => *count = len,
            Self::Boolean(vec) => vec.truncate(len),
            Self::Int32(vec) => vec.truncate(len),
            Self::Int64(vec) => vec.truncate(len),
            Self::Float32(vec) => vec.truncate(len),
            Self::Float64(vec) => vec.truncate(len),
            Self::String { offsets, data } | Self::Binary { offsets, data } => {
                // Truncate to len+1 offsets (for len rows)
                let target_offsets = if len == 0 { 0 } else { len + 1 };
                if offsets.len() > target_offsets {
                    let data_len = if target_offsets > 0 {
                        offsets[target_offsets - 1] as usize
                    } else {
                        0
                    };
                    offsets.truncate(target_offsets);
                    data.truncate(data_len);
                }
            }
            Self::Fixed { size, data } => data.truncate(len * *size),
            Self::Decimal128(vec) => vec.truncate(len),
            Self::Interval(vec) => vec.truncate(len),
            Self::Bytes(_) => { /* Can't truncate easily */ }
        }
    }

    /// Convert to Arrow array
    fn to_array(&self, data_type: &DataType, nulls: Option<NullBuffer>) -> Result<ArrayRef, ArrowError> {
        match (self, data_type) {
            (Self::Null(count), DataType::Null) => {
                Ok(Arc::new(NullArray::new(*count)))
            }
            (Self::Boolean(vec), DataType::Boolean) => {
                let mut builder = arrow_array::builder::BooleanBuilder::with_capacity(vec.len());
                for (i, &b) in vec.iter().enumerate() {
                    let is_valid = nulls.as_ref().map(|n| n.is_valid(i)).unwrap_or(true);
                    if is_valid {
                        builder.append_value(b != 0);
                    } else {
                        builder.append_null();
                    }
                }
                Ok(Arc::new(builder.finish()))
            }
            (Self::Int32(vec), DataType::Int32) => {
                let buffer = ScalarBuffer::from(vec.clone());
                Ok(Arc::new(Int32Array::new(buffer, nulls)))
            }
            (Self::Int32(vec), DataType::Date32) => {
                let buffer = ScalarBuffer::from(vec.clone());
                Ok(Arc::new(Date32Array::new(buffer, nulls)))
            }
            (Self::Int32(vec), DataType::Time32(unit)) => {
                let buffer = ScalarBuffer::from(vec.clone());
                let array = match unit {
                    arrow_schema::TimeUnit::Second => Arc::new(Time32SecondArray::new(buffer, nulls)) as ArrayRef,
                    arrow_schema::TimeUnit::Millisecond => Arc::new(Time32MillisecondArray::new(buffer, nulls)) as ArrayRef,
                    _ => return Err(ArrowError::InvalidArgumentError(format!("Invalid Time32 unit: {:?}", unit))),
                };
                Ok(array)
            }
            (Self::Int64(vec), DataType::Int64) => {
                let buffer = ScalarBuffer::from(vec.clone());
                Ok(Arc::new(Int64Array::new(buffer, nulls)))
            }
            (Self::Int64(vec), DataType::Timestamp(unit, tz)) => {
                let buffer = ScalarBuffer::from(vec.clone());
                let array = match unit {
                    arrow_schema::TimeUnit::Second => Arc::new(TimestampSecondArray::new(buffer, nulls).with_timezone_opt(tz.clone())) as ArrayRef,
                    arrow_schema::TimeUnit::Millisecond => Arc::new(TimestampMillisecondArray::new(buffer, nulls).with_timezone_opt(tz.clone())) as ArrayRef,
                    arrow_schema::TimeUnit::Microsecond => Arc::new(TimestampMicrosecondArray::new(buffer, nulls).with_timezone_opt(tz.clone())) as ArrayRef,
                    arrow_schema::TimeUnit::Nanosecond => Arc::new(TimestampNanosecondArray::new(buffer, nulls).with_timezone_opt(tz.clone())) as ArrayRef,
                };
                Ok(array)
            }
            (Self::Float32(vec), DataType::Float32) => {
                let buffer = ScalarBuffer::from(vec.clone());
                Ok(Arc::new(Float32Array::new(buffer, nulls)))
            }
            (Self::Float64(vec), DataType::Float64) => {
                let buffer = ScalarBuffer::from(vec.clone());
                Ok(Arc::new(Float64Array::new(buffer, nulls)))
            }
            (Self::String { offsets, data }, DataType::Utf8) => {
                let offsets = if offsets.is_empty() {
                    OffsetBuffer::new(ScalarBuffer::from(vec![0i32]))
                } else {
                    OffsetBuffer::new(ScalarBuffer::from(offsets.clone()))
                };
                let values = Buffer::from_slice_ref(data);
                Ok(Arc::new(StringArray::new(offsets, values, nulls)))
            }
            (Self::Binary { offsets, data }, DataType::Binary) => {
                let offsets = if offsets.is_empty() {
                    OffsetBuffer::new(ScalarBuffer::from(vec![0i32]))
                } else {
                    OffsetBuffer::new(ScalarBuffer::from(offsets.clone()))
                };
                let values = Buffer::from_slice_ref(data);
                Ok(Arc::new(BinaryArray::new(offsets, values, nulls)))
            }
            (Self::Fixed { size, data }, DataType::FixedSizeBinary(expected_size)) => {
                if *size != *expected_size as usize {
                    return Err(ArrowError::InvalidArgumentError(format!(
                        "Fixed size mismatch: {} vs {}", size, expected_size
                    )));
                }
                Ok(Arc::new(FixedSizeBinaryArray::try_new(*expected_size, Buffer::from_slice_ref(data), nulls)?))
            }
            (Self::Decimal128(vec), DataType::Decimal128(precision, scale)) => {
                let buffer = ScalarBuffer::from(vec.clone());
                let array = Decimal128Array::new(buffer, nulls)
                    .with_precision_and_scale(*precision, *scale)
                    .map_err(|e| ArrowError::InvalidArgumentError(e.to_string()))?;
                Ok(Arc::new(array))
            }
            (Self::Interval(vec), DataType::Interval(IntervalUnit::MonthDayNano)) => {
                // Convert [u8; 16] to IntervalMonthDayNano
                let values: Vec<IntervalMonthDayNano> = vec.iter().map(|bytes| {
                    unsafe { std::mem::transmute::<[u8; 16], IntervalMonthDayNano>(*bytes) }
                }).collect();
                let buffer = ScalarBuffer::from(values);
                Ok(Arc::new(IntervalMonthDayNanoArray::new(buffer, nulls)))
            }
            _ => Err(ArrowError::InvalidArgumentError(format!(
                "TypedStorage type mismatch for {:?}", data_type
            ))),
        }
    }
}

#[allow(dead_code)]
impl FieldDecoder {
    /// Build decoder tree from Arrow DataType (pre-compile at construction time)
    fn from_data_type(data_type: &DataType) -> Self {
        match data_type {
            DataType::Null => Self::Null,
            DataType::Boolean => Self::Boolean,
            DataType::Int32 | DataType::Date32 | DataType::Time32(_) => Self::Int32,
            DataType::Int64 | DataType::Time64(_) | DataType::Timestamp(_, _) => Self::Int64,
            DataType::Float32 => Self::Float32,
            DataType::Float64 => Self::Float64,
            DataType::Utf8 => Self::String,
            DataType::Binary => Self::Binary,
            DataType::FixedSizeBinary(size) => Self::FixedBytes { size: *size as usize },
            DataType::Decimal128(_, _) => Self::Decimal128,
            DataType::Decimal256(_, _) => Self::Decimal256,
            DataType::Interval(IntervalUnit::MonthDayNano) => Self::Interval,
            DataType::Struct(fields) => {
                let child_decoders: Vec<FieldDecoder> = fields
                    .iter()
                    .map(|f| Self::from_data_type(f.data_type()))
                    .collect();
                let nullable_flags: Vec<bool> = fields.iter().map(|f| f.is_nullable()).collect();
                Self::Struct {
                    fields: child_decoders,
                    nullable_flags,
                }
            }
            DataType::List(child_field) => Self::List {
                child: Box::new(Self::from_data_type(child_field.data_type())),
                child_nullable: child_field.is_nullable(),
            },
            DataType::Map(entries_field, _) => {
                let (value_type, value_nullable) =
                    if let DataType::Struct(fields) = entries_field.data_type() {
                        fields
                            .iter()
                            .nth(1)
                            .map(|f| (f.data_type().clone(), f.is_nullable()))
                            .unwrap_or((DataType::Utf8, false))
                    } else {
                        (DataType::Utf8, false)
                    };
                Self::Map {
                    value: Box::new(Self::from_data_type(&value_type)),
                    value_nullable,
                }
            }
            DataType::Union(union_fields, _) => {
                let variants: Vec<FieldDecoder> = union_fields
                    .iter()
                    .map(|(_, field)| Self::from_data_type(field.data_type()))
                    .collect();
                let type_ids: Vec<i8> = union_fields.iter().map(|(id, _)| id).collect();
                Self::Union { variants, type_ids }
            }
            // Fallback for other types
            _ => Self::Int64,
        }
    }

    /// Decode value using pre-compiled type info (no runtime type matching)
    #[inline]
    fn decode(&self, cursor: &mut AvroCursor<'_>, output: &mut Vec<u8>) -> Result<(), ArrowError> {
        match self {
            Self::Null => {
                // Null takes no space
            }
            Self::Boolean => {
                let value = cursor.get_bool()?;
                output.push(if value { 1 } else { 0 });
            }
            Self::Int32 => {
                let value = cursor.get_int()?;
                output.extend_from_slice(&value.to_le_bytes());
            }
            Self::Int64 => {
                let value = cursor.get_long()?;
                output.extend_from_slice(&value.to_le_bytes());
            }
            Self::Float32 => {
                let value = cursor.get_float()?;
                output.extend_from_slice(&value.to_le_bytes());
            }
            Self::Float64 => {
                let value = cursor.get_double()?;
                output.extend_from_slice(&value.to_le_bytes());
            }
            Self::String | Self::Binary => {
                let bytes = cursor.get_bytes()?;
                output.extend_from_slice(&(bytes.len() as i32).to_le_bytes());
                output.extend_from_slice(bytes);
            }
            Self::FixedBytes { size } => {
                let bytes = cursor.get_fixed(*size)?;
                output.extend_from_slice(bytes);
            }
            Self::Decimal128 => {
                let bytes = cursor.get_bytes()?;
                let value = decode_decimal_bytes_to_i128(bytes);
                output.extend_from_slice(&value.to_le_bytes());
            }
            Self::Decimal256 => {
                let bytes = cursor.get_bytes()?;
                let value = decode_decimal_bytes_to_i256(bytes);
                output.extend_from_slice(&value);
            }
            Self::Interval => {
                let bytes = cursor.get_fixed(12)?;
                let months = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as i32;
                let days = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as i32;
                let millis = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
                let nanos = (millis as i64) * 1_000_000;
                let interval = IntervalMonthDayNano::new(months, days, nanos);
                let interval_bytes: [u8; 16] = unsafe { std::mem::transmute(interval) };
                output.extend_from_slice(&interval_bytes);
            }
            Self::Struct {
                fields,
                nullable_flags,
            } => {
                // Pre-compiled field iteration - no runtime type lookup
                for (decoder, is_nullable) in fields.iter().zip(nullable_flags.iter()) {
                    if *is_nullable {
                        let union_index = cursor.get_long()?;
                        if union_index == 0 {
                            output.push(0); // null marker
                        } else {
                            output.push(1); // valid marker
                            decoder.decode(cursor, output)?;
                        }
                    } else {
                        decoder.decode(cursor, output)?;
                    }
                }
            }
            Self::List {
                child,
                child_nullable,
            } => {
                // List block decoding
                loop {
                    let block_count = cursor.get_long()?;
                    if block_count == 0 {
                        break;
                    }
                    let count = if block_count < 0 {
                        let _size = cursor.get_long()?;
                        (-block_count) as usize
                    } else {
                        block_count as usize
                    };
                    for _ in 0..count {
                        if *child_nullable {
                            let union_index = cursor.get_long()?;
                            if union_index == 0 {
                                output.push(0); // null marker
                            } else {
                                output.push(1); // valid marker
                                child.decode(cursor, output)?;
                            }
                        } else {
                            child.decode(cursor, output)?;
                        }
                    }
                }
            }
            Self::Map {
                value,
                value_nullable,
            } => {
                // Map block decoding
                loop {
                    let block_count = cursor.get_long()?;
                    if block_count == 0 {
                        break;
                    }
                    let count = if block_count < 0 {
                        let _size = cursor.get_long()?;
                        (-block_count) as usize
                    } else {
                        block_count as usize
                    };
                    for _ in 0..count {
                        // Key is always string
                        let key_bytes = cursor.get_bytes()?;
                        output.extend_from_slice(&(key_bytes.len() as i32).to_le_bytes());
                        output.extend_from_slice(key_bytes);
                        // Value
                        if *value_nullable {
                            let union_index = cursor.get_long()?;
                            if union_index == 0 {
                                output.push(0); // null marker
                            } else {
                                output.push(1); // valid marker
                                value.decode(cursor, output)?;
                            }
                        } else {
                            value.decode(cursor, output)?;
                        }
                    }
                }
            }
            Self::Union { variants, type_ids } => {
                let type_index = cursor.get_int()? as usize;
                if type_index >= variants.len() {
                    return Err(ArrowError::ParseError(format!(
                        "Union type index {} out of bounds",
                        type_index
                    )));
                }
                let tag = type_ids[type_index];
                output.extend_from_slice(&(tag as i32).to_le_bytes());
                if !matches!(variants[type_index], Self::Null) {
                    variants[type_index].decode(cursor, output)?;
                }
            }
        }
        Ok(())
    }

    /// Decode with separate validity tracking for struct fields
    #[inline]
    fn decode_to_field_data(
        &self,
        cursor: &mut AvroCursor<'_>,
        field_data: &mut [Vec<u8>],
        field_validity: &mut [Vec<bool>],
        nullable_flags: &[bool],
    ) -> Result<(), ArrowError> {
        if let Self::Struct { fields, .. } = self {
            // Pre-compiled field iteration - no runtime type lookup
            for (field_idx, decoder) in fields.iter().enumerate() {
                if nullable_flags[field_idx] {
                    let union_index = cursor.get_long()?;
                    if union_index == 0 {
                        field_validity[field_idx].push(false);
                        // Append type-appropriate placeholder
                        Self::append_placeholder(decoder, &mut field_data[field_idx]);
                    } else {
                        field_validity[field_idx].push(true);
                        decoder.decode(cursor, &mut field_data[field_idx])?;
                    }
                } else {
                    decoder.decode(cursor, &mut field_data[field_idx])?;
                }
            }
            Ok(())
        } else {
            Err(ArrowError::InvalidArgumentError(
                "decode_to_field_data only works with Struct decoder".to_string(),
            ))
        }
    }

    /// Append type-appropriate null placeholder
    #[inline]
    fn append_placeholder(decoder: &FieldDecoder, output: &mut Vec<u8>) {
        match decoder {
            Self::Null => {}
            Self::Boolean => output.push(0),
            Self::Int32 => output.extend_from_slice(&0i32.to_le_bytes()),
            Self::Int64 => output.extend_from_slice(&0i64.to_le_bytes()),
            Self::Float32 => output.extend_from_slice(&0f32.to_le_bytes()),
            Self::Float64 => output.extend_from_slice(&0f64.to_le_bytes()),
            Self::String | Self::Binary => {
                output.extend_from_slice(&0i32.to_le_bytes()); // empty length
            }
            Self::FixedBytes { size } => {
                output.extend(std::iter::repeat(0u8).take(*size));
            }
            Self::Decimal128 => output.extend_from_slice(&0i128.to_le_bytes()),
            Self::Decimal256 => output.extend_from_slice(&[0u8; 32]),
            Self::Interval => output.extend_from_slice(&[0u8; 16]),
            Self::Struct { fields, .. } => {
                for field in fields {
                    Self::append_placeholder(field, output);
                }
            }
            Self::List { .. } | Self::Map { .. } | Self::Union { .. } => {
                // Nested types: placeholder is empty/default representation
            }
        }
    }

    /// Get fixed size for this decoder (None for variable-length types)
    #[inline]
    fn fixed_size(&self) -> Option<usize> {
        match self {
            Self::Null => Some(0),
            Self::Boolean => Some(1),
            Self::Int32 | Self::Float32 => Some(4),
            Self::Int64 | Self::Float64 => Some(8),
            Self::Decimal128 | Self::Interval => Some(16),
            Self::Decimal256 => Some(32),
            Self::FixedBytes { size } => Some(*size),
            Self::String | Self::Binary | Self::List { .. } | Self::Map { .. }
            | Self::Union { .. } | Self::Struct { .. } => None,
        }
    }

    /// Decode fixed-size value directly to a pre-allocated slice (returns bytes written)
    /// For variable-length types, returns None (caller should use decode() instead)
    #[inline]
    fn decode_to_slice(&self, cursor: &mut AvroCursor<'_>, output: &mut [u8]) -> Result<usize, ArrowError> {
        match self {
            Self::Null => Ok(0),
            Self::Boolean => {
                let value = cursor.get_bool()?;
                output[0] = if value { 1 } else { 0 };
                Ok(1)
            }
            Self::Int32 => {
                let value = cursor.get_int()?;
                output[..4].copy_from_slice(&value.to_le_bytes());
                Ok(4)
            }
            Self::Int64 => {
                let value = cursor.get_long()?;
                output[..8].copy_from_slice(&value.to_le_bytes());
                Ok(8)
            }
            Self::Float32 => {
                let value = cursor.get_float()?;
                output[..4].copy_from_slice(&value.to_le_bytes());
                Ok(4)
            }
            Self::Float64 => {
                let value = cursor.get_double()?;
                output[..8].copy_from_slice(&value.to_le_bytes());
                Ok(8)
            }
            Self::FixedBytes { size } => {
                let bytes = cursor.get_fixed(*size)?;
                output[..*size].copy_from_slice(bytes);
                Ok(*size)
            }
            Self::Decimal128 => {
                let bytes = cursor.get_bytes()?;
                let value = decode_decimal_bytes_to_i128(bytes);
                output[..16].copy_from_slice(&value.to_le_bytes());
                Ok(16)
            }
            Self::Decimal256 => {
                let bytes = cursor.get_bytes()?;
                let value = decode_decimal_bytes_to_i256(bytes);
                output[..32].copy_from_slice(&value);
                Ok(32)
            }
            Self::Interval => {
                let bytes = cursor.get_fixed(12)?;
                let months = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as i32;
                let days = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as i32;
                let millis = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
                let nanos = (millis as i64) * 1_000_000;
                let interval = IntervalMonthDayNano::new(months, days, nanos);
                let interval_bytes: [u8; 16] = unsafe { std::mem::transmute(interval) };
                output[..16].copy_from_slice(&interval_bytes);
                Ok(16)
            }
            // Variable-length and complex types - caller should handle differently
            Self::String | Self::Binary | Self::List { .. } | Self::Map { .. }
            | Self::Union { .. } | Self::Struct { .. } => {
                Err(ArrowError::InvalidArgumentError(
                    "decode_to_slice only works with fixed-size types".to_string(),
                ))
            }
        }
    }

}

/// Nested type state for complex types (Map, List, Union, Struct, Dictionary)
#[derive(Debug)]
#[allow(dead_code)]
enum NestedState {
    /// No nested state (primitive types)
    None,
    /// Map type: keys (strings) and values - always uses Vec mode
    Map {
        /// Map offsets (one per row, tracking number of entries)
        map_offsets: Vec<i32>,
        /// Key offsets for string keys
        key_offsets: Vec<i32>,
        /// Key data bytes
        key_data: Vec<u8>,
        /// Value offsets for variable-length values
        value_offsets: Vec<i32>,
        /// Value data - stored as raw bytes (no markers, placeholders for nulls)
        value_data: Vec<u8>,
        /// Value validity bitmap (true=valid, false=null) - separate from data
        value_validity: Vec<bool>,
        /// Value type
        value_type: DataType,
        /// Whether value field is nullable
        value_nullable: bool,
        /// Current total entries
        total_entries: usize,
    },
    /// Union type: type IDs and child data - always uses Vec mode
    Union {
        /// Type IDs for each row
        type_ids: Vec<i8>,
        /// Offsets for dense union (optional)
        offsets: Option<Vec<i32>>,
        /// Child data per variant (index by type_id)
        child_data: Vec<Vec<u8>>,
        /// Child offsets per variant for variable-length types (Utf8/Binary)
        child_offsets: Vec<Vec<i32>>,
        /// Child counts per variant
        child_counts: Vec<usize>,
        /// Union fields info
        fields: UnionFields,
        /// Union mode
        mode: UnionMode,
    },
    /// List type: offsets and child data
    List {
        /// List offsets
        offsets: Vec<i32>,
        /// Child element data (Vec for transitional, will migrate to pre-allocated)
        child_data: Vec<u8>,
        /// Child validity tracking (true=valid, false=null) - separate from data
        child_validity: Vec<bool>,
        /// Current position in child data (for pre-allocated buffer mode)
        child_data_position: usize,
        /// Current position in child offsets (for pre-allocated buffer mode with Utf8/Binary)
        child_offsets_position: usize,
        /// Child element type
        child_type: DataType,
        /// Whether child elements are nullable
        child_nullable: bool,
        /// Total elements
        total_elements: usize,
        /// Whether to use pre-allocated buffers (vs Vec)
        use_preallocated: bool,
    },
    /// Dictionary type: indices and dictionary values (for Avro enum)
    Dictionary {
        /// Indices (varint encoded enum values become i32 indices)
        indices: Vec<i32>,
        /// Dictionary values (enum symbols from schema)
        dictionary: Vec<String>,
    },
    /// Struct type: nested fields with their own data buffers
    Struct {
        /// Field data stored per-field (each field has its own Vec<u8>)
        /// DEPRECATED: Use typed_storage instead for better performance
        field_data: Vec<Vec<u8>>,
        /// Field types
        field_types: Vec<DataType>,
        /// Field names for building StructArray
        field_names: Vec<String>,
        /// Field nullability
        field_nullable: Vec<bool>,
        /// Null bitmaps for nullable fields (bit-packed, 1=valid, 0=null)
        field_null_bitmaps: Vec<Vec<u8>>,
        /// Separate validity tracking per field (true=valid, false=null) - single pass decoding
        field_validity: Vec<Vec<bool>>,
        /// Row count for this struct
        row_count: usize,
        /// Current data positions per field (for pre-allocated buffer mode)
        field_positions: Vec<usize>,
        /// Current offsets positions per field (for variable-length fields in pre-allocated mode)
        field_offsets_positions: Vec<usize>,
        /// Whether to use pre-allocated buffers (vs Vec)
        use_preallocated: bool,
        /// Pre-compiled decoder for fast typed dispatch (no runtime type matching)
        decoder: FieldDecoder,
        /// Function pointer specs for zero-overhead fixed-size struct decoding
        /// None if any field is variable-length
        field_specs: Option<Vec<StructFieldSpec>>,
        /// Typed storage for each field - uses typed Vecs like Apache's Decoder
        /// This replaces field_data for better performance (direct typed push vs byte conversion)
        typed_storage: Vec<TypedStorage>,
        /// Apache Decoders for each field - uses Apache's proven Decoder directly
        /// This provides the best performance for struct decoding
        apache_decoders: Vec<ApacheDecoder>,
        /// Single Apache Record Decoder for entire struct - matches Apache's exact code path
        /// This handles nullable fields internally via Decoder::Nullable wrappers
        record_decoder: ApacheDecoder,
    },
}

/// Column state for pooled decoding
#[derive(Debug)]
#[allow(dead_code)]
struct ColumnState {
    /// Column index in the schema
    column_index: usize,
    /// Arrow data type for this column
    data_type: DataType,
    /// Current write position in data buffer
    data_position: usize,
    /// Current row count
    row_count: usize,
    /// Is this column nullable?
    is_nullable: bool,
    /// Null bitmap for nullable columns (bit-packed, 1=valid, 0=null)
    /// For Avro nullable unions, we read the union tag and set the bit accordingly
    null_bitmap: Vec<u8>,
    /// Nested type state for complex types
    nested: NestedState,
}

impl ColumnState {
    fn new(column_index: usize, field: &Field, use_preallocated: bool, row_hint: usize) -> Self {
        let nested = Self::create_nested_state(field.data_type(), use_preallocated, row_hint);
        Self {
            column_index,
            data_type: field.data_type().clone(),
            data_position: 0,
            row_count: 0,
            is_nullable: field.is_nullable(),
            null_bitmap: Vec::new(), // Will grow as needed during decode
            nested,
        }
    }

    /// Set a bit in the null bitmap (1 = valid/non-null, 0 = null)
    #[inline]
    fn set_validity(&mut self, row: usize, is_valid: bool) {
        if !self.is_nullable {
            return; // Non-nullable columns don't need bitmap
        }

        let byte_idx = row / 8;
        let bit_idx = row % 8;

        // Grow bitmap if needed
        while self.null_bitmap.len() <= byte_idx {
            self.null_bitmap.push(0);
        }

        if is_valid {
            self.null_bitmap[byte_idx] |= 1 << bit_idx;
        }
        // Note: bits default to 0 (null), so we only set for valid values
    }

    /// Get the null bitmap as a NullBuffer for Arrow array construction
    fn get_null_buffer(&self) -> Option<arrow_buffer::NullBuffer> {
        if !self.is_nullable || self.null_bitmap.is_empty() {
            return None;
        }

        // Convert Vec<u8> to Buffer for NullBuffer
        let buffer = arrow_buffer::Buffer::from(self.null_bitmap.clone());
        Some(arrow_buffer::NullBuffer::new(arrow_buffer::BooleanBuffer::new(
            buffer,
            0,
            self.row_count,
        )))
    }

    fn create_nested_state(data_type: &DataType, use_preallocated: bool, row_hint: usize) -> NestedState {
        match data_type {
            DataType::Map(entries_field, _) => {
                // Extract value type and nullability from map entries struct
                let (value_type, value_nullable) = if let DataType::Struct(fields) = entries_field.data_type() {
                    fields.iter()
                        .find(|f| f.name() == "value")
                        .map(|f| (f.data_type().clone(), f.is_nullable()))
                        .unwrap_or((DataType::Utf8, false))
                } else {
                    (DataType::Utf8, false)
                };
                NestedState::Map {
                    map_offsets: vec![0],
                    key_offsets: vec![0],
                    key_data: Vec::new(),
                    value_offsets: vec![0],
                    value_data: Vec::new(),
                    value_validity: Vec::new(), // Separate validity tracking
                    value_type,
                    value_nullable,
                    total_entries: 0,
                }
            }
            DataType::Union(fields, mode) => {
                let num_variants = fields.len();
                // Pre-size vectors based on row_hint to reduce reallocations
                // For unions, type_ids has one entry per row
                let type_ids = Vec::with_capacity(row_hint);
                let offsets = if *mode == UnionMode::Dense {
                    Some(Vec::with_capacity(row_hint))
                } else {
                    None
                };
                // Pre-size child_data based on estimated element sizes
                // For nullable unions (null + type), estimate 50% null, 50% values
                let child_data: Vec<Vec<u8>> = fields.iter().map(|(_, field)| {
                    let element_size = Self::estimate_element_size(field.data_type());
                    // Estimate half of rows go to each variant in a binary union
                    let estimated_count = if num_variants == 2 { row_hint / 2 } else { row_hint / num_variants };
                    Vec::with_capacity(estimated_count * element_size)
                }).collect();
                let child_offsets: Vec<Vec<i32>> = fields.iter().map(|(_, field)| {
                    match field.data_type() {
                        DataType::Utf8 | DataType::Binary => {
                            let estimated_count = if num_variants == 2 { row_hint / 2 } else { row_hint / num_variants };
                            let mut v = Vec::with_capacity(estimated_count + 1);
                            v.push(0);
                            v
                        }
                        _ => vec![0],
                    }
                }).collect();
                NestedState::Union {
                    type_ids,
                    offsets,
                    child_data,
                    child_offsets,
                    child_counts: vec![0; num_variants],
                    fields: fields.clone(),
                    mode: *mode,
                }
            }
            DataType::List(child_field) => {
                NestedState::List {
                    offsets: vec![0],
                    child_data: Vec::new(),
                    child_validity: Vec::new(),
                    child_data_position: 0,
                    // For pre-allocated mode with variable-length children, skip initial offset 0
                    child_offsets_position: if use_preallocated { 4 } else { 0 },
                    child_type: child_field.data_type().clone(),
                    child_nullable: child_field.is_nullable(),
                    total_elements: 0,
                    use_preallocated,
                }
            }
            DataType::Dictionary(_, _) => {
                // Dictionary values will be set via set_dictionary_values()
                // or extracted from field metadata
                NestedState::Dictionary {
                    indices: Vec::new(),
                    dictionary: Vec::new(),
                }
            }
            DataType::Struct(fields) => {
                // NOTE: Struct always uses Vec mode - preallocated mode has excessive
                // per-row overhead from get_child_data_slice calls and batch-of-1 decoding.
                // Vec mode with simple appends is significantly faster.
                let num_fields = fields.len();
                let field_data: Vec<Vec<u8>> = fields.iter().map(|_| Vec::new()).collect();
                let field_types: Vec<DataType> = fields.iter().map(|f| f.data_type().clone()).collect();
                let field_names: Vec<String> = fields.iter().map(|f| f.name().clone()).collect();
                let field_nullable: Vec<bool> = fields.iter().map(|f| f.is_nullable()).collect();
                let field_null_bitmaps: Vec<Vec<u8>> = fields.iter().map(|_| Vec::new()).collect();
                let field_validity: Vec<Vec<bool>> = fields.iter().map(|_| Vec::new()).collect();
                // Pre-compile the decoder tree for fast dispatch (no runtime type matching)
                let decoder = FieldDecoder::from_data_type(data_type);

                // Build function pointer specs for zero-overhead decoding (if all fields are fixed-size)
                let field_specs: Option<Vec<StructFieldSpec>> = if let FieldDecoder::Struct { fields: field_decoders, .. } = &decoder {
                    let specs: Vec<Option<StructFieldSpec>> = field_decoders
                        .iter()
                        .map(StructFieldSpec::from_field_decoder)
                        .collect();
                    if specs.iter().all(|s| s.is_some()) {
                        Some(specs.into_iter().map(|s| s.unwrap()).collect())
                    } else {
                        None // Has variable-length fields, can't use fast path
                    }
                } else {
                    None
                };

                // Initialize typed storage for each field (Apache-style typed Vecs)
                let typed_storage: Vec<TypedStorage> = fields
                    .iter()
                    .map(|f| TypedStorage::from_data_type(f.data_type(), row_hint))
                    .collect();

                // Initialize Apache Decoders for each field (uses Apache's proven Decoder)
                let apache_decoders: Vec<ApacheDecoder> = fields
                    .iter()
                    .map(|f| ApacheDecoder::from_arrow_type(f.data_type()))
                    .collect();

                // Create single Record decoder from DataType::Struct - this wraps nullable fields
                // in Decoder::Nullable automatically, giving us Apache's exact decode path
                let record_decoder = ApacheDecoder::from_arrow_type(data_type);

                NestedState::Struct {
                    field_data,
                    field_types,
                    field_names,
                    field_nullable,
                    field_null_bitmaps,
                    field_validity,
                    row_count: 0,
                    field_positions: vec![0; num_fields],
                    field_offsets_positions: vec![0; num_fields],
                    // Only use preallocated for structs with all fixed-size fields (field_specs is Some)
                    use_preallocated: use_preallocated && field_specs.is_some(),
                    decoder,
                    field_specs,
                    typed_storage,
                    apache_decoders,
                    record_decoder,
                }
            }
            _ => NestedState::None,
        }
    }

    fn reset(&mut self) {
        self.data_position = 0;
        self.row_count = 0;
        // Reset nested state
        match &mut self.nested {
            NestedState::None => {}
            NestedState::Map { map_offsets, key_offsets, key_data, value_offsets, value_data, value_validity, total_entries, .. } => {
                map_offsets.clear();
                map_offsets.push(0);
                key_offsets.clear();
                key_offsets.push(0);
                key_data.clear();
                value_offsets.clear();
                value_offsets.push(0);
                value_data.clear();
                value_validity.clear();
                *total_entries = 0;
            }
            NestedState::Union { type_ids, offsets, child_data, child_offsets, child_counts, .. } => {
                type_ids.clear();
                if let Some(off) = offsets {
                    off.clear();
                }
                for data in child_data.iter_mut() {
                    data.clear();
                }
                for offs in child_offsets.iter_mut() {
                    offs.clear();
                    offs.push(0);
                }
                for count in child_counts.iter_mut() {
                    *count = 0;
                }
            }
            NestedState::List { offsets, child_data, child_validity, child_data_position, child_offsets_position, total_elements, use_preallocated, .. } => {
                offsets.clear();
                offsets.push(0);
                child_data.clear();
                child_validity.clear();
                *child_data_position = 0;
                // For pre-allocated mode with variable-length children (Utf8/Binary),
                // skip the first 4 bytes which contain the initial offset 0 (pre-zeroed buffer)
                *child_offsets_position = if *use_preallocated { 4 } else { 0 };
                *total_elements = 0;
            }
            NestedState::Dictionary { indices, dictionary: _ } => {
                // Keep dictionary values (they come from schema), clear indices
                indices.clear();
            }
            NestedState::Struct { field_data, field_null_bitmaps, field_validity, row_count, field_positions, field_offsets_positions, field_types, field_names, field_nullable, use_preallocated, typed_storage, apache_decoders, record_decoder, .. } => {
                for data in field_data.iter_mut() {
                    data.clear();
                }
                for bitmap in field_null_bitmaps.iter_mut() {
                    bitmap.clear();
                }
                for validity in field_validity.iter_mut() {
                    validity.clear();
                }
                // Clear typed storage (keeps allocated capacity for reuse)
                for storage in typed_storage.iter_mut() {
                    storage.clear();
                }
                // Recreate apache decoders (they auto-reset after flush, but need recreation for explicit reset)
                for (decoder, field_type) in apache_decoders.iter_mut().zip(field_types.iter()) {
                    *decoder = ApacheDecoder::from_arrow_type(field_type);
                }
                // Recreate record_decoder from struct type
                let struct_fields: Vec<FieldRef> = field_names.iter()
                    .zip(field_types.iter())
                    .zip(field_nullable.iter())
                    .map(|((name, dtype), nullable)| Arc::new(Field::new(name, dtype.clone(), *nullable)))
                    .collect();
                *record_decoder = ApacheDecoder::from_arrow_type(&DataType::Struct(Fields::from(struct_fields)));
                *row_count = 0;
                for pos in field_positions.iter_mut() {
                    *pos = 0;
                }
                // Reset offset positions - skip initial 0 for variable-length fields in pre-allocated mode
                for (i, pos) in field_offsets_positions.iter_mut().enumerate() {
                    if *use_preallocated && matches!(field_types[i], DataType::Utf8 | DataType::Binary) {
                        *pos = 4; // Skip initial offset 0
                    } else {
                        *pos = 0;
                    }
                }
            }
        }
    }

    /// Create a snapshot of nested state for rollback
    fn create_snapshot(&self) -> NestedSnapshot {
        match &self.nested {
            NestedState::None => NestedSnapshot::None,
            NestedState::Map { map_offsets, key_offsets, key_data, value_offsets, value_data, value_validity, total_entries, .. } => {
                NestedSnapshot::Map {
                    map_offsets_len: map_offsets.len(),
                    key_offsets_len: key_offsets.len(),
                    key_data_len: key_data.len(),
                    value_offsets_len: value_offsets.len(),
                    value_data_len: value_data.len(),
                    value_validity_len: value_validity.len(),
                    total_entries: *total_entries,
                }
            }
            NestedState::Union { type_ids, offsets, child_data, child_offsets, child_counts, .. } => {
                NestedSnapshot::Union {
                    type_ids_len: type_ids.len(),
                    offsets_len: offsets.as_ref().map(|o| o.len()),
                    child_data_lens: child_data.iter().map(|d| d.len()).collect(),
                    child_offsets_lens: child_offsets.iter().map(|o| o.len()).collect(),
                    child_counts: child_counts.clone(),
                }
            }
            NestedState::List { offsets, child_data, child_validity, child_data_position, child_offsets_position, total_elements, .. } => {
                NestedSnapshot::List {
                    offsets_len: offsets.len(),
                    child_data_len: child_data.len(),
                    child_validity_len: child_validity.len(),
                    child_data_position: *child_data_position,
                    child_offsets_position: *child_offsets_position,
                    total_elements: *total_elements,
                }
            }
            NestedState::Dictionary { indices, .. } => {
                NestedSnapshot::Dictionary {
                    indices_len: indices.len(),
                }
            }
            NestedState::Struct { field_data, field_null_bitmaps, field_validity, row_count, field_positions, field_offsets_positions, typed_storage, .. } => {
                NestedSnapshot::Struct {
                    field_data_lens: field_data.iter().map(|d| d.len()).collect(),
                    field_null_bitmap_lens: field_null_bitmaps.iter().map(|b| b.len()).collect(),
                    field_validity_lens: field_validity.iter().map(|v| v.len()).collect(),
                    row_count: *row_count,
                    field_positions: field_positions.clone(),
                    field_offsets_positions: field_offsets_positions.clone(),
                    typed_storage_lens: typed_storage.iter().map(|s| s.len()).collect(),
                }
            }
        }
    }

    /// Restore nested state from a snapshot
    fn restore_snapshot(&mut self, snapshot: &NestedSnapshot) {
        match (&mut self.nested, snapshot) {
            (NestedState::None, NestedSnapshot::None) => {}
            (
                NestedState::Map { map_offsets, key_offsets, key_data, value_offsets, value_data, value_validity, total_entries, .. },
                NestedSnapshot::Map {
                    map_offsets_len, key_offsets_len, key_data_len, value_offsets_len, value_data_len,
                    value_validity_len, total_entries: snap_entries,
                }
            ) => {
                map_offsets.truncate(*map_offsets_len);
                key_offsets.truncate(*key_offsets_len);
                key_data.truncate(*key_data_len);
                value_offsets.truncate(*value_offsets_len);
                value_data.truncate(*value_data_len);
                value_validity.truncate(*value_validity_len);
                *total_entries = *snap_entries;
            }
            (
                NestedState::Union { type_ids, offsets, child_data, child_offsets, child_counts, .. },
                NestedSnapshot::Union { type_ids_len, offsets_len, child_data_lens, child_offsets_lens, child_counts: snap_counts }
            ) => {
                type_ids.truncate(*type_ids_len);
                if let (Some(offs), Some(len)) = (offsets.as_mut(), offsets_len) {
                    offs.truncate(*len);
                }
                for (data, len) in child_data.iter_mut().zip(child_data_lens.iter()) {
                    data.truncate(*len);
                }
                for (offs, len) in child_offsets.iter_mut().zip(child_offsets_lens.iter()) {
                    offs.truncate(*len);
                }
                child_counts.clone_from(snap_counts);
            }
            (
                NestedState::List { offsets, child_data, child_validity, child_data_position, child_offsets_position, total_elements, .. },
                NestedSnapshot::List { offsets_len, child_data_len, child_validity_len, child_data_position: snap_position, child_offsets_position: snap_offsets_position, total_elements: snap_elements }
            ) => {
                offsets.truncate(*offsets_len);
                child_data.truncate(*child_data_len);
                child_validity.truncate(*child_validity_len);
                *child_data_position = *snap_position;
                *child_offsets_position = *snap_offsets_position;
                *total_elements = *snap_elements;
            }
            (
                NestedState::Dictionary { indices, .. },
                NestedSnapshot::Dictionary { indices_len }
            ) => {
                indices.truncate(*indices_len);
            }
            (
                NestedState::Struct { field_data, field_null_bitmaps, field_validity, row_count, field_positions, field_offsets_positions, typed_storage, .. },
                NestedSnapshot::Struct { field_data_lens, field_null_bitmap_lens, field_validity_lens, row_count: snap_count, field_positions: snap_positions, field_offsets_positions: snap_offsets_positions, typed_storage_lens }
            ) => {
                for (data, len) in field_data.iter_mut().zip(field_data_lens.iter()) {
                    data.truncate(*len);
                }
                for (bitmap, len) in field_null_bitmaps.iter_mut().zip(field_null_bitmap_lens.iter()) {
                    bitmap.truncate(*len);
                }
                for (validity, len) in field_validity.iter_mut().zip(field_validity_lens.iter()) {
                    validity.truncate(*len);
                }
                // Truncate typed storage
                for (storage, len) in typed_storage.iter_mut().zip(typed_storage_lens.iter()) {
                    storage.truncate(*len);
                }
                *row_count = *snap_count;
                *field_positions = snap_positions.clone();
                *field_offsets_positions = snap_offsets_positions.clone();
            }
            _ => {
                // Type mismatch - this shouldn't happen
            }
        }
    }

    /// Estimate element size in bytes for a type (used for capacity pre-sizing)
    /// Returns a reasonable estimate for variable-length types based on typical usage
    fn estimate_element_size(data_type: &DataType) -> usize {
        match data_type {
            DataType::Null => 0,
            DataType::Boolean => 1,
            DataType::Int8 | DataType::UInt8 => 1,
            DataType::Int16 | DataType::UInt16 => 2,
            DataType::Int32 | DataType::UInt32 | DataType::Date32 | DataType::Time32(_) => 4,
            DataType::Int64 | DataType::UInt64 | DataType::Date64 | DataType::Time64(_) | DataType::Timestamp(_, _) => 8,
            DataType::Float16 => 2,
            DataType::Float32 => 4,
            DataType::Float64 => 8,
            DataType::Decimal128(_, _) => 16,
            DataType::Decimal256(_, _) => 32,
            DataType::Interval(_) => 16,
            DataType::Duration(_) => 8,
            // Variable-length types: estimate 32 bytes avg
            DataType::Utf8 | DataType::LargeUtf8 => 32,
            DataType::Binary | DataType::LargeBinary => 32,
            DataType::FixedSizeBinary(size) => *size as usize,
            // Complex types: rough estimates
            DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _) => 64,
            DataType::Struct(_) => 64,
            DataType::Union(_, _) => 16,
            DataType::Dictionary(_, _) => 4, // Index size
            DataType::Map(_, _) => 64,
            _ => 16, // Default fallback
        }
    }
}

/// Pooled record decoder for zero-allocation Avro decoding
///
/// This decoder writes decoded Avro data directly into pre-allocated buffers
/// provided by a [`BufferProvider`], eliminating memory allocation during
/// the decoding hot path.
pub struct PooledRecordDecoder<P: BufferProvider> {
    /// Arrow schema for the output
    schema: SchemaRef,
    /// Per-column state
    columns: Vec<ColumnState>,
    /// Buffer provider for pre-allocated memory
    provider: P,
    /// Current row count across all columns
    row_count: usize,
    /// Maximum rows this decoder can handle
    max_rows: usize,
}

impl<P: BufferProvider> PooledRecordDecoder<P> {
    /// Create a new pooled decoder from an Arrow schema and buffer provider
    pub fn try_new(schema: SchemaRef, provider: P) -> Result<Self, ArrowError> {
        let max_rows = provider.max_rows();
        let columns = schema
            .fields()
            .iter()
            .enumerate()
            .map(|(idx, field)| {
                // Check if provider has child regions configured for this column
                // List needs child_index 0 for element data
                let use_preallocated = provider.has_child_region(idx, 0);
                ColumnState::new(idx, field.as_ref(), use_preallocated, max_rows)
            })
            .collect();

        let mut decoder = Self {
            schema,
            columns,
            provider,
            row_count: 0,
            max_rows,
        };

        // Initialize buffer offsets for nested types in Pooled mode
        decoder.initialize_nested_offsets()?;

        Ok(decoder)
    }

    /// Initialize buffer-based offsets for nested types in Pooled mode.
    /// Map and Union use Vec mode, so no buffer initialization needed for them.
    fn initialize_nested_offsets(&mut self) -> Result<(), ArrowError> {
        // Map and Union use Vec mode - no buffer initialization needed
        Ok(())
    }

    /// Create from an AvroDataType (for compatibility with existing API)
    #[allow(dead_code)]
    pub(crate) fn try_from_avro(data_type: &AvroDataType, provider: P) -> Result<Self, ArrowError> {
        match data_type.codec() {
            Codec::Struct(fields) => {
                let arrow_fields: Vec<FieldRef> =
                    fields.iter().map(|f| Arc::new(f.field())).collect();
                let schema = Arc::new(Schema::new(arrow_fields));
                Self::try_new(schema, provider)
            }
            other => Err(ArrowError::ParseError(format!(
                "Expected record got {other:?}"
            ))),
        }
    }

    /// Get the schema
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Get current row count
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// Get maximum rows this decoder supports
    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    /// Check if decoder can accept more rows
    pub fn has_capacity(&self) -> bool {
        self.row_count < self.max_rows
    }

    /// Check if schema is "flat" - all columns are fixed-size primitive types
    /// This enables fast batch decoding without per-value overhead
    fn is_flat_schema(&self) -> bool {
        self.columns.iter().all(|col| {
            matches!(
                col.data_type,
                DataType::Null
                    | DataType::Boolean
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::Float32
                    | DataType::Float64
                    | DataType::Date32
                    | DataType::Time32(_)
                    | DataType::Time64(_)
                    | DataType::Timestamp(_, _)
            ) && matches!(col.nested, NestedState::None)
        })
    }

    /// Get the size in bytes for a fixed-size type (None for variable-length)
    #[inline]
    fn get_type_size(data_type: &DataType) -> Option<usize> {
        match data_type {
            DataType::Null => Some(0),
            DataType::Boolean => Some(1), // 1 byte per boolean in packed format
            DataType::Int32 | DataType::Date32 | DataType::Time32(_) | DataType::Float32 => Some(4),
            DataType::Int64 | DataType::Time64(_) | DataType::Timestamp(_, _) | DataType::Float64 => Some(8),
            _ => None,
        }
    }

    /// Fast batch decode for flat schemas with only fixed-size types
    /// Returns bytes consumed from buffer
    ///
    /// For nullable fields (Avro unions like ['null', 'int']), reads the union index
    /// before each value and tracks null bits accordingly.
    fn decode_batch_flat(
        &mut self,
        buf: &[u8],
        count: usize,
    ) -> Result<usize, ArrowError> {
        let rows_to_decode = count.min(self.max_rows - self.row_count);
        if rows_to_decode == 0 {
            return Ok(0);
        }

        let num_columns = self.columns.len();

        // Build column metadata: (column_index, data_type, type_size, start_position, base_row_count, is_nullable)
        // We clone data_type to avoid borrowing issues
        let col_meta: Vec<(usize, DataType, usize, usize, usize, bool)> = self.columns
            .iter()
            .map(|col| {
                let type_size = Self::get_type_size(&col.data_type).unwrap_or(0);
                (col.column_index, col.data_type.clone(), type_size, col.data_position, col.row_count, col.is_nullable)
            })
            .collect();

        // Get raw pointers for all columns upfront - this avoids borrow checker issues
        // since column regions don't overlap
        let mut ptrs: Vec<Option<*mut u8>> = Vec::with_capacity(num_columns);
        for (column_index, data_type, _, start_pos, _, _) in &col_meta {
            if matches!(data_type, DataType::Null) {
                ptrs.push(None);
            } else if matches!(data_type, DataType::Boolean) {
                // Boolean uses bit-packed storage from the beginning of the region
                let (ptr, _) = self.provider.get_data_ptr(*column_index, 0)?;
                ptrs.push(Some(ptr));
            } else {
                let (ptr, _) = self.provider.get_data_ptr(*column_index, *start_pos)?;
                ptrs.push(Some(ptr));
            }
        }

        // Track validity for nullable columns during decode
        // validity_bits[col_idx][row] = true if value is valid (non-null)
        let mut validity_bits: Vec<Vec<bool>> = col_meta.iter()
            .map(|(_, _, _, _, _, is_nullable)| {
                if *is_nullable {
                    Vec::with_capacity(rows_to_decode)
                } else {
                    Vec::new()
                }
            })
            .collect();

        // Now decode in a tight loop using raw pointer writes
        let mut cursor = AvroCursor::new(buf);

        for row in 0..rows_to_decode {
            for col_idx in 0..num_columns {
                let (_, ref data_type, type_size, _, base_row_count, is_nullable) = col_meta[col_idx];

                // For nullable columns, read the Avro union index first
                // Index 0 = null, Index 1 (zigzag: 2) = non-null value
                let is_null = if is_nullable {
                    let union_index = cursor.get_long()?;
                    let is_valid = union_index != 0;
                    validity_bits[col_idx].push(is_valid);
                    !is_valid
                } else {
                    false
                };

                // Skip value reading if null
                if is_null {
                    // For null values, we still need to write something to the buffer
                    // (a default/zero value) to maintain alignment
                    if let Some(ptr) = ptrs[col_idx] {
                        match data_type {
                            DataType::Boolean => {
                                // Boolean bit is already 0 by default
                            }
                            DataType::Int32 | DataType::Date32 | DataType::Time32(_) | DataType::Float32 => {
                                unsafe {
                                    let dest = ptr.add(row * type_size);
                                    std::ptr::write_bytes(dest, 0, 4);
                                }
                            }
                            DataType::Int64 | DataType::Time64(_) | DataType::Timestamp(_, _) | DataType::Float64 => {
                                unsafe {
                                    let dest = ptr.add(row * type_size);
                                    std::ptr::write_bytes(dest, 0, 8);
                                }
                            }
                            _ => {}
                        }
                    }
                    continue;
                }

                match data_type {
                    DataType::Null => {
                        // Nothing to read or write
                    }
                    DataType::Boolean => {
                        let value = cursor.get_bool()?;
                        let base_row = base_row_count + row;
                        let byte_idx = base_row / 8;
                        let bit_idx = base_row % 8;
                        if let Some(ptr) = ptrs[col_idx] {
                            // SAFETY: byte_idx is within the allocated bitmap region
                            unsafe {
                                let byte_ptr = ptr.add(byte_idx);
                                if value {
                                    *byte_ptr |= 1 << bit_idx;
                                }
                            }
                        }
                    }
                    DataType::Int32 | DataType::Date32 | DataType::Time32(_) => {
                        let value = cursor.get_int()?;
                        if let Some(ptr) = ptrs[col_idx] {
                            // SAFETY: offset is within the allocated region
                            unsafe {
                                let dest = ptr.add(row * type_size);
                                std::ptr::copy_nonoverlapping(value.to_le_bytes().as_ptr(), dest, 4);
                            }
                        }
                    }
                    DataType::Int64 | DataType::Time64(_) | DataType::Timestamp(_, _) => {
                        let value = cursor.get_long()?;
                        if let Some(ptr) = ptrs[col_idx] {
                            // SAFETY: offset is within the allocated region
                            unsafe {
                                let dest = ptr.add(row * type_size);
                                std::ptr::copy_nonoverlapping(value.to_le_bytes().as_ptr(), dest, 8);
                            }
                        }
                    }
                    DataType::Float32 => {
                        let value = cursor.get_float()?;
                        if let Some(ptr) = ptrs[col_idx] {
                            // SAFETY: offset is within the allocated region
                            unsafe {
                                let dest = ptr.add(row * type_size);
                                std::ptr::copy_nonoverlapping(value.to_le_bytes().as_ptr(), dest, 4);
                            }
                        }
                    }
                    DataType::Float64 => {
                        let value = cursor.get_double()?;
                        if let Some(ptr) = ptrs[col_idx] {
                            // SAFETY: offset is within the allocated region
                            unsafe {
                                let dest = ptr.add(row * type_size);
                                std::ptr::copy_nonoverlapping(value.to_le_bytes().as_ptr(), dest, 8);
                            }
                        }
                    }
                    _ => {
                        // Should not happen for flat schema
                        return Err(ArrowError::InvalidArgumentError(
                            "Non-flat type in batch decode".to_string(),
                        ));
                    }
                }
            }
        }

        // Update column positions, counts, and null bitmaps
        for (col_idx, (_, _, type_size, _, base_row_count, is_nullable)) in col_meta.iter().enumerate() {
            self.columns[col_idx].data_position += rows_to_decode * type_size;
            self.columns[col_idx].row_count += rows_to_decode;

            // Update null bitmap for nullable columns
            if *is_nullable {
                for (row, is_valid) in validity_bits[col_idx].iter().enumerate() {
                    self.columns[col_idx].set_validity(base_row_count + row, *is_valid);
                }
            }
        }
        self.row_count += rows_to_decode;

        Ok(cursor.position())
    }

    /// Check if schema has only simple types (fixed + variable-length primitives, no nested)
    fn is_simple_schema(&self) -> bool {
        self.columns.iter().all(|col| {
            matches!(
                col.data_type,
                DataType::Null
                    | DataType::Boolean
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::Float32
                    | DataType::Float64
                    | DataType::Date32
                    | DataType::Time32(_)
                    | DataType::Time64(_)
                    | DataType::Timestamp(_, _)
                    | DataType::Utf8
                    | DataType::Binary
                    | DataType::FixedSizeBinary(_)
            ) && matches!(col.nested, NestedState::None)
        })
    }

    /// Check if schema has only string/binary columns (no fixed types)
    fn is_strings_only_schema(&self) -> bool {
        self.columns.iter().all(|col| {
            matches!(col.data_type, DataType::Utf8 | DataType::Binary)
                && matches!(col.nested, NestedState::None)
        })
    }

    /// Check if schema has only Dictionary columns (Avro enums)
    fn is_dictionary_only_schema(&self) -> bool {
        self.columns.iter().all(|col| {
            matches!(col.nested, NestedState::Dictionary { .. })
        })
    }

    /// Check if schema is a "simple nested" schema - can be batch processed
    /// This includes flat types + Dictionary (enum) types
    fn is_simple_nested_schema(&self) -> bool {
        self.columns.iter().all(|col| {
            match &col.nested {
                NestedState::None => matches!(
                    col.data_type,
                    DataType::Null
                        | DataType::Boolean
                        | DataType::Int32
                        | DataType::Int64
                        | DataType::Float32
                        | DataType::Float64
                        | DataType::Date32
                        | DataType::Time32(_)
                        | DataType::Time64(_)
                        | DataType::Timestamp(_, _)
                        | DataType::Utf8
                        | DataType::Binary
                        | DataType::FixedSizeBinary(_)
                ),
                NestedState::Dictionary { .. } => true,
                _ => false,
            }
        })
    }

    /// Check if this schema has only struct columns (uses Apache's Record decoder fast path)
    fn is_struct_only_schema(&self) -> bool {
        self.columns.iter().all(|col| matches!(col.nested, NestedState::Struct { .. }))
    }

    /// Fast path for struct-only schemas using Apache's Record decoder directly
    /// This bypasses all per-column dispatch overhead and matches Apache's exact code path
    #[inline(always)]
    fn decode_batch_struct(
        &mut self,
        buf: &[u8],
        count: usize,
    ) -> Result<usize, ArrowError> {
        let total_rows = count.min(self.max_rows - self.row_count);
        if total_rows == 0 {
            return Ok(0);
        }

        let mut cursor = AvroCursor::new(buf);

        // Tight decode loop - just call record_decoder.decode() for each row
        // No per-column dispatch, no ColKind matching, no field_nullable_flags extraction
        for _ in 0..total_rows {
            for col in self.columns.iter_mut() {
                // Handle column-level nullable (the struct itself being nullable)
                if col.is_nullable {
                    let union_index = cursor.get_long()?;
                    col.set_validity(col.row_count, union_index != 0);
                    if union_index == 0 {
                        // Null struct - append nulls to record_decoder
                        if let NestedState::Struct { row_count, record_decoder, .. } = &mut col.nested {
                            record_decoder.append_null()?;
                            *row_count += 1;
                        }
                        col.row_count += 1;
                        continue;
                    }
                }
                // Non-null struct - decode using Apache's Record decoder
                if let NestedState::Struct { row_count, record_decoder, .. } = &mut col.nested {
                    record_decoder.decode(&mut cursor)?;
                    *row_count += 1;
                }
                col.row_count += 1;
            }
        }

        self.row_count += total_rows;
        Ok(cursor.position())
    }

    /// Batch decode for Dictionary-only schemas (Avro enums)
    /// Enums are just varints, so this is very fast
    #[inline(always)]
    fn decode_batch_dictionary_only(
        &mut self,
        buf: &[u8],
        count: usize,
    ) -> Result<usize, ArrowError> {
        let total_rows = count.min(self.max_rows - self.row_count);
        if total_rows == 0 {
            return Ok(0);
        }

        let num_columns = self.columns.len();
        let mut cursor = AvroCursor::new(buf);

        // Tight decode loop for enums
        for _ in 0..total_rows {
            for col_idx in 0..num_columns {
                let index = cursor.get_int()?;
                if let NestedState::Dictionary { indices, .. } = &mut self.columns[col_idx].nested {
                    indices.push(index);
                }
                self.columns[col_idx].row_count += 1;
            }
        }

        self.row_count += total_rows;
        Ok(cursor.position())
    }

    /// Batch decode for "simple nested" schemas - flat types + Dictionary
    /// Uses raw pointers for fixed types and Vec for dictionary indices
    #[inline(always)]
    fn decode_batch_simple_nested(
        &mut self,
        buf: &[u8],
        count: usize,
    ) -> Result<usize, ArrowError> {
        let total_rows = count.min(self.max_rows - self.row_count);
        if total_rows == 0 {
            return Ok(0);
        }

        let num_columns = self.columns.len();
        let mut cursor = AvroCursor::new(buf);

        // Classify columns once - distinguish varint vs fixed encoding
        #[derive(Clone, Copy)]
        enum ColKind {
            Int32,       // Varint -> 4 bytes output
            Int64,       // Varint -> 8 bytes output
            Float32,     // Fixed 4 bytes
            Float64,     // Fixed 8 bytes
            Boolean,     // 1 byte -> 4 bytes output (for BooleanArray stored as bytes)
            FixedBytes { size: usize }, // Fixed-size binary
            String,
            Binary,
            Dictionary,
            Null,
        }

        let mut col_kinds: Vec<ColKind> = Vec::with_capacity(num_columns);
        for col in &self.columns {
            let kind = match &col.nested {
                NestedState::Dictionary { .. } => ColKind::Dictionary,
                NestedState::None => match &col.data_type {
                    DataType::Boolean => ColKind::Boolean,
                    DataType::Int32 | DataType::Date32 | DataType::Time32(_) => ColKind::Int32,
                    DataType::Int64 | DataType::Time64(_) | DataType::Timestamp(_, _) => ColKind::Int64,
                    DataType::Float32 => ColKind::Float32,
                    DataType::Float64 => ColKind::Float64,
                    DataType::Utf8 => ColKind::String,
                    DataType::Binary => ColKind::Binary,
                    DataType::FixedSizeBinary(n) => ColKind::FixedBytes { size: *n as usize },
                    DataType::Null => ColKind::Null,
                    _ => ColKind::Int64, // Fallback to varint
                },
                _ => return Err(ArrowError::InvalidArgumentError(
                    "decode_batch_simple_nested called with unsupported nested type".to_string(),
                )),
            };
            col_kinds.push(kind);
        }

        // Pre-compute column indices
        let column_indices: Vec<usize> = self.columns.iter().map(|c| c.column_index).collect();

        // Process in mini-batches
        let mut rows_processed = 0;
        while rows_processed < total_rows {
            let batch_rows = (total_rows - rows_processed).min(Self::BATCH_SIZE);

            // Get pointers for fixed and string columns
            let mut data_ptrs: Vec<Option<*mut u8>> = Vec::with_capacity(num_columns);
            let mut offsets_ptrs: Vec<Option<*mut u8>> = Vec::with_capacity(num_columns);
            let mut data_capacities: Vec<usize> = Vec::with_capacity(num_columns);
            let mut base_data_positions: Vec<usize> = Vec::with_capacity(num_columns);

            for (col_idx, &column_index) in column_indices.iter().enumerate() {
                match col_kinds[col_idx] {
                    ColKind::Boolean => {
                        // Boolean uses bit-packed storage from beginning of region
                        let col = &self.columns[col_idx];
                        let (data_ptr, cap) = self.provider.get_data_ptr(column_index, 0)?;
                        data_ptrs.push(Some(data_ptr));
                        offsets_ptrs.push(None);
                        data_capacities.push(cap);
                        base_data_positions.push(col.row_count); // Store base row count for bit indexing
                    }
                    ColKind::Int32 | ColKind::Int64 | ColKind::Float32 | ColKind::Float64
                    | ColKind::FixedBytes { .. } => {
                        let col = &self.columns[col_idx];
                        let (data_ptr, cap) = self.provider.get_data_ptr(column_index, col.data_position)?;
                        data_ptrs.push(Some(data_ptr));
                        offsets_ptrs.push(None);
                        data_capacities.push(cap);
                        base_data_positions.push(0); // Not used for fixed types
                    }
                    ColKind::String | ColKind::Binary => {
                        let col = &self.columns[col_idx];
                        let (data_ptr, cap) = self.provider.get_data_ptr(column_index, col.data_position)?;
                        let offsets_pos = (col.row_count + 1) * 4;
                        let offsets_slice = self.provider.get_offsets_slice(column_index, offsets_pos, batch_rows * 4)?;
                        data_ptrs.push(Some(data_ptr));
                        offsets_ptrs.push(Some(offsets_slice.as_mut_ptr()));
                        data_capacities.push(cap);
                        base_data_positions.push(col.data_position); // Track base position for offset calculation
                    }
                    ColKind::Dictionary | ColKind::Null => {
                        data_ptrs.push(None);
                        offsets_ptrs.push(None);
                        data_capacities.push(0);
                        base_data_positions.push(0);
                    }
                }
            }

            // Track positions within this batch
            let mut fixed_offsets: Vec<usize> = vec![0; num_columns];
            let mut var_data_offsets: Vec<usize> = vec![0; num_columns];

            // Decode batch
            for row in 0..batch_rows {
                for col_idx in 0..num_columns {
                    match col_kinds[col_idx] {
                        ColKind::Null => {
                            // Skip - nothing to read or write
                        }
                        ColKind::Boolean => {
                            let val = cursor.get_bool()?;
                            // Bit-packed boolean storage
                            let base_row = base_data_positions[col_idx] + row;
                            let byte_idx = base_row / 8;
                            let bit_idx = base_row % 8;
                            unsafe {
                                let byte_ptr = data_ptrs[col_idx].unwrap().add(byte_idx);
                                if val {
                                    *byte_ptr |= 1 << bit_idx;
                                }
                            }
                        }
                        ColKind::Int32 => {
                            let val = cursor.get_int()?;
                            let offset = fixed_offsets[col_idx];
                            unsafe {
                                let dest = data_ptrs[col_idx].unwrap().add(offset);
                                std::ptr::copy_nonoverlapping(val.to_le_bytes().as_ptr(), dest, 4);
                            }
                            fixed_offsets[col_idx] += 4;
                        }
                        ColKind::Int64 => {
                            let val = cursor.get_long()?;
                            let offset = fixed_offsets[col_idx];
                            unsafe {
                                let dest = data_ptrs[col_idx].unwrap().add(offset);
                                std::ptr::copy_nonoverlapping(val.to_le_bytes().as_ptr(), dest, 8);
                            }
                            fixed_offsets[col_idx] += 8;
                        }
                        ColKind::Float32 => {
                            let bytes = cursor.get_fixed(4)?;
                            let offset = fixed_offsets[col_idx];
                            unsafe {
                                let dest = data_ptrs[col_idx].unwrap().add(offset);
                                std::ptr::copy_nonoverlapping(bytes.as_ptr(), dest, 4);
                            }
                            fixed_offsets[col_idx] += 4;
                        }
                        ColKind::Float64 => {
                            let bytes = cursor.get_fixed(8)?;
                            let offset = fixed_offsets[col_idx];
                            unsafe {
                                let dest = data_ptrs[col_idx].unwrap().add(offset);
                                std::ptr::copy_nonoverlapping(bytes.as_ptr(), dest, 8);
                            }
                            fixed_offsets[col_idx] += 8;
                        }
                        ColKind::FixedBytes { size } => {
                            let bytes = cursor.get_fixed(size)?;
                            let offset = fixed_offsets[col_idx];
                            unsafe {
                                let dest = data_ptrs[col_idx].unwrap().add(offset);
                                std::ptr::copy_nonoverlapping(bytes.as_ptr(), dest, size);
                            }
                            fixed_offsets[col_idx] += size;
                        }
                        ColKind::String | ColKind::Binary => {
                            let bytes = cursor.get_bytes()?;
                            let len = bytes.len();
                            let current_offset = var_data_offsets[col_idx];

                            if current_offset + len > data_capacities[col_idx] {
                                return Err(ArrowError::InvalidArgumentError(format!(
                                    "Capacity exceeded for column {}", col_idx
                                )));
                            }

                            // Write data
                            unsafe {
                                let dest = data_ptrs[col_idx].unwrap().add(current_offset);
                                std::ptr::copy_nonoverlapping(bytes.as_ptr(), dest, len);
                            }

                            // Write offset - must be absolute position (base + batch_offset + len)
                            let new_end = (base_data_positions[col_idx] + current_offset + len) as i32;
                            unsafe {
                                let offsets_ptr = offsets_ptrs[col_idx].unwrap().add(row * 4);
                                std::ptr::copy_nonoverlapping(new_end.to_le_bytes().as_ptr(), offsets_ptr, 4);
                            }
                            var_data_offsets[col_idx] = current_offset + len;
                        }
                        ColKind::Dictionary => {
                            let index = cursor.get_int()?;
                            if let NestedState::Dictionary { indices, .. } = &mut self.columns[col_idx].nested {
                                indices.push(index);
                            }
                        }
                    }
                }
            }

            // Update positions and row_count AFTER processing entire batch
            for col_idx in 0..num_columns {
                match col_kinds[col_idx] {
                    ColKind::Int32 | ColKind::Float32 => {
                        self.columns[col_idx].data_position += 4 * batch_rows;
                    }
                    ColKind::Int64 | ColKind::Float64 => {
                        self.columns[col_idx].data_position += 8 * batch_rows;
                    }
                    ColKind::Boolean => {
                        // Boolean uses bit-packing, no data_position update needed
                    }
                    ColKind::FixedBytes { size } => {
                        self.columns[col_idx].data_position += size * batch_rows;
                    }
                    ColKind::String | ColKind::Binary => {
                        self.columns[col_idx].data_position += var_data_offsets[col_idx];
                    }
                    ColKind::Dictionary | ColKind::Null => {}
                }
                self.columns[col_idx].row_count += batch_rows;
            }

            rows_processed += batch_rows;
        }

        self.row_count += total_rows;
        Ok(cursor.position())
    }

    /// Universal batch decode for schemas with complex nested types (List, Map, Union, Struct)
    /// Uses pre-fetched pointers and inlined complex type decoding for optimal performance.
    #[inline(always)]
    fn decode_batch_universal(
        &mut self,
        buf: &[u8],
        count: usize,
    ) -> Result<usize, ArrowError> {
        let total_rows = count.min(self.max_rows - self.row_count);
        if total_rows == 0 {
            return Ok(0);
        }

        let num_columns = self.columns.len();
        let mut cursor = AvroCursor::new(buf);

        // Pre-classify column types for efficient dispatch
        #[derive(Clone)]
        enum ColKind {
            Null,
            Boolean,
            Int32,
            Int64,
            Float32,
            Float64,
            String,
            Binary,
            FixedBytes { size: usize },
            Decimal128,
            Dictionary,
            List { child_type: DataType, fixed_elem_size: Option<usize>, use_preallocated: bool },
            Map { value_type: DataType },
            Union { num_variants: usize, field_type_ids: Vec<i8>, child_types: Vec<DataType> },
            Struct { _field_types: Vec<DataType>, use_preallocated: bool },
        }

        // Pre-extract type info from nested state to avoid repeated pattern matching
        let col_kinds: Vec<ColKind> = self.columns.iter().map(|col| {
            match &col.nested {
                NestedState::Dictionary { .. } => ColKind::Dictionary,
                NestedState::List { child_type, use_preallocated, .. } => {
                    let fixed_elem_size = Self::get_fixed_element_size(child_type);
                    ColKind::List { child_type: child_type.clone(), fixed_elem_size, use_preallocated: *use_preallocated }
                }
                NestedState::Map { value_type, .. } => {
                    ColKind::Map { value_type: value_type.clone() }
                }
                NestedState::Union { fields, .. } => {
                    let field_type_ids: Vec<i8> = fields.iter().map(|(id, _)| id).collect();
                    let child_types: Vec<DataType> = fields.iter().map(|(_, f)| f.data_type().clone()).collect();
                    ColKind::Union { num_variants: fields.len(), field_type_ids, child_types }
                }
                NestedState::Struct { field_types, use_preallocated, .. } => ColKind::Struct { _field_types: field_types.clone(), use_preallocated: *use_preallocated },
                NestedState::None => match &col.data_type {
                    DataType::Null => ColKind::Null,
                    DataType::Boolean => ColKind::Boolean,
                    DataType::Int32 | DataType::Date32 | DataType::Time32(_) => ColKind::Int32,
                    DataType::Int64 | DataType::Time64(_) | DataType::Timestamp(_, _) => ColKind::Int64,
                    DataType::Float32 => ColKind::Float32,
                    DataType::Float64 => ColKind::Float64,
                    DataType::Utf8 => ColKind::String,
                    DataType::Binary => ColKind::Binary,
                    DataType::FixedSizeBinary(n) => ColKind::FixedBytes { size: *n as usize },
                    DataType::Decimal128(_, _) => ColKind::Decimal128,
                    _ => ColKind::Int64, // Fallback
                },
            }
        }).collect();

        // Pre-compute column indices and nullability
        let column_indices: Vec<usize> = self.columns.iter().map(|c| c.column_index).collect();
        let nullable_flags: Vec<bool> = self.columns.iter().map(|c| c.is_nullable).collect();

        // Process in mini-batches for better cache efficiency
        let mut rows_processed = 0;
        while rows_processed < total_rows {
            let batch_rows = (total_rows - rows_processed).min(Self::BATCH_SIZE);

            // Pre-fetch pointers for primitive columns in this batch
            let mut data_ptrs: Vec<Option<*mut u8>> = Vec::with_capacity(num_columns);
            let mut offsets_ptrs: Vec<Option<*mut u8>> = Vec::with_capacity(num_columns);
            let mut data_capacities: Vec<usize> = Vec::with_capacity(num_columns);
            let mut base_data_positions: Vec<usize> = Vec::with_capacity(num_columns);
            let mut base_row_counts: Vec<usize> = Vec::with_capacity(num_columns);

            for (col_idx, &column_index) in column_indices.iter().enumerate() {
                match &col_kinds[col_idx] {
                    ColKind::Boolean => {
                        let (data_ptr, cap) = self.provider.get_data_ptr(column_index, 0)?;
                        data_ptrs.push(Some(data_ptr));
                        offsets_ptrs.push(None);
                        data_capacities.push(cap);
                        base_data_positions.push(0);
                        base_row_counts.push(self.columns[col_idx].row_count);
                    }
                    ColKind::Int32 | ColKind::Int64 | ColKind::Float32 | ColKind::Float64
                    | ColKind::FixedBytes { .. } | ColKind::Decimal128 => {
                        let col = &self.columns[col_idx];
                        let (data_ptr, cap) = self.provider.get_data_ptr(column_index, col.data_position)?;
                        data_ptrs.push(Some(data_ptr));
                        offsets_ptrs.push(None);
                        data_capacities.push(cap);
                        base_data_positions.push(0);
                        base_row_counts.push(col.row_count);
                    }
                    ColKind::String | ColKind::Binary => {
                        let col = &self.columns[col_idx];
                        let (data_ptr, cap) = self.provider.get_data_ptr(column_index, col.data_position)?;
                        let offsets_pos = (col.row_count + 1) * 4;
                        let offsets_slice = self.provider.get_offsets_slice(column_index, offsets_pos, batch_rows * 4)?;
                        data_ptrs.push(Some(data_ptr));
                        offsets_ptrs.push(Some(offsets_slice.as_mut_ptr()));
                        data_capacities.push(cap);
                        base_data_positions.push(col.data_position);
                        base_row_counts.push(col.row_count);
                    }
                    ColKind::Null | ColKind::Dictionary | ColKind::List { .. } | ColKind::Map { .. }
                    | ColKind::Union { .. } | ColKind::Struct { .. } => {
                        data_ptrs.push(None);
                        offsets_ptrs.push(None);
                        data_capacities.push(0);
                        base_data_positions.push(0);
                        base_row_counts.push(self.columns[col_idx].row_count);
                    }
                }
            }

            // Track offsets within this batch for primitive columns
            let mut fixed_offsets: Vec<usize> = vec![0; num_columns];
            let mut var_data_offsets: Vec<usize> = vec![0; num_columns];

            // Track validity for nullable columns during decode
            let mut validity_bits: Vec<Vec<bool>> = nullable_flags.iter()
                .map(|&is_nullable| {
                    if is_nullable {
                        Vec::with_capacity(batch_rows)
                    } else {
                        Vec::new()
                    }
                })
                .collect();

            // Tight decode loop
            for row in 0..batch_rows {
                for col_idx in 0..num_columns {
                    let is_nullable = nullable_flags[col_idx];

                    // For nullable columns, read the Avro union index first
                    let is_null = if is_nullable {
                        let union_index = cursor.get_long()?;
                        let is_valid = union_index != 0;
                        validity_bits[col_idx].push(is_valid);
                        !is_valid
                    } else {
                        false
                    };

                    // Handle null values by writing placeholder and skipping value read
                    if is_null {
                        match &col_kinds[col_idx] {
                            ColKind::Null | ColKind::Boolean => {}
                            ColKind::Int32 | ColKind::Float32 => {
                                if let Some(ptr) = data_ptrs[col_idx] {
                                    unsafe {
                                        let dest = ptr.add(fixed_offsets[col_idx]);
                                        std::ptr::write_bytes(dest, 0, 4);
                                    }
                                }
                                fixed_offsets[col_idx] += 4;
                            }
                            ColKind::Int64 | ColKind::Float64 => {
                                if let Some(ptr) = data_ptrs[col_idx] {
                                    unsafe {
                                        let dest = ptr.add(fixed_offsets[col_idx]);
                                        std::ptr::write_bytes(dest, 0, 8);
                                    }
                                }
                                fixed_offsets[col_idx] += 8;
                            }
                            ColKind::FixedBytes { size } => {
                                if let Some(ptr) = data_ptrs[col_idx] {
                                    unsafe {
                                        let dest = ptr.add(fixed_offsets[col_idx]);
                                        std::ptr::write_bytes(dest, 0, *size);
                                    }
                                }
                                fixed_offsets[col_idx] += size;
                            }
                            ColKind::Decimal128 => {
                                // Null Decimal128: write 16 zero bytes
                                if let Some(ptr) = data_ptrs[col_idx] {
                                    unsafe {
                                        let dest = ptr.add(fixed_offsets[col_idx]);
                                        std::ptr::write_bytes(dest, 0, 16);
                                    }
                                }
                                fixed_offsets[col_idx] += 16;
                            }
                            ColKind::String | ColKind::Binary => {
                                // Write same offset as previous (empty string for null)
                                let current_offset = var_data_offsets[col_idx];
                                let new_end = (base_data_positions[col_idx] + current_offset) as i32;
                                if let Some(ptr) = offsets_ptrs[col_idx] {
                                    unsafe {
                                        let dest = ptr.add(row * 4);
                                        std::ptr::copy_nonoverlapping(new_end.to_le_bytes().as_ptr(), dest, 4);
                                    }
                                }
                            }
                            ColKind::Dictionary => {
                                // Push a placeholder index for null dictionary value
                                if let NestedState::Dictionary { indices, .. } = &mut self.columns[col_idx].nested {
                                    indices.push(0);
                                }
                                self.columns[col_idx].row_count += 1;
                            }
                            ColKind::List { .. } => {
                                // Null List: add offset for empty list (same position as current)
                                if let NestedState::List { offsets, total_elements, .. } = &mut self.columns[col_idx].nested {
                                    offsets.push(*total_elements as i32);
                                }
                                self.columns[col_idx].row_count += 1;
                            }
                            ColKind::Map { .. } => {
                                // Null Map: add offset for empty map (same position as current)
                                if let NestedState::Map { map_offsets, total_entries, .. } = &mut self.columns[col_idx].nested {
                                    map_offsets.push(*total_entries as i32);
                                }
                                self.columns[col_idx].row_count += 1;
                            }
                            ColKind::Struct { .. } => {
                                // Null Struct: add placeholder row with null/empty field values
                                if let NestedState::Struct { field_data, field_types, field_nullable, field_validity, row_count, .. } = &mut self.columns[col_idx].nested {
                                    // Add placeholder data for each field
                                    for (field_idx, field_type) in field_types.iter().enumerate() {
                                        if field_nullable[field_idx] {
                                            // Nullable field: track validity separately and add placeholder
                                            field_validity[field_idx].push(false); // null
                                            Self::append_placeholder_to_vec(field_type, &mut field_data[field_idx]);
                                        } else {
                                            // Non-nullable field: push placeholder value
                                            Self::append_placeholder_to_vec(field_type, &mut field_data[field_idx]);
                                        }
                                    }
                                    *row_count += 1;
                                }
                                self.columns[col_idx].row_count += 1;
                            }
                            ColKind::Union { .. } => {
                                // Null Union: add placeholder
                                if let NestedState::Union { type_ids, offsets, child_counts, .. } = &mut self.columns[col_idx].nested {
                                    // Use type_id 0 for null union value
                                    type_ids.push(0);
                                    if let Some(offs) = offsets {
                                        offs.push(child_counts.first().copied().unwrap_or(0) as i32);
                                    }
                                }
                                self.columns[col_idx].row_count += 1;
                            }
                        }
                        continue;
                    }

                    match &col_kinds[col_idx] {
                        ColKind::Null => {}
                        ColKind::Boolean => {
                            let val = cursor.get_bool()?;
                            let base_row = base_row_counts[col_idx] + row;
                            let byte_idx = base_row / 8;
                            let bit_idx = base_row % 8;
                            unsafe {
                                let byte_ptr = data_ptrs[col_idx].unwrap().add(byte_idx);
                                if val { *byte_ptr |= 1 << bit_idx; }
                            }
                        }
                        ColKind::Int32 => {
                            let val = cursor.get_int()?;
                            let offset = fixed_offsets[col_idx];
                            unsafe {
                                let dest = data_ptrs[col_idx].unwrap().add(offset);
                                std::ptr::copy_nonoverlapping(val.to_le_bytes().as_ptr(), dest, 4);
                            }
                            fixed_offsets[col_idx] += 4;
                        }
                        ColKind::Int64 => {
                            let val = cursor.get_long()?;
                            let offset = fixed_offsets[col_idx];
                            unsafe {
                                let dest = data_ptrs[col_idx].unwrap().add(offset);
                                std::ptr::copy_nonoverlapping(val.to_le_bytes().as_ptr(), dest, 8);
                            }
                            fixed_offsets[col_idx] += 8;
                        }
                        ColKind::Float32 => {
                            let bytes = cursor.get_fixed(4)?;
                            let offset = fixed_offsets[col_idx];
                            unsafe {
                                let dest = data_ptrs[col_idx].unwrap().add(offset);
                                std::ptr::copy_nonoverlapping(bytes.as_ptr(), dest, 4);
                            }
                            fixed_offsets[col_idx] += 4;
                        }
                        ColKind::Float64 => {
                            let bytes = cursor.get_fixed(8)?;
                            let offset = fixed_offsets[col_idx];
                            unsafe {
                                let dest = data_ptrs[col_idx].unwrap().add(offset);
                                std::ptr::copy_nonoverlapping(bytes.as_ptr(), dest, 8);
                            }
                            fixed_offsets[col_idx] += 8;
                        }
                        ColKind::FixedBytes { size } => {
                            let bytes = cursor.get_fixed(*size)?;
                            let offset = fixed_offsets[col_idx];
                            unsafe {
                                let dest = data_ptrs[col_idx].unwrap().add(offset);
                                std::ptr::copy_nonoverlapping(bytes.as_ptr(), dest, *size);
                            }
                            fixed_offsets[col_idx] += size;
                        }
                        ColKind::Decimal128 => {
                            // Read length-prefixed bytes and convert to i128
                            let bytes = cursor.get_bytes()?;
                            let value = decode_decimal_bytes_to_i128(bytes);
                            let offset = fixed_offsets[col_idx];
                            unsafe {
                                let dest = data_ptrs[col_idx].unwrap().add(offset);
                                std::ptr::copy_nonoverlapping(value.to_le_bytes().as_ptr(), dest, 16);
                            }
                            fixed_offsets[col_idx] += 16;
                        }
                        ColKind::String | ColKind::Binary => {
                            let bytes = cursor.get_bytes()?;
                            let len = bytes.len();
                            let current_offset = var_data_offsets[col_idx];
                            if current_offset + len > data_capacities[col_idx] {
                                return Err(ArrowError::InvalidArgumentError(format!(
                                    "Capacity exceeded for column {}", col_idx
                                )));
                            }
                            unsafe {
                                let dest = data_ptrs[col_idx].unwrap().add(current_offset);
                                std::ptr::copy_nonoverlapping(bytes.as_ptr(), dest, len);
                            }
                            let new_end = (base_data_positions[col_idx] + current_offset + len) as i32;
                            unsafe {
                                let offsets_ptr = offsets_ptrs[col_idx].unwrap().add(row * 4);
                                std::ptr::copy_nonoverlapping(new_end.to_le_bytes().as_ptr(), offsets_ptr, 4);
                            }
                            var_data_offsets[col_idx] = current_offset + len;
                        }
                        ColKind::Dictionary => {
                            // Inlined dictionary decode
                            let index = cursor.get_int()?;
                            if let NestedState::Dictionary { indices, .. } = &mut self.columns[col_idx].nested {
                                indices.push(index);
                            }
                            self.columns[col_idx].row_count += 1;
                        }
                        ColKind::List { child_type, fixed_elem_size, use_preallocated } => {
                            // Extract child_nullable flag from nested state
                            let is_child_nullable = if let NestedState::List { child_nullable, .. } = &self.columns[col_idx].nested {
                                *child_nullable
                            } else {
                                false
                            };

                            // Inlined list decode - read blocks until 0 count
                            let mut elements_in_row = 0usize;
                            loop {
                                let block_count = cursor.get_long()?;
                                if block_count == 0 { break; }
                                let count = if block_count < 0 {
                                    let _size = cursor.get_long()?;
                                    (-block_count) as usize
                                } else {
                                    block_count as usize
                                };

                                if *use_preallocated && !is_child_nullable {
                                    // Pre-allocated path: decode directly into provider's child region
                                    // Note: pre-allocated path doesn't support nullable children yet
                                    let provider_col_idx = column_indices[col_idx];
                                    if let NestedState::List { child_data_position, child_offsets_position, .. } = &mut self.columns[col_idx].nested {
                                        if let Some(elem_size) = fixed_elem_size {
                                            // Fixed-size elements directly into pre-allocated buffer
                                            let needed = count * elem_size;
                                            let pos = *child_data_position;
                                            let slice = self.provider.get_child_data_slice(provider_col_idx, 0, pos, needed)?;
                                            Self::decode_fixed_elements_batch(child_type, &mut cursor, slice, count, *elem_size)?;
                                            *child_data_position += needed;
                                        } else {
                                            // Variable-length elements (Utf8/Binary) - need to write both data and offsets
                                            for _ in 0..count {
                                                let bytes = cursor.get_bytes()?;

                                                // Write string data
                                                let data_pos = *child_data_position;
                                                let data_slice = self.provider.get_child_data_slice(provider_col_idx, 0, data_pos, bytes.len())?;
                                                data_slice.copy_from_slice(bytes);
                                                *child_data_position += bytes.len();

                                                // Write offset (current position after writing this element)
                                                let off_pos = *child_offsets_position;
                                                let off_slice = self.provider.get_child_offsets_slice(provider_col_idx, 0, off_pos, 4)?;
                                                off_slice.copy_from_slice(&(*child_data_position as i32).to_le_bytes());
                                                *child_offsets_position += 4;
                                            }
                                        }
                                    }
                                } else {
                                    // Vec fallback path (also used for nullable children)
                                    if let NestedState::List { child_data, child_validity, .. } = &mut self.columns[col_idx].nested {
                                        if is_child_nullable {
                                            // Nullable elements: read union index, track validity separately
                                            for _ in 0..count {
                                                let union_index = cursor.get_long()?;
                                                if union_index == 0 {
                                                    child_validity.push(false); // null
                                                    Self::append_placeholder_to_vec(child_type, child_data);
                                                } else {
                                                    child_validity.push(true); // valid
                                                    Self::decode_value_to_vec(child_type, &mut cursor, child_data)?;
                                                }
                                            }
                                        } else if let Some(elem_size) = fixed_elem_size {
                                            // Fast path: fixed-size elements
                                            let old_len = child_data.len();
                                            child_data.reserve(count * elem_size);
                                            unsafe { child_data.set_len(old_len + count * elem_size); }
                                            Self::decode_fixed_elements_batch(child_type, &mut cursor, &mut child_data[old_len..], count, *elem_size)?;
                                        } else {
                                            // Variable-length elements
                                            for _ in 0..count {
                                                Self::decode_value_to_vec(child_type, &mut cursor, child_data)?;
                                            }
                                        }
                                    }
                                }
                                elements_in_row += count;
                            }
                            // Update offsets
                            if let NestedState::List { offsets, total_elements, .. } = &mut self.columns[col_idx].nested {
                                *total_elements += elements_in_row;
                                offsets.push(*total_elements as i32);
                            }
                            self.columns[col_idx].row_count += 1;
                        }
                        ColKind::Map { value_type, .. } => {
                            // Map decode using Vec mode (more efficient for nested types)
                            // Get value_nullable flag from nested state
                            let is_value_nullable = if let NestedState::Map { value_nullable, .. } = &self.columns[col_idx].nested {
                                *value_nullable
                            } else {
                                false
                            };

                            let mut entries_in_row = 0usize;
                            loop {
                                let block_count = cursor.get_long()?;
                                if block_count == 0 { break; }
                                let count = if block_count < 0 {
                                    let _size = cursor.get_long()?;
                                    (-block_count) as usize
                                } else {
                                    block_count as usize
                                };

                                if let NestedState::Map { key_offsets, key_data, value_offsets, value_data, value_validity, .. } = &mut self.columns[col_idx].nested {
                                    let needs_value_offsets = matches!(value_type, DataType::Utf8 | DataType::Binary);
                                    for _ in 0..count {
                                        let key_bytes = cursor.get_bytes()?;
                                        key_data.extend_from_slice(key_bytes);
                                        key_offsets.push(key_data.len() as i32);
                                        // Handle nullable values with separate validity tracking
                                        if is_value_nullable {
                                            let union_index = cursor.get_long()?;
                                            if union_index == 0 {
                                                value_validity.push(false); // null
                                                Self::append_placeholder_to_vec(value_type, value_data);
                                            } else {
                                                value_validity.push(true); // valid
                                                Self::decode_value_to_vec(value_type, &mut cursor, value_data)?;
                                            }
                                        } else {
                                            Self::decode_value_to_vec(value_type, &mut cursor, value_data)?;
                                        }
                                        // Track offsets for variable-length value types
                                        if needs_value_offsets {
                                            value_offsets.push(value_data.len() as i32);
                                        }
                                    }
                                }
                                entries_in_row += count;
                            }
                            if let NestedState::Map { map_offsets, total_entries, .. } = &mut self.columns[col_idx].nested {
                                *total_entries += entries_in_row;
                                map_offsets.push(*total_entries as i32);
                            }
                            self.columns[col_idx].row_count += 1;
                        }
                        ColKind::Union { num_variants, field_type_ids, child_types } => {
                            // Union decode using Vec mode with O(1) lookups
                            let type_index = cursor.get_long()? as usize;
                            if type_index >= *num_variants {
                                return Err(ArrowError::ParseError(format!(
                                    "Union type index {} out of bounds", type_index
                                )));
                            }

                            // O(1) lookup from pre-extracted arrays
                            let field_type_id = field_type_ids[type_index];
                            let child_type = &child_types[type_index];

                            if let NestedState::Union { type_ids, offsets, child_data, child_counts, .. } = &mut self.columns[col_idx].nested {
                                type_ids.push(field_type_id);
                                if let Some(offs) = offsets {
                                    offs.push(child_counts[type_index] as i32);
                                }
                                // Inline fast paths for common fixed-size types
                                let child_vec = &mut child_data[type_index];
                                match child_type {
                                    DataType::Null => {
                                        // Null takes no space
                                    }
                                    DataType::Int32 | DataType::Date32 | DataType::Time32(_) => {
                                        let value = cursor.get_int()?;
                                        child_vec.extend_from_slice(&value.to_le_bytes());
                                    }
                                    DataType::Int64 | DataType::Time64(_) | DataType::Timestamp(_, _) => {
                                        let value = cursor.get_long()?;
                                        child_vec.extend_from_slice(&value.to_le_bytes());
                                    }
                                    DataType::Float32 => {
                                        let value = cursor.get_float()?;
                                        child_vec.extend_from_slice(&value.to_le_bytes());
                                    }
                                    DataType::Float64 => {
                                        let value = cursor.get_double()?;
                                        child_vec.extend_from_slice(&value.to_le_bytes());
                                    }
                                    DataType::Boolean => {
                                        let value = cursor.get_bool()?;
                                        child_vec.push(if value { 1 } else { 0 });
                                    }
                                    other => {
                                        // Fall back to generic decode for complex types
                                        Self::decode_value_to_vec(other, &mut cursor, child_vec)?;
                                    }
                                }
                                child_counts[type_index] += 1;
                            }
                            self.columns[col_idx].row_count += 1;
                        }
                        ColKind::Struct { _field_types: _, use_preallocated } => {
                            // Fast path: use record_decoder which handles nullable fields internally
                            // This skips all per-row overhead from field_nullable extraction
                            if !*use_preallocated {
                                // Direct to Apache Record Decoder - no field_nullable checks needed
                                if let NestedState::Struct { row_count, record_decoder, .. } = &mut self.columns[col_idx].nested {
                                    record_decoder.decode(&mut cursor)?;
                                    *row_count += 1;
                                }
                                self.columns[col_idx].row_count += 1;
                                continue;
                            }

                            // Pre-allocated path only (use_preallocated == true)
                            // Extract field_nullable flags from nested state
                            let field_nullable_flags: Vec<bool> = if let NestedState::Struct { field_nullable, .. } = &self.columns[col_idx].nested {
                                field_nullable.clone()
                            } else {
                                Vec::new()
                            };
                            let has_any_nullable = field_nullable_flags.iter().any(|&n| n);

                            // Inlined struct decode for pre-allocated path
                            if !has_any_nullable {
                                // Pre-allocated path with function pointers (zero enum match overhead)
                                let provider_col_idx = column_indices[col_idx];
                                if let NestedState::Struct { row_count, field_positions, field_specs, decoder, .. } = &mut self.columns[col_idx].nested {
                                    // Fast path: use function pointer specs if all fields are fixed-size
                                    if let Some(specs) = field_specs {
                                        for (field_idx, spec) in specs.iter().enumerate() {
                                            if spec.size > 0 {
                                                let pos = field_positions[field_idx];
                                                let slice = self.provider.get_child_data_slice(provider_col_idx, field_idx, pos, spec.size)?;
                                                // Direct function pointer call - no enum match
                                                (spec.decode_fn)(&mut cursor, slice)?;
                                                field_positions[field_idx] += spec.size;
                                            }
                                        }
                                    } else if let FieldDecoder::Struct { fields, .. } = decoder {
                                        // Fallback for structs with variable-length fields
                                        for (field_idx, field_decoder) in fields.iter().enumerate() {
                                            let pos = field_positions[field_idx];
                                            if let Some(fixed_size) = field_decoder.fixed_size() {
                                                if fixed_size > 0 {
                                                    let slice = self.provider.get_child_data_slice(provider_col_idx, field_idx, pos, fixed_size)?;
                                                    field_decoder.decode_to_slice(&mut cursor, slice)?;
                                                    field_positions[field_idx] += fixed_size;
                                                }
                                            } else {
                                                // Variable-length field (String/Binary)
                                                let bytes = cursor.get_bytes()?;
                                                let slice = self.provider.get_child_data_slice(provider_col_idx, field_idx, pos, bytes.len())?;
                                                slice.copy_from_slice(bytes);
                                                field_positions[field_idx] += bytes.len();
                                            }
                                        }
                                    }
                                    *row_count += 1;
                                }
                            }
                            // has_any_nullable == true falls through here (no preallocated path available)
                            // This shouldn't happen in practice since we check !use_preallocated first
                            self.columns[col_idx].row_count += 1;
                        }
                    }
                }
            }

            // Update positions, row counts, and null bitmaps after processing batch
            for col_idx in 0..num_columns {
                // Update null bitmaps for nullable columns
                if nullable_flags[col_idx] && !validity_bits[col_idx].is_empty() {
                    let base_row = base_row_counts[col_idx];
                    for (row, is_valid) in validity_bits[col_idx].iter().enumerate() {
                        self.columns[col_idx].set_validity(base_row + row, *is_valid);
                    }
                }

                match &col_kinds[col_idx] {
                    ColKind::Int32 | ColKind::Float32 => {
                        self.columns[col_idx].data_position += 4 * batch_rows;
                        self.columns[col_idx].row_count += batch_rows;
                    }
                    ColKind::Int64 | ColKind::Float64 => {
                        self.columns[col_idx].data_position += 8 * batch_rows;
                        self.columns[col_idx].row_count += batch_rows;
                    }
                    ColKind::Boolean => {
                        self.columns[col_idx].row_count += batch_rows;
                    }
                    ColKind::FixedBytes { size } => {
                        self.columns[col_idx].data_position += size * batch_rows;
                        self.columns[col_idx].row_count += batch_rows;
                    }
                    ColKind::Decimal128 => {
                        self.columns[col_idx].data_position += 16 * batch_rows;
                        self.columns[col_idx].row_count += batch_rows;
                    }
                    ColKind::String | ColKind::Binary => {
                        self.columns[col_idx].data_position += var_data_offsets[col_idx];
                        self.columns[col_idx].row_count += batch_rows;
                    }
                    ColKind::Null => {
                        self.columns[col_idx].row_count += batch_rows;
                    }
                    // Complex types update row_count inline
                    ColKind::Dictionary | ColKind::List { .. } | ColKind::Map { .. }
                    | ColKind::Union { .. } | ColKind::Struct { .. } => {}
                }
            }

            self.row_count += batch_rows;
            rows_processed += batch_rows;
        }

        Ok(cursor.position())
    }

    /// Batch decode for schemas with mixed fixed and variable-length types
    /// Processes in mini-batches of BATCH_SIZE rows for efficient pointer management
    const BATCH_SIZE: usize = 1024;

    /// Ultra-fast batch decode for string-only schemas
    /// Optimized tight loop without type dispatch overhead
    #[inline(always)]
    fn decode_batch_strings_only(
        &mut self,
        buf: &[u8],
        count: usize,
    ) -> Result<usize, ArrowError> {
        let total_rows = count.min(self.max_rows - self.row_count);
        if total_rows == 0 {
            return Ok(0);
        }

        let num_columns = self.columns.len();
        let mut cursor = AvroCursor::new(buf);

        // Pre-compute column info once (not per-batch)
        // (column_index, is_nullable, base_row_count)
        let col_info: Vec<(usize, bool, usize)> = self.columns
            .iter()
            .map(|c| (c.column_index, c.is_nullable, c.row_count))
            .collect();

        // Process in mini-batches
        let mut rows_processed = 0;
        while rows_processed < total_rows {
            let batch_rows = (total_rows - rows_processed).min(Self::BATCH_SIZE);

            // Get pointers for this batch - minimal overhead
            let mut data_ptrs: Vec<*mut u8> = Vec::with_capacity(num_columns);
            let mut offsets_ptrs: Vec<*mut u8> = Vec::with_capacity(num_columns);
            let mut data_capacities: Vec<usize> = Vec::with_capacity(num_columns);
            let mut base_data_positions: Vec<usize> = Vec::with_capacity(num_columns);
            let mut base_row_counts: Vec<usize> = Vec::with_capacity(num_columns);

            for (col_idx, &(column_index, _is_nullable, row_count)) in col_info.iter().enumerate() {
                let col = &self.columns[col_idx];
                let (data_ptr, data_cap) = self.provider.get_data_ptr(column_index, col.data_position)?;
                data_ptrs.push(data_ptr);
                data_capacities.push(data_cap);
                base_data_positions.push(col.data_position);
                base_row_counts.push(row_count + rows_processed);

                let offsets_pos = (col.row_count + 1) * 4;
                let offsets_slice = self.provider.get_offsets_slice(column_index, offsets_pos, batch_rows * 4)?;
                offsets_ptrs.push(offsets_slice.as_mut_ptr());
            }

            // Track positions within batch
            let mut var_data_offsets: Vec<usize> = vec![0; num_columns];

            // Track validity for nullable columns
            let mut validity_bits: Vec<Vec<bool>> = col_info.iter()
                .map(|(_, is_nullable, _)| {
                    if *is_nullable {
                        Vec::with_capacity(batch_rows)
                    } else {
                        Vec::new()
                    }
                })
                .collect();

            // Tight decode loop
            for row in 0..batch_rows {
                for col_idx in 0..num_columns {
                    let is_nullable = col_info[col_idx].1;

                    // For nullable columns, read the Avro union index first
                    let is_null = if is_nullable {
                        let union_index = cursor.get_long()?;
                        let is_valid = union_index != 0;
                        validity_bits[col_idx].push(is_valid);
                        !is_valid
                    } else {
                        false
                    };

                    if is_null {
                        // Write same offset as previous (empty string for null)
                        let current_offset = var_data_offsets[col_idx];
                        let new_end = (base_data_positions[col_idx] + current_offset) as i32;
                        unsafe {
                            let dest = offsets_ptrs[col_idx].add(row * 4);
                            std::ptr::copy_nonoverlapping(new_end.to_le_bytes().as_ptr(), dest, 4);
                        }
                        continue;
                    }

                    let bytes = cursor.get_bytes()?;
                    let len = bytes.len();
                    let current_offset = var_data_offsets[col_idx];

                    // Check capacity
                    if current_offset + len > data_capacities[col_idx] {
                        return Err(ArrowError::InvalidArgumentError(format!(
                            "String data capacity exceeded for column {}",
                            col_idx
                        )));
                    }

                    // Write data directly
                    unsafe {
                        let dest = data_ptrs[col_idx].add(current_offset);
                        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dest, len);
                    }

                    // Write offset
                    let new_end = (base_data_positions[col_idx] + current_offset + len) as i32;
                    unsafe {
                        let dest = offsets_ptrs[col_idx].add(row * 4);
                        std::ptr::copy_nonoverlapping(new_end.to_le_bytes().as_ptr(), dest, 4);
                    }

                    var_data_offsets[col_idx] += len;
                }
            }

            // Update positions, counts, and null bitmaps after batch
            for col_idx in 0..num_columns {
                let is_nullable = col_info[col_idx].1;

                // Update null bitmaps for nullable columns
                if is_nullable {
                    let base_row = base_row_counts[col_idx];
                    for (row, is_valid) in validity_bits[col_idx].iter().enumerate() {
                        self.columns[col_idx].set_validity(base_row + row, *is_valid);
                    }
                }

                self.columns[col_idx].data_position += var_data_offsets[col_idx];
                self.columns[col_idx].row_count += batch_rows;
            }

            rows_processed += batch_rows;
            self.row_count += batch_rows;
        }

        Ok(cursor.position())
    }

    fn decode_batch_mixed(
        &mut self,
        buf: &[u8],
        count: usize,
    ) -> Result<usize, ArrowError> {
        let total_rows = count.min(self.max_rows - self.row_count);
        if total_rows == 0 {
            return Ok(0);
        }

        let num_columns = self.columns.len();
        let mut cursor = AvroCursor::new(buf);

        // Process in mini-batches
        let mut rows_processed = 0;
        while rows_processed < total_rows {
            let batch_rows = (total_rows - rows_processed).min(Self::BATCH_SIZE);

            // Build column metadata for this batch
            // (column_index, data_type, type_size, start_position, base_row_count, is_nullable)
            let col_meta: Vec<(usize, DataType, Option<usize>, usize, usize, bool)> = self.columns
                .iter()
                .map(|col| {
                    let type_size = Self::get_type_size(&col.data_type);
                    (col.column_index, col.data_type.clone(), type_size, col.data_position, col.row_count, col.is_nullable)
                })
                .collect();

            // Get raw pointers for fixed-size columns
            // For variable-length columns, we'll get pointers to data and offsets regions
            let mut data_ptrs: Vec<Option<*mut u8>> = Vec::with_capacity(num_columns);
            let mut offsets_ptrs: Vec<Option<*mut u8>> = Vec::with_capacity(num_columns);
            let mut data_capacities: Vec<usize> = Vec::with_capacity(num_columns);

            for (column_index, data_type, type_size, start_pos, row_count, _is_nullable) in &col_meta {
                match data_type {
                    DataType::Null => {
                        data_ptrs.push(None);
                        offsets_ptrs.push(None);
                        data_capacities.push(0);
                    }
                    DataType::Boolean => {
                        let (ptr, cap) = self.provider.get_data_ptr(*column_index, 0)?;
                        data_ptrs.push(Some(ptr));
                        offsets_ptrs.push(None);
                        data_capacities.push(cap);
                    }
                    DataType::Utf8 | DataType::Binary => {
                        // Get data pointer at current position
                        let (data_ptr, data_cap) = self.provider.get_data_ptr(*column_index, *start_pos)?;
                        data_ptrs.push(Some(data_ptr));
                        data_capacities.push(data_cap);
                        // Get offsets pointer - offsets start at (row_count + 1) * 4
                        let offsets_pos = (*row_count + 1) * 4;
                        let offsets_slice = self.provider.get_offsets_slice(*column_index, offsets_pos, batch_rows * 4)?;
                        offsets_ptrs.push(Some(offsets_slice.as_mut_ptr()));
                    }
                    DataType::FixedSizeBinary(size) => {
                        let (ptr, cap) = self.provider.get_data_ptr(*column_index, *start_pos)?;
                        data_ptrs.push(Some(ptr));
                        offsets_ptrs.push(None);
                        data_capacities.push(cap.min(batch_rows * (*size as usize)));
                    }
                    _ => {
                        // Fixed-size types
                        if let Some(ts) = type_size {
                            let (ptr, cap) = self.provider.get_data_ptr(*column_index, *start_pos)?;
                            data_ptrs.push(Some(ptr));
                            offsets_ptrs.push(None);
                            data_capacities.push(cap.min(batch_rows * ts));
                        } else {
                            data_ptrs.push(None);
                            offsets_ptrs.push(None);
                            data_capacities.push(0);
                        }
                    }
                }
            }

            // Track data positions for variable-length columns within this batch
            let mut var_data_offsets: Vec<usize> = vec![0; num_columns];
            let mut var_offset_idx: Vec<usize> = vec![0; num_columns];

            // Track validity for nullable columns during decode
            let mut validity_bits: Vec<Vec<bool>> = col_meta.iter()
                .map(|(_, _, _, _, _, is_nullable)| {
                    if *is_nullable {
                        Vec::with_capacity(batch_rows)
                    } else {
                        Vec::new()
                    }
                })
                .collect();

            // Decode batch
            for row in 0..batch_rows {
                for col_idx in 0..num_columns {
                    let (_, ref data_type, _type_size, _, base_row_count, is_nullable) = col_meta[col_idx];

                    // For nullable columns, read the Avro union index first
                    let is_null = if is_nullable {
                        let union_index = cursor.get_long()?;
                        let is_valid = union_index != 0;
                        validity_bits[col_idx].push(is_valid);
                        !is_valid
                    } else {
                        false
                    };

                    // Skip value reading for null values, write zero placeholder
                    if is_null {
                        match data_type {
                            DataType::Null | DataType::Boolean => {}
                            DataType::Int32 | DataType::Date32 | DataType::Time32(_) | DataType::Float32 => {
                                if let Some(ptr) = data_ptrs[col_idx] {
                                    unsafe {
                                        let dest = ptr.add(row * 4);
                                        std::ptr::write_bytes(dest, 0, 4);
                                    }
                                }
                            }
                            DataType::Int64 | DataType::Time64(_) | DataType::Timestamp(_, _) | DataType::Float64 => {
                                if let Some(ptr) = data_ptrs[col_idx] {
                                    unsafe {
                                        let dest = ptr.add(row * 8);
                                        std::ptr::write_bytes(dest, 0, 8);
                                    }
                                }
                            }
                            DataType::Utf8 | DataType::Binary => {
                                // Write same offset as previous (empty string/bytes)
                                let current_offset = var_data_offsets[col_idx];
                                let new_end = (col_meta[col_idx].3 + current_offset) as i32;
                                if let Some(ptr) = offsets_ptrs[col_idx] {
                                    unsafe {
                                        let dest = ptr.add(var_offset_idx[col_idx] * 4);
                                        std::ptr::copy_nonoverlapping(new_end.to_le_bytes().as_ptr(), dest, 4);
                                    }
                                }
                                var_offset_idx[col_idx] += 1;
                            }
                            DataType::FixedSizeBinary(size) => {
                                if let Some(ptr) = data_ptrs[col_idx] {
                                    unsafe {
                                        let dest = ptr.add(row * (*size as usize));
                                        std::ptr::write_bytes(dest, 0, *size as usize);
                                    }
                                }
                            }
                            _ => {}
                        }
                        continue;
                    }

                    match data_type {
                        DataType::Null => {}
                        DataType::Boolean => {
                            let value = cursor.get_bool()?;
                            let base_row = base_row_count + row;
                            let byte_idx = base_row / 8;
                            let bit_idx = base_row % 8;
                            if let Some(ptr) = data_ptrs[col_idx] {
                                unsafe {
                                    let byte_ptr = ptr.add(byte_idx);
                                    if value {
                                        *byte_ptr |= 1 << bit_idx;
                                    }
                                }
                            }
                        }
                        DataType::Int32 | DataType::Date32 | DataType::Time32(_) => {
                            let value = cursor.get_int()?;
                            if let Some(ptr) = data_ptrs[col_idx] {
                                unsafe {
                                    let dest = ptr.add(row * 4);
                                    std::ptr::copy_nonoverlapping(value.to_le_bytes().as_ptr(), dest, 4);
                                }
                            }
                        }
                        DataType::Int64 | DataType::Time64(_) | DataType::Timestamp(_, _) => {
                            let value = cursor.get_long()?;
                            if let Some(ptr) = data_ptrs[col_idx] {
                                unsafe {
                                    let dest = ptr.add(row * 8);
                                    std::ptr::copy_nonoverlapping(value.to_le_bytes().as_ptr(), dest, 8);
                                }
                            }
                        }
                        DataType::Float32 => {
                            let value = cursor.get_float()?;
                            if let Some(ptr) = data_ptrs[col_idx] {
                                unsafe {
                                    let dest = ptr.add(row * 4);
                                    std::ptr::copy_nonoverlapping(value.to_le_bytes().as_ptr(), dest, 4);
                                }
                            }
                        }
                        DataType::Float64 => {
                            let value = cursor.get_double()?;
                            if let Some(ptr) = data_ptrs[col_idx] {
                                unsafe {
                                    let dest = ptr.add(row * 8);
                                    std::ptr::copy_nonoverlapping(value.to_le_bytes().as_ptr(), dest, 8);
                                }
                            }
                        }
                        DataType::Utf8 | DataType::Binary => {
                            let bytes = cursor.get_bytes()?;
                            let len = bytes.len();
                            let current_offset = var_data_offsets[col_idx];

                            // Check capacity
                            if current_offset + len > data_capacities[col_idx] {
                                return Err(ArrowError::InvalidArgumentError(format!(
                                    "Variable data capacity exceeded for column {}",
                                    col_idx
                                )));
                            }

                            // Write data
                            if let Some(ptr) = data_ptrs[col_idx] {
                                unsafe {
                                    let dest = ptr.add(current_offset);
                                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), dest, len);
                                }
                            }

                            // Write offset (end position for this string)
                            let new_end = (col_meta[col_idx].3 + current_offset + len) as i32;
                            if let Some(ptr) = offsets_ptrs[col_idx] {
                                unsafe {
                                    let dest = ptr.add(var_offset_idx[col_idx] * 4);
                                    std::ptr::copy_nonoverlapping(new_end.to_le_bytes().as_ptr(), dest, 4);
                                }
                            }

                            var_data_offsets[col_idx] += len;
                            var_offset_idx[col_idx] += 1;
                        }
                        DataType::FixedSizeBinary(size) => {
                            let bytes = cursor.get_fixed(*size as usize)?;
                            if let Some(ptr) = data_ptrs[col_idx] {
                                unsafe {
                                    let dest = ptr.add(row * (*size as usize));
                                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), dest, *size as usize);
                                }
                            }
                        }
                        _ => {
                            return Err(ArrowError::InvalidArgumentError(
                                "Unsupported type in batch decode".to_string(),
                            ));
                        }
                    }
                }
            }

            // Update column positions, counts, and null bitmaps after batch
            for col_idx in 0..num_columns {
                let (_, ref data_type, type_size, _, base_row_count, is_nullable) = col_meta[col_idx];

                // Update null bitmaps for nullable columns
                if is_nullable {
                    for (row, is_valid) in validity_bits[col_idx].iter().enumerate() {
                        self.columns[col_idx].set_validity(base_row_count + row, *is_valid);
                    }
                }

                match data_type {
                    DataType::Null => {
                        self.columns[col_idx].row_count += batch_rows;
                    }
                    DataType::Boolean => {
                        self.columns[col_idx].row_count += batch_rows;
                    }
                    DataType::Utf8 | DataType::Binary => {
                        self.columns[col_idx].data_position += var_data_offsets[col_idx];
                        self.columns[col_idx].row_count += batch_rows;
                    }
                    _ => {
                        if let Some(ts) = type_size {
                            self.columns[col_idx].data_position += batch_rows * ts;
                        }
                        self.columns[col_idx].row_count += batch_rows;
                    }
                }
            }

            rows_processed += batch_rows;
            self.row_count += batch_rows;
        }

        Ok(cursor.position())
    }

    /// Create a checkpoint of current decoder state for potential rollback
    fn create_checkpoint(&self, cursor_position: usize) -> RowCheckpoint {
        RowCheckpoint {
            cursor_position,
            data_positions: self.columns.iter().map(|c| c.data_position).collect(),
            row_counts: self.columns.iter().map(|c| c.row_count).collect(),
            nested_snapshots: self.columns.iter().map(|c| c.create_snapshot()).collect(),
        }
    }

    /// Restore decoder state from a checkpoint (rollback partial row)
    fn restore_checkpoint(&mut self, checkpoint: &RowCheckpoint) {
        for (col_idx, col) in self.columns.iter_mut().enumerate() {
            col.data_position = checkpoint.data_positions[col_idx];
            col.row_count = checkpoint.row_counts[col_idx];
            col.restore_snapshot(&checkpoint.nested_snapshots[col_idx]);
        }
    }

    /// Check if there's capacity for a fixed-size write
    #[allow(dead_code)]
    fn check_fixed_capacity(
        &self,
        column_index: usize,
        bytes_needed: usize,
    ) -> Result<(), CapacityExceededError> {
        if !self.provider.has_capacity_for_variable_data(column_index, bytes_needed) {
            Err(CapacityExceededError {
                column_index,
                bytes_needed,
                bytes_available: 0, // We don't have exact info, provider just says no
            })
        } else {
            Ok(())
        }
    }

    /// Set dictionary values for a Dictionary column (Avro enum symbols)
    ///
    /// This must be called before decoding for Dictionary columns,
    /// as Avro enums encode values as indices into the symbol list.
    pub fn set_dictionary_values(&mut self, col_idx: usize, values: Vec<String>) -> Result<(), ArrowError> {
        if col_idx >= self.columns.len() {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Column index {} out of bounds (max {})",
                col_idx, self.columns.len()
            )));
        }

        let col = &mut self.columns[col_idx];
        if let NestedState::Dictionary { dictionary, .. } = &mut col.nested {
            *dictionary = values;
            Ok(())
        } else {
            Err(ArrowError::InvalidArgumentError(format!(
                "Column {} is not a Dictionary type",
                col_idx
            )))
        }
    }

    /// Enable pre-allocated buffer mode for a List column
    ///
    /// When enabled, list child data is written directly to the provider's
    /// pre-allocated buffer instead of growing a Vec. This enables zero-allocation
    /// decoding for List types.
    ///
    /// The provider must have child regions configured for the specified column.
    pub fn enable_preallocated_list(&mut self, col_idx: usize) -> Result<(), ArrowError> {
        if col_idx >= self.columns.len() {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Column index {} out of bounds (max {})",
                col_idx, self.columns.len()
            )));
        }

        let col = &mut self.columns[col_idx];
        if let NestedState::List { use_preallocated, .. } = &mut col.nested {
            *use_preallocated = true;
            Ok(())
        } else {
            Err(ArrowError::InvalidArgumentError(format!(
                "Column {} is not a List type",
                col_idx
            )))
        }
    }

    /// Decode a single row from the cursor
    ///
    /// This method uses checkpointing for partial row rollback. If any column
    /// decode fails due to capacity limits, the entire row is rolled back
    /// and a `CapacityExceeded` error is returned.
    pub(crate) fn decode_row(&mut self, cursor: &mut AvroCursor<'_>) -> Result<(), ArrowError> {
        if self.row_count >= self.max_rows {
            return Err(ArrowError::InvalidArgumentError(
                "Decoder at capacity".to_string(),
            ));
        }

        // Create checkpoint before decoding row
        let checkpoint = self.create_checkpoint(cursor.position());

        let num_columns = self.columns.len();
        for col_idx in 0..num_columns {
            match self.decode_column_value(col_idx, cursor) {
                Ok(()) => {}
                Err(e) => {
                    // Rollback to checkpoint on any error
                    self.restore_checkpoint(&checkpoint);
                    return Err(e);
                }
            }
        }

        self.row_count += 1;
        Ok(())
    }

    /// Decode a single row with explicit capacity error handling
    ///
    /// Returns `Ok(true)` if row was decoded successfully.
    /// Returns `Ok(false)` if capacity was exceeded (row was rolled back).
    /// Returns `Err` for other errors.
    pub(crate) fn try_decode_row(&mut self, cursor: &mut AvroCursor<'_>) -> Result<bool, ArrowError> {
        if self.row_count >= self.max_rows {
            return Ok(false); // Capacity exceeded
        }

        match self.decode_row(cursor) {
            Ok(()) => Ok(true),
            Err(ArrowError::InvalidArgumentError(msg)) if msg.contains("exceeds capacity") => {
                Ok(false) // Capacity exceeded, row was rolled back
            }
            Err(e) => Err(e),
        }
    }

    /// Decode a column value directly into the buffer
    fn decode_column_value(
        &mut self,
        col_idx: usize,
        cursor: &mut AvroCursor<'_>,
    ) -> Result<(), ArrowError> {
        let col = &self.columns[col_idx];
        let column_index = col.column_index;
        let data_position = col.data_position;
        let row_count = col.row_count;
        let data_type = col.data_type.clone();

        match &data_type {
            DataType::Null => {
                // Avro null takes 0 bytes, just increment count
                self.columns[col_idx].row_count += 1;
            }
            DataType::Int32 | DataType::Date32 | DataType::Time32(_) => {
                let value = cursor.get_int()?;
                let bytes = value.to_le_bytes();
                let slice = self.provider.get_data_slice(column_index, data_position, 4)?;
                slice.copy_from_slice(&bytes);
                self.columns[col_idx].data_position += 4;
                self.columns[col_idx].row_count += 1;
            }
            DataType::Int64 | DataType::Time64(_) | DataType::Timestamp(_, _) => {
                let value = cursor.get_long()?;
                let bytes = value.to_le_bytes();
                let slice = self.provider.get_data_slice(column_index, data_position, 8)?;
                slice.copy_from_slice(&bytes);
                self.columns[col_idx].data_position += 8;
                self.columns[col_idx].row_count += 1;
            }
            DataType::Float32 => {
                let value = cursor.get_float()?;
                let bytes = value.to_le_bytes();
                let slice = self.provider.get_data_slice(column_index, data_position, 4)?;
                slice.copy_from_slice(&bytes);
                self.columns[col_idx].data_position += 4;
                self.columns[col_idx].row_count += 1;
            }
            DataType::Float64 => {
                let value = cursor.get_double()?;
                let bytes = value.to_le_bytes();
                let slice = self.provider.get_data_slice(column_index, data_position, 8)?;
                slice.copy_from_slice(&bytes);
                self.columns[col_idx].data_position += 8;
                self.columns[col_idx].row_count += 1;
            }
            DataType::Boolean => {
                let value = cursor.get_bool()?;
                let byte_idx = row_count / 8;
                let bit_idx = row_count % 8;
                let slice = self.provider.get_data_slice(column_index, byte_idx, 1)?;
                if value {
                    slice[0] |= 1 << bit_idx;
                }
                self.columns[col_idx].row_count += 1;
            }
            DataType::Utf8 | DataType::Binary => {
                let data = cursor.get_bytes()?;
                let data_len = data.len();

                if !self.provider.has_capacity_for_variable_data(column_index, data_len) {
                    return Err(ArrowError::InvalidArgumentError(format!(
                        "Variable data exceeds buffer capacity for column {}",
                        column_index
                    )));
                }

                // Write data
                let data_slice = self.provider.get_data_slice(column_index, data_position, data_len)?;
                data_slice.copy_from_slice(data);

                // Write offset (i32)
                let new_end = (data_position + data_len) as i32;
                let offset_bytes = new_end.to_le_bytes();
                let offset_slice = self.provider.get_offsets_slice(column_index, (row_count + 1) * 4, 4)?;
                offset_slice.copy_from_slice(&offset_bytes);

                self.columns[col_idx].data_position += data_len;
                self.columns[col_idx].row_count += 1;
            }
            DataType::FixedSizeBinary(size) => {
                let size = *size as usize;
                let data = cursor.get_fixed(size)?;
                let slice = self.provider.get_data_slice(column_index, data_position, size)?;
                slice.copy_from_slice(data);
                self.columns[col_idx].data_position += size;
                self.columns[col_idx].row_count += 1;
            }
            DataType::Decimal128(_, _) => {
                // Avro decimals are stored as bytes (variable) or fixed (we handle fixed here)
                // For variable bytes, read length-prefixed bytes
                let bytes = cursor.get_bytes()?;
                // Convert from big-endian two's complement to i128
                let value = decode_decimal_bytes_to_i128(bytes);
                let slice = self.provider.get_data_slice(column_index, data_position, 16)?;
                slice.copy_from_slice(&value.to_le_bytes());
                self.columns[col_idx].data_position += 16;
                self.columns[col_idx].row_count += 1;
            }
            DataType::Decimal256(_, _) => {
                let bytes = cursor.get_bytes()?;
                // Convert from big-endian two's complement to i256 (stored as [u8; 32])
                let value = decode_decimal_bytes_to_i256(bytes);
                let slice = self.provider.get_data_slice(column_index, data_position, 32)?;
                slice.copy_from_slice(&value);
                self.columns[col_idx].data_position += 32;
                self.columns[col_idx].row_count += 1;
            }
            DataType::Interval(IntervalUnit::MonthDayNano) => {
                // Avro duration is fixed 12 bytes: months (u32 LE) + days (u32 LE) + millis (u32 LE)
                let bytes = cursor.get_fixed(12)?;
                let months = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as i32;
                let days = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as i32;
                let millis = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
                let nanos = (millis as i64) * 1_000_000;

                // Arrow stores IntervalMonthDayNano as 16 bytes: months (i32) + days (i32) + nanos (i64)
                let interval = IntervalMonthDayNano::new(months, days, nanos);
                let interval_bytes: [u8; 16] = unsafe { std::mem::transmute(interval) };
                let slice = self.provider.get_data_slice(column_index, data_position, 16)?;
                slice.copy_from_slice(&interval_bytes);
                self.columns[col_idx].data_position += 16;
                self.columns[col_idx].row_count += 1;
            }
            DataType::Map(_, _) => {
                // Decode Avro map using block encoding into nested state
                self.decode_map_value(col_idx, cursor)?;
            }
            DataType::Union(_, _) => {
                // Decode Avro union (type tag + value)
                self.decode_union_value(col_idx, cursor)?;
            }
            DataType::List(_) => {
                // Decode Avro array using block encoding
                self.decode_list_value(col_idx, cursor)?;
            }
            DataType::Dictionary(_, _) => {
                // Decode Avro enum (varint index into symbols)
                self.decode_dictionary_value(col_idx, cursor)?;
            }
            DataType::Struct(_) => {
                // Decode Avro record (nested fields in order)
                self.decode_struct_value(col_idx, cursor)?;
            }
            other => {
                return Err(ArrowError::NotYetImplemented(format!(
                    "Pooled decoding not yet implemented for {:?}",
                    other
                )));
            }
        }
        Ok(())
    }

    /// Decode a Dictionary value (Avro enum encoded as varint index)
    fn decode_dictionary_value(
        &mut self,
        col_idx: usize,
        cursor: &mut AvroCursor<'_>,
    ) -> Result<(), ArrowError> {
        let col = &mut self.columns[col_idx];

        if let NestedState::Dictionary { indices, .. } = &mut col.nested {
            // Avro enum is encoded as varint index into the symbol list
            let index = cursor.get_int()?;
            indices.push(index);
            col.row_count += 1;
            Ok(())
        } else {
            Err(ArrowError::InvalidArgumentError(
                "Dictionary column missing nested state".to_string()
            ))
        }
    }

    /// Decode a Struct value (Avro record - fields in schema order)
    /// Uses Apache's Decoder directly for best performance
    fn decode_struct_value(
        &mut self,
        col_idx: usize,
        cursor: &mut AvroCursor<'_>,
    ) -> Result<(), ArrowError> {
        let col = &mut self.columns[col_idx];

        if let NestedState::Struct { field_nullable, field_validity, row_count, apache_decoders, field_names, .. } = &mut col.nested {
            let num_fields = apache_decoders.len();

            // DEBUG: Force output to see if we get here
            if *row_count == 0 && num_fields > 0 {
                panic!("DEBUG decode_struct_value called: num_fields={}, field_names={:?}", num_fields, field_names);
            }

            // Avro records encode fields in schema order
            // Use Apache's Decoder directly for best performance
            for field_idx in 0..num_fields {
                let is_nullable = field_nullable[field_idx];
                let decoder = &mut apache_decoders[field_idx];

                if is_nullable {
                    // Read union index for nullable fields
                    let union_index = cursor.get_long()?;
                    let is_null = union_index == 0;

                    // Track validity separately (single pass, no markers)
                    field_validity[field_idx].push(!is_null);

                    if !is_null {
                        // Decode using Apache's Decoder
                        decoder.decode(cursor)?;
                        if *row_count == 0 {
                            eprintln!("DEBUG: field {} decoded (nullable, non-null), decoder={:?}", field_idx, std::any::type_name_of_val(decoder));
                        }
                    } else {
                        // For null values, push placeholder using Apache's Decoder
                        decoder.append_null()?;
                    }
                } else {
                    // Non-nullable field - decode directly using Apache's Decoder
                    decoder.decode(cursor)?;
                    if *row_count == 0 {
                        eprintln!("DEBUG: field {} decoded (non-nullable), decoder={:?}", field_idx, std::any::type_name_of_val(decoder));
                    }
                }
            }

            *row_count += 1;
            // Also increment column row count
            self.columns[col_idx].row_count += 1;
            Ok(())
        } else {
            Err(ArrowError::InvalidArgumentError(
                "Struct column missing nested state".to_string()
            ))
        }
    }

    /// Append null placeholder to TypedStorage
    #[allow(dead_code)]
    fn append_null_to_typed_storage(storage: &mut TypedStorage) {
        match storage {
            TypedStorage::Null(count) => *count += 1,
            TypedStorage::Boolean(v) => v.push(0),
            TypedStorage::Int32(v) => v.push(0),
            TypedStorage::Int64(v) => v.push(0),
            TypedStorage::Float32(v) => v.push(0.0),
            TypedStorage::Float64(v) => v.push(0.0),
            TypedStorage::String { offsets, .. } | TypedStorage::Binary { offsets, .. } => {
                if offsets.is_empty() { offsets.push(0); }
                offsets.push(*offsets.last().unwrap());
            }
            TypedStorage::Fixed { data, size } => {
                data.extend(std::iter::repeat(0u8).take(*size));
            }
            TypedStorage::Decimal128(v) => v.push(0),
            TypedStorage::Interval(v) => v.push([0u8; 16]),
            TypedStorage::Bytes(v) => {
                v.extend_from_slice(&0i32.to_le_bytes());
            }
        }
    }

    /// Append null placeholder data for a given type
    /// This ensures fixed-size arrays have consistent element counts
    #[allow(dead_code)]
    fn append_null_placeholder(data_type: &DataType, output: &mut Vec<u8>) {
        match data_type {
            DataType::Boolean => output.push(0),
            DataType::Int32 | DataType::Date32 | DataType::Time32(_) | DataType::Float32 => {
                output.extend_from_slice(&[0u8; 4]);
            }
            DataType::Int64 | DataType::Time64(_) | DataType::Timestamp(_, _) | DataType::Float64 => {
                output.extend_from_slice(&[0u8; 8]);
            }
            DataType::Utf8 | DataType::Binary => {
                // For variable-length types, store 0 length
                output.extend_from_slice(&0i32.to_le_bytes());
            }
            DataType::FixedSizeBinary(size) => {
                output.extend(std::iter::repeat(0u8).take(*size as usize));
            }
            _ => {
                // For complex types (List, Map, Struct), null is tracked in bitmap only
                // No placeholder data needed as they use their own internal structures
            }
        }
    }

    /// Decode a Map value from Avro block encoding (Vec mode only)
    /// Avro maps: { key: string, value: T } with block encoding
    fn decode_map_value(
        &mut self,
        col_idx: usize,
        cursor: &mut AvroCursor<'_>,
    ) -> Result<(), ArrowError> {
        // Extract value type and nullability from nested state
        let (value_type_clone, is_value_nullable) = {
            let col = &self.columns[col_idx];
            if let NestedState::Map { value_type, value_nullable, .. } = &col.nested {
                (value_type.clone(), *value_nullable)
            } else {
                return Err(ArrowError::InvalidArgumentError(
                    "Map column missing nested state".to_string()
                ));
            }
        };

        let mut entries_in_row = 0usize;

        // Read blocks until we get a 0 count
        loop {
            let block_count = cursor.get_long()?;
            match block_count.cmp(&0) {
                Ordering::Equal => break,
                Ordering::Less => {
                    let count = (-block_count) as usize;
                    let _size_in_bytes = cursor.get_long()?; // Skip size hint
                    if let NestedState::Map { key_offsets, key_data, value_data, value_validity, .. } = &mut self.columns[col_idx].nested {
                        for _ in 0..count {
                            let key_bytes = cursor.get_bytes()?;
                            key_data.extend_from_slice(key_bytes);
                            key_offsets.push(key_data.len() as i32);
                            // Handle nullable values with separate validity tracking
                            if is_value_nullable {
                                let union_index = cursor.get_long()?;
                                if union_index == 0 {
                                    value_validity.push(false); // null
                                    Self::append_placeholder_to_vec(&value_type_clone, value_data);
                                } else {
                                    value_validity.push(true); // valid
                                    Self::decode_value_to_vec(&value_type_clone, cursor, value_data)?;
                                }
                            } else {
                                Self::decode_value_to_vec(&value_type_clone, cursor, value_data)?;
                            }
                        }
                    }
                    entries_in_row += count;
                }
                Ordering::Greater => {
                    let count = block_count as usize;
                    if let NestedState::Map { key_offsets, key_data, value_data, value_validity, .. } = &mut self.columns[col_idx].nested {
                        for _ in 0..count {
                            let key_bytes = cursor.get_bytes()?;
                            key_data.extend_from_slice(key_bytes);
                            key_offsets.push(key_data.len() as i32);
                            // Handle nullable values with separate validity tracking
                            if is_value_nullable {
                                let union_index = cursor.get_long()?;
                                if union_index == 0 {
                                    value_validity.push(false); // null
                                    Self::append_placeholder_to_vec(&value_type_clone, value_data);
                                } else {
                                    value_validity.push(true); // valid
                                    Self::decode_value_to_vec(&value_type_clone, cursor, value_data)?;
                                }
                            } else {
                                Self::decode_value_to_vec(&value_type_clone, cursor, value_data)?;
                            }
                        }
                    }
                    entries_in_row += count;
                }
            }
        }

        // Update final state
        let col = &mut self.columns[col_idx];
        if let NestedState::Map { map_offsets, total_entries, .. } = &mut col.nested {
            *total_entries += entries_in_row;
            map_offsets.push(*total_entries as i32);
        }
        col.row_count += 1;
        Ok(())
    }

    /// Decode a Union value (type tag + value)
    /// Optimized with inline fast paths for common fixed-size types
    fn decode_union_value(
        &mut self,
        col_idx: usize,
        cursor: &mut AvroCursor<'_>,
    ) -> Result<(), ArrowError> {
        let col = &mut self.columns[col_idx];

        if let NestedState::Union {
            type_ids, offsets, child_data, child_counts, fields, ..
        } = &mut col.nested {
            // Read the type tag (varint index into union variants)
            let type_index = cursor.get_long()? as usize;

            if type_index >= fields.len() {
                return Err(ArrowError::ParseError(format!(
                    "Union type index {} out of bounds (max {})",
                    type_index, fields.len()
                )));
            }

            // Get the actual type_id from the union fields
            let (field_type_id, field) = fields.iter().nth(type_index)
                .ok_or_else(|| ArrowError::ParseError("Union field not found".to_string()))?;

            type_ids.push(field_type_id);

            // For dense union, track offset into child array
            if let Some(offs) = offsets {
                offs.push(child_counts[type_index] as i32);
            }

            // Inline fast paths for common fixed-size types to avoid function call overhead
            let child_vec = &mut child_data[type_index];
            match field.data_type() {
                DataType::Null => {
                    // Null takes no space - nothing to decode
                }
                DataType::Int32 | DataType::Date32 | DataType::Time32(_) => {
                    let value = cursor.get_int()?;
                    child_vec.extend_from_slice(&value.to_le_bytes());
                }
                DataType::Int64 | DataType::Time64(_) | DataType::Timestamp(_, _) => {
                    let value = cursor.get_long()?;
                    child_vec.extend_from_slice(&value.to_le_bytes());
                }
                DataType::Float32 => {
                    let value = cursor.get_float()?;
                    child_vec.extend_from_slice(&value.to_le_bytes());
                }
                DataType::Float64 => {
                    let value = cursor.get_double()?;
                    child_vec.extend_from_slice(&value.to_le_bytes());
                }
                DataType::Boolean => {
                    let value = cursor.get_bool()?;
                    child_vec.push(if value { 1 } else { 0 });
                }
                other => {
                    // Fall back to generic decode for complex/variable-length types
                    Self::decode_value_to_vec(other, cursor, child_vec)?;
                }
            }
            child_counts[type_index] += 1;

            col.row_count += 1;
            Ok(())
        } else {
            Err(ArrowError::InvalidArgumentError(
                "Union column missing nested state".to_string()
            ))
        }
    }

    /// Decode a List value from Avro block encoding
    fn decode_list_value(
        &mut self,
        col_idx: usize,
        cursor: &mut AvroCursor<'_>,
    ) -> Result<(), ArrowError> {
        // Extract info we need from nested state first
        let (child_type_clone, is_child_nullable, use_prealloc, mut current_position) = {
            let col = &self.columns[col_idx];
            if let NestedState::List { child_type, child_nullable, use_preallocated, child_data_position, .. } = &col.nested {
                (child_type.clone(), *child_nullable, *use_preallocated, *child_data_position)
            } else {
                return Err(ArrowError::InvalidArgumentError(
                    "List column missing nested state".to_string()
                ));
            }
        };

        // For fixed-size types, calculate element size once
        let fixed_element_size = Self::get_fixed_element_size(&child_type_clone);

        let mut elements_in_row = 0usize;

        // Read blocks until we get a 0 count
        loop {
            let block_count = cursor.get_long()?;
            match block_count.cmp(&0) {
                Ordering::Equal => break,
                Ordering::Less => {
                    let count = (-block_count) as usize;
                    let _size_in_bytes = cursor.get_long()?;
                    if use_prealloc {
                        current_position = self.decode_list_block_preallocated(
                            col_idx, cursor, count, &child_type_clone,
                            fixed_element_size, current_position
                        )?;
                    } else {
                        current_position = self.decode_list_block_vec(
                            col_idx, cursor, count, &child_type_clone, is_child_nullable
                        )?;
                    }
                    elements_in_row += count;
                }
                Ordering::Greater => {
                    let count = block_count as usize;
                    if use_prealloc {
                        current_position = self.decode_list_block_preallocated(
                            col_idx, cursor, count, &child_type_clone,
                            fixed_element_size, current_position
                        )?;
                    } else {
                        current_position = self.decode_list_block_vec(
                            col_idx, cursor, count, &child_type_clone, is_child_nullable
                        )?;
                    }
                    elements_in_row += count;
                }
            }
        }

        // Update final state
        let col = &mut self.columns[col_idx];
        if let NestedState::List { offsets, child_data_position, total_elements, .. } = &mut col.nested {
            *child_data_position = current_position;
            *total_elements += elements_in_row;
            let new_offset = *total_elements as i32;
            offsets.push(new_offset);
        }
        col.row_count += 1;
        Ok(())
    }

    /// Get fixed element size for a type, or None if variable-length
    #[inline]
    fn get_fixed_element_size(data_type: &DataType) -> Option<usize> {
        match data_type {
            DataType::Null => Some(0),
            DataType::Boolean => Some(1),
            DataType::Int32 | DataType::Date32 | DataType::Time32(_) => Some(4),
            DataType::Int64 | DataType::Time64(_) | DataType::Timestamp(_, _) => Some(8),
            DataType::Float32 => Some(4),
            DataType::Float64 => Some(8),
            DataType::FixedSizeBinary(size) => Some(*size as usize),
            _ => None, // Variable-length types
        }
    }

    /// Decode a block of list elements using pre-allocated buffer (optimized)
    #[inline]
    fn decode_list_block_preallocated(
        &mut self,
        col_idx: usize,
        cursor: &mut AvroCursor<'_>,
        count: usize,
        child_type: &DataType,
        fixed_element_size: Option<usize>,
        start_position: usize,
    ) -> Result<usize, ArrowError> {
        // Get the provider column index
        let provider_col_idx = self.columns[col_idx].column_index;

        if let Some(elem_size) = fixed_element_size {
            // FAST PATH: Fixed-size elements - get one large slice, decode all elements
            let total_bytes = count * elem_size;

            // Single capacity check for entire block
            if !self.provider.has_capacity_for_child_data(provider_col_idx, 0, total_bytes) {
                return Err(ArrowError::InvalidArgumentError(
                    "List child data capacity exceeded".to_string()
                ));
            }

            // Get single slice for entire block
            let slice = self.provider.get_child_data_slice(provider_col_idx, 0, start_position, total_bytes)?;

            // Decode all elements directly into the slice
            Self::decode_fixed_elements_batch(child_type, cursor, slice, count, elem_size)?;

            Ok(start_position + total_bytes)
        } else {
            // SLOW PATH: Variable-length elements - decode one at a time
            let mut current_position = start_position;
            for _ in 0..count {
                let bytes_needed = Self::get_value_size(child_type, cursor)?;

                if !self.provider.has_capacity_for_child_data(provider_col_idx, 0, bytes_needed) {
                    return Err(ArrowError::InvalidArgumentError(
                        "List child data capacity exceeded".to_string()
                    ));
                }

                let slice = self.provider.get_child_data_slice(provider_col_idx, 0, current_position, bytes_needed)?;
                Self::decode_value_to_slice(child_type, cursor, slice)?;
                current_position += bytes_needed;
            }
            Ok(current_position)
        }
    }

    /// Decode a block of list elements using Vec (original approach)
    #[inline]
    fn decode_list_block_vec(
        &mut self,
        col_idx: usize,
        cursor: &mut AvroCursor<'_>,
        count: usize,
        child_type: &DataType,
        child_nullable: bool,
    ) -> Result<usize, ArrowError> {
        for _ in 0..count {
            let col = &mut self.columns[col_idx];
            if let NestedState::List { child_data, child_validity, .. } = &mut col.nested {
                if child_nullable {
                    // Read union index for nullable items, track validity separately
                    let union_index = cursor.get_long()?;
                    if union_index == 0 {
                        child_validity.push(false); // null
                        Self::append_placeholder_to_vec(child_type, child_data);
                    } else {
                        child_validity.push(true); // valid
                        Self::decode_value_to_vec(child_type, cursor, child_data)?;
                    }
                } else {
                    Self::decode_value_to_vec(child_type, cursor, child_data)?;
                }
            }
        }
        // Return final position
        let col = &self.columns[col_idx];
        if let NestedState::List { child_data, .. } = &col.nested {
            Ok(child_data.len())
        } else {
            Ok(0)
        }
    }

    /// Batch decode fixed-size elements directly into a slice (optimized hot path)
    #[inline]
    fn decode_fixed_elements_batch(
        data_type: &DataType,
        cursor: &mut AvroCursor<'_>,
        output: &mut [u8],
        count: usize,
        elem_size: usize,
    ) -> Result<(), ArrowError> {
        match data_type {
            DataType::Int32 | DataType::Date32 | DataType::Time32(_) => {
                for i in 0..count {
                    let value = cursor.get_int()?;
                    let offset = i * elem_size;
                    output[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                }
            }
            DataType::Int64 | DataType::Time64(_) | DataType::Timestamp(_, _) => {
                for i in 0..count {
                    let value = cursor.get_long()?;
                    let offset = i * elem_size;
                    output[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
                }
            }
            DataType::Float32 => {
                for i in 0..count {
                    let value = cursor.get_float()?;
                    let offset = i * elem_size;
                    output[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                }
            }
            DataType::Float64 => {
                for i in 0..count {
                    let value = cursor.get_double()?;
                    let offset = i * elem_size;
                    output[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
                }
            }
            DataType::Boolean => {
                for i in 0..count {
                    let value = cursor.get_bool()?;
                    output[i] = if value { 1 } else { 0 };
                }
            }
            DataType::Null => {
                // Nothing to decode for null type
            }
            DataType::FixedSizeBinary(size) => {
                let size = *size as usize;
                for i in 0..count {
                    let bytes = cursor.get_fixed(size)?;
                    let offset = i * size;
                    output[offset..offset + size].copy_from_slice(bytes);
                }
            }
            _ => {
                return Err(ArrowError::NotYetImplemented(format!(
                    "decode_fixed_elements_batch not implemented for {:?}",
                    data_type
                )));
            }
        }
        Ok(())
    }

    /// Get the size in bytes needed to decode a value (for capacity checking)
    fn get_value_size(data_type: &DataType, cursor: &mut AvroCursor<'_>) -> Result<usize, ArrowError> {
        match data_type {
            DataType::Null => Ok(0),
            DataType::Boolean => Ok(1),
            DataType::Int32 | DataType::Date32 | DataType::Time32(_) => Ok(4),
            DataType::Int64 | DataType::Time64(_) | DataType::Timestamp(_, _) => Ok(8),
            DataType::Float32 => Ok(4),
            DataType::Float64 => Ok(8),
            DataType::Utf8 | DataType::Binary => {
                // Peek at length without consuming
                let len = cursor.peek_long()? as usize;
                Ok(4 + len) // 4 bytes for length prefix + data
            }
            DataType::FixedSizeBinary(size) => Ok(*size as usize),
            other => Err(ArrowError::NotYetImplemented(format!(
                "get_value_size not implemented for {:?}",
                other
            ))),
        }
    }

    /// Helper to decode a primitive value directly into a pre-allocated slice
    fn decode_value_to_slice(
        data_type: &DataType,
        cursor: &mut AvroCursor<'_>,
        output: &mut [u8],
    ) -> Result<usize, ArrowError> {
        match data_type {
            DataType::Null => Ok(0),
            DataType::Int32 | DataType::Date32 | DataType::Time32(_) => {
                let value = cursor.get_int()?;
                output[..4].copy_from_slice(&value.to_le_bytes());
                Ok(4)
            }
            DataType::Int64 | DataType::Time64(_) | DataType::Timestamp(_, _) => {
                let value = cursor.get_long()?;
                output[..8].copy_from_slice(&value.to_le_bytes());
                Ok(8)
            }
            DataType::Float32 => {
                let value = cursor.get_float()?;
                output[..4].copy_from_slice(&value.to_le_bytes());
                Ok(4)
            }
            DataType::Float64 => {
                let value = cursor.get_double()?;
                output[..8].copy_from_slice(&value.to_le_bytes());
                Ok(8)
            }
            DataType::Boolean => {
                let value = cursor.get_bool()?;
                output[0] = if value { 1 } else { 0 };
                Ok(1)
            }
            DataType::Utf8 | DataType::Binary => {
                let bytes = cursor.get_bytes()?;
                let len = bytes.len();
                output[..4].copy_from_slice(&(len as i32).to_le_bytes());
                output[4..4 + len].copy_from_slice(bytes);
                Ok(4 + len)
            }
            DataType::FixedSizeBinary(size) => {
                let bytes = cursor.get_fixed(*size as usize)?;
                output[..*size as usize].copy_from_slice(bytes);
                Ok(*size as usize)
            }
            other => Err(ArrowError::NotYetImplemented(format!(
                "Nested value decoding to slice not implemented for {:?}",
                other
            ))),
        }
    }

    /// Helper to decode a value directly into a Vec<u8> (recursive for nested types)
    ///
    /// Serialization format for complex types:
    /// - List: [count:i32][elem0][elem1]... (count=0 means empty)
    /// - Map: [count:i32][key_len:i32][key_bytes][value]... per entry
    /// - Struct: [field0][field1]... (fields in schema order)
    /// - Union: [tag:i32][value] (tag is variant index)
    /// - Dictionary: [index:i32]
    fn decode_value_to_vec(
        data_type: &DataType,
        cursor: &mut AvroCursor<'_>,
        output: &mut Vec<u8>,
    ) -> Result<(), ArrowError> {
        match data_type {
            DataType::Null => {
                // Null takes no space
            }
            DataType::Int32 | DataType::Date32 | DataType::Time32(_) => {
                let value = cursor.get_int()?;
                output.extend_from_slice(&value.to_le_bytes());
            }
            DataType::Int64 | DataType::Time64(_) | DataType::Timestamp(_, _) => {
                let value = cursor.get_long()?;
                output.extend_from_slice(&value.to_le_bytes());
            }
            DataType::Float32 => {
                let value = cursor.get_float()?;
                output.extend_from_slice(&value.to_le_bytes());
            }
            DataType::Float64 => {
                let value = cursor.get_double()?;
                output.extend_from_slice(&value.to_le_bytes());
            }
            DataType::Boolean => {
                let value = cursor.get_bool()?;
                output.push(if value { 1 } else { 0 });
            }
            DataType::Utf8 | DataType::Binary => {
                let bytes = cursor.get_bytes()?;
                // Store length + data for variable-length
                output.extend_from_slice(&(bytes.len() as i32).to_le_bytes());
                output.extend_from_slice(bytes);
            }
            DataType::FixedSizeBinary(size) => {
                let bytes = cursor.get_fixed(*size as usize)?;
                output.extend_from_slice(bytes);
            }
            DataType::Decimal128(_, _) => {
                let bytes = cursor.get_bytes()?;
                let value = decode_decimal_bytes_to_i128(bytes);
                output.extend_from_slice(&value.to_le_bytes());
            }
            DataType::Decimal256(_, _) => {
                let bytes = cursor.get_bytes()?;
                let value = decode_decimal_bytes_to_i256(bytes);
                output.extend_from_slice(&value);
            }
            DataType::Interval(IntervalUnit::MonthDayNano) => {
                // Avro duration: 12 bytes (months u32 LE, days u32 LE, millis u32 LE)
                let bytes = cursor.get_fixed(12)?;
                let months = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as i32;
                let days = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as i32;
                let millis = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
                let nanos = (millis as i64) * 1_000_000;
                let interval = IntervalMonthDayNano::new(months, days, nanos);
                let interval_bytes: [u8; 16] = unsafe { std::mem::transmute(interval) };
                output.extend_from_slice(&interval_bytes);
            }
            // Recursive: List<T>
            DataType::List(child_field) => {
                Self::decode_list_to_vec(child_field.data_type(), child_field.is_nullable(), cursor, output)?;
            }
            // Recursive: Map<K, V>
            DataType::Map(entries_field, _) => {
                let (value_type, value_nullable) = if let DataType::Struct(fields) = entries_field.data_type() {
                    fields.iter().nth(1)
                        .map(|f| (f.data_type().clone(), f.is_nullable()))
                        .unwrap_or((DataType::Utf8, false))
                } else {
                    (DataType::Utf8, false)
                };
                Self::decode_map_to_vec(&value_type, value_nullable, cursor, output)?;
            }
            // Recursive: Struct
            DataType::Struct(fields) => {
                for field in fields.iter() {
                    if field.is_nullable() {
                        // Read union index for nullable fields
                        let union_index = cursor.get_long()?;
                        if union_index == 0 {
                            // Field is null - store marker and skip value
                            output.push(0); // null marker
                        } else {
                            output.push(1); // valid marker
                            Self::decode_value_to_vec(field.data_type(), cursor, output)?;
                        }
                    } else {
                        Self::decode_value_to_vec(field.data_type(), cursor, output)?;
                    }
                }
            }
            // Recursive: Union
            DataType::Union(union_fields, _) => {
                let tag = cursor.get_int()? as i32;
                output.extend_from_slice(&tag.to_le_bytes());
                if let Some((_, field)) = union_fields.iter().find(|(id, _)| *id == tag as i8) {
                    if !matches!(field.data_type(), DataType::Null) {
                        Self::decode_value_to_vec(field.data_type(), cursor, output)?;
                    }
                }
            }
            // Dictionary (Avro enum) - just store the index
            DataType::Dictionary(_, _) => {
                let index = cursor.get_int()?;
                output.extend_from_slice(&index.to_le_bytes());
            }
            other => {
                return Err(ArrowError::NotYetImplemented(format!(
                    "Nested value decoding not implemented for {:?}",
                    other
                )));
            }
        }
        Ok(())
    }

    /// Decode an Avro array (List) into Vec<u8> with format: [total_count:i32][elem0][elem1]...
    fn decode_list_to_vec(
        child_type: &DataType,
        child_nullable: bool,
        cursor: &mut AvroCursor<'_>,
        output: &mut Vec<u8>,
    ) -> Result<(), ArrowError> {
        // Reserve space for count, will fill in later
        let count_pos = output.len();
        output.extend_from_slice(&0i32.to_le_bytes());

        let mut total_count = 0i32;

        // Read Avro array blocks
        loop {
            let block_count = cursor.get_long()?;
            match block_count.cmp(&0) {
                Ordering::Equal => break,
                Ordering::Less => {
                    let count = (-block_count) as usize;
                    let _size = cursor.get_long()?; // skip size hint
                    for _ in 0..count {
                        if child_nullable {
                            let union_index = cursor.get_long()?;
                            if union_index == 0 {
                                output.push(0); // null marker
                            } else {
                                output.push(1); // valid marker
                                Self::decode_value_to_vec(child_type, cursor, output)?;
                            }
                        } else {
                            Self::decode_value_to_vec(child_type, cursor, output)?;
                        }
                    }
                    total_count += count as i32;
                }
                Ordering::Greater => {
                    let count = block_count as usize;
                    for _ in 0..count {
                        if child_nullable {
                            let union_index = cursor.get_long()?;
                            if union_index == 0 {
                                output.push(0); // null marker
                            } else {
                                output.push(1); // valid marker
                                Self::decode_value_to_vec(child_type, cursor, output)?;
                            }
                        } else {
                            Self::decode_value_to_vec(child_type, cursor, output)?;
                        }
                    }
                    total_count += count as i32;
                }
            }
        }

        // Write final count at reserved position
        output[count_pos..count_pos + 4].copy_from_slice(&total_count.to_le_bytes());
        Ok(())
    }

    /// Decode an Avro map into Vec<u8> with format: [total_count:i32][key_len:i32][key][value]...
    fn decode_map_to_vec(
        value_type: &DataType,
        value_nullable: bool,
        cursor: &mut AvroCursor<'_>,
        output: &mut Vec<u8>,
    ) -> Result<(), ArrowError> {
        // Reserve space for count
        let count_pos = output.len();
        output.extend_from_slice(&0i32.to_le_bytes());

        let mut total_count = 0i32;

        // Read Avro map blocks
        loop {
            let block_count = cursor.get_long()?;
            match block_count.cmp(&0) {
                Ordering::Equal => break,
                Ordering::Less => {
                    let count = (-block_count) as usize;
                    let _size = cursor.get_long()?;
                    for _ in 0..count {
                        // Key is always string
                        let key = cursor.get_bytes()?;
                        output.extend_from_slice(&(key.len() as i32).to_le_bytes());
                        output.extend_from_slice(key);
                        // Value may be nullable
                        if value_nullable {
                            let union_index = cursor.get_long()?;
                            if union_index == 0 {
                                output.push(0); // null marker
                            } else {
                                output.push(1); // valid marker
                                Self::decode_value_to_vec(value_type, cursor, output)?;
                            }
                        } else {
                            Self::decode_value_to_vec(value_type, cursor, output)?;
                        }
                    }
                    total_count += count as i32;
                }
                Ordering::Greater => {
                    let count = block_count as usize;
                    for _ in 0..count {
                        let key = cursor.get_bytes()?;
                        output.extend_from_slice(&(key.len() as i32).to_le_bytes());
                        output.extend_from_slice(key);
                        if value_nullable {
                            let union_index = cursor.get_long()?;
                            if union_index == 0 {
                                output.push(0); // null marker
                            } else {
                                output.push(1); // valid marker
                                Self::decode_value_to_vec(value_type, cursor, output)?;
                            }
                        } else {
                            Self::decode_value_to_vec(value_type, cursor, output)?;
                        }
                    }
                    total_count += count as i32;
                }
            }
        }

        output[count_pos..count_pos + 4].copy_from_slice(&total_count.to_le_bytes());
        Ok(())
    }

    /// Decode multiple rows from a buffer
    ///
    /// Returns the number of bytes consumed from the buffer.
    /// If capacity is exceeded mid-decode, complete rows are kept and
    /// the cursor position reflects only successfully decoded rows.
    pub fn decode(&mut self, buf: &[u8], count: usize) -> Result<usize, ArrowError> {
        // Use fast path for flat schemas (all fixed-size primitive types)
        if self.is_flat_schema() {
            return self.decode_batch_flat(buf, count);
        }

        // Use ultra-fast path for strings-only schemas
        if self.is_strings_only_schema() {
            return self.decode_batch_strings_only(buf, count);
        }

        // Use batched path for simple schemas (fixed + variable-length primitives)
        if self.is_simple_schema() {
            return self.decode_batch_mixed(buf, count);
        }

        // Use fast path for dictionary-only schemas (Avro enums)
        if self.is_dictionary_only_schema() {
            return self.decode_batch_dictionary_only(buf, count);
        }

        // Use batched path for simple nested schemas (flat types + Dictionary)
        if self.is_simple_nested_schema() {
            return self.decode_batch_simple_nested(buf, count);
        }

        // Use fast path for struct-only schemas (Apache's Record decoder)
        if self.is_struct_only_schema() {
            return self.decode_batch_struct(buf, count);
        }

        // Use universal batch decode for complex/nested schemas (Map, Union, Struct, List)
        // No row-by-row fallback - all types handled in batch
        self.decode_batch_universal(buf, count)
    }

    /// Decode multiple rows with capacity handling
    ///
    /// Returns `(bytes_consumed, rows_decoded, capacity_exceeded)`.
    /// If capacity is exceeded, the last partial row is rolled back
    /// and only complete rows are returned.
    ///
    /// `capacity_exceeded` is true if:
    /// - More rows were requested than could fit (count > available capacity)
    /// - A row decode failed due to buffer capacity limits
    pub fn decode_with_capacity(
        &mut self,
        buf: &[u8],
        count: usize,
    ) -> Result<(usize, usize, bool), ArrowError> {
        let mut cursor = AvroCursor::new(buf);
        let available_rows = self.max_rows - self.row_count;
        let rows_to_decode = count.min(available_rows);
        let mut rows_decoded = 0;
        let mut capacity_exceeded = count > available_rows; // Exceeded if more requested than available

        for _ in 0..rows_to_decode {
            match self.try_decode_row(&mut cursor)? {
                true => rows_decoded += 1,
                false => {
                    capacity_exceeded = true;
                    break;
                }
            }
        }

        Ok((cursor.position(), rows_decoded, capacity_exceeded))
    }

    /// Flush decoded data to a RecordBatch
    pub fn flush(&mut self) -> Result<RecordBatch, ArrowError> {
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(self.columns.len());

        let num_columns = self.columns.len();
        for col_idx in 0..num_columns {
            let array = self.flush_column(col_idx)?;
            arrays.push(array);
        }

        // Reset for next batch
        self.row_count = 0;
        for col in &mut self.columns {
            col.reset();
        }
        self.provider.reset();

        RecordBatch::try_new(self.schema.clone(), arrays)
    }

    /// Flush a single column to an Arrow array
    fn flush_column(&mut self, col_idx: usize) -> Result<ArrayRef, ArrowError> {
        let col = &self.columns[col_idx];
        let column_index = col.column_index;
        let row_count = col.row_count;
        let data_position = col.data_position;
        let data_type = col.data_type.clone();
        // Get null buffer for nullable columns
        let null_buffer = col.get_null_buffer();

        match &data_type {
            DataType::Null => {
                Ok(Arc::new(NullArray::new(row_count)))
            }
            DataType::Int32 => {
                let buffer = self.provider.finalize_data_buffer(column_index, row_count * 4)?;
                let values = ScalarBuffer::<i32>::from(buffer);
                Ok(Arc::new(Int32Array::new(values, null_buffer)))
            }
            DataType::Int64 => {
                let buffer = self.provider.finalize_data_buffer(column_index, row_count * 8)?;
                let values = ScalarBuffer::<i64>::from(buffer);
                Ok(Arc::new(Int64Array::new(values, null_buffer)))
            }
            DataType::Float32 => {
                let buffer = self.provider.finalize_data_buffer(column_index, row_count * 4)?;
                let values = ScalarBuffer::<f32>::from(buffer);
                Ok(Arc::new(Float32Array::new(values, null_buffer)))
            }
            DataType::Float64 => {
                let buffer = self.provider.finalize_data_buffer(column_index, row_count * 8)?;
                let values = ScalarBuffer::<f64>::from(buffer);
                Ok(Arc::new(Float64Array::new(values, null_buffer)))
            }
            DataType::Date32 => {
                let buffer = self.provider.finalize_data_buffer(column_index, row_count * 4)?;
                let values = ScalarBuffer::<i32>::from(buffer);
                Ok(Arc::new(Date32Array::new(values, null_buffer)))
            }
            DataType::Time32(unit) => {
                let buffer = self.provider.finalize_data_buffer(column_index, row_count * 4)?;
                let values = ScalarBuffer::<i32>::from(buffer);
                let array: ArrayRef = match unit {
                    arrow_schema::TimeUnit::Second => {
                        Arc::new(Time32SecondArray::new(values, null_buffer))
                    }
                    arrow_schema::TimeUnit::Millisecond => {
                        Arc::new(Time32MillisecondArray::new(values, null_buffer))
                    }
                    _ => {
                        return Err(ArrowError::InvalidArgumentError(format!(
                            "Time32 only supports Second and Millisecond, got {:?}",
                            unit
                        )));
                    }
                };
                Ok(array)
            }
            DataType::Time64(unit) => {
                let buffer = self.provider.finalize_data_buffer(column_index, row_count * 8)?;
                let values = ScalarBuffer::<i64>::from(buffer);
                let array: ArrayRef = match unit {
                    arrow_schema::TimeUnit::Microsecond => {
                        Arc::new(Time64MicrosecondArray::new(values, null_buffer))
                    }
                    arrow_schema::TimeUnit::Nanosecond => {
                        Arc::new(Time64NanosecondArray::new(values, null_buffer))
                    }
                    _ => {
                        return Err(ArrowError::InvalidArgumentError(format!(
                            "Time64 only supports Microsecond and Nanosecond, got {:?}",
                            unit
                        )));
                    }
                };
                Ok(array)
            }
            DataType::Timestamp(unit, tz) => {
                let buffer = self.provider.finalize_data_buffer(column_index, row_count * 8)?;
                let values = ScalarBuffer::<i64>::from(buffer);
                let array: ArrayRef = match unit {
                    arrow_schema::TimeUnit::Second => {
                        Arc::new(TimestampSecondArray::new(values, null_buffer.clone()).with_timezone_opt(tz.clone()))
                    }
                    arrow_schema::TimeUnit::Millisecond => {
                        Arc::new(TimestampMillisecondArray::new(values, null_buffer.clone()).with_timezone_opt(tz.clone()))
                    }
                    arrow_schema::TimeUnit::Microsecond => {
                        Arc::new(TimestampMicrosecondArray::new(values, null_buffer.clone()).with_timezone_opt(tz.clone()))
                    }
                    arrow_schema::TimeUnit::Nanosecond => {
                        Arc::new(TimestampNanosecondArray::new(values, null_buffer.clone()).with_timezone_opt(tz.clone()))
                    }
                };
                Ok(array)
            }
            DataType::Boolean => {
                let bitmap_bytes = (row_count + 7) / 8;
                let buffer = self.provider.finalize_data_buffer(column_index, bitmap_bytes)?;
                let bool_buffer = arrow_buffer::BooleanBuffer::new(buffer, 0, row_count);
                Ok(Arc::new(BooleanArray::new(bool_buffer, null_buffer)))
            }
            DataType::Utf8 => {
                let data_buffer = self.provider.finalize_data_buffer(column_index, data_position)?;
                let offsets_buffer = self.provider.finalize_offsets_buffer(column_index, (row_count + 1) * 4)?;
                let offsets = OffsetBuffer::new(ScalarBuffer::<i32>::from(offsets_buffer));
                Ok(Arc::new(StringArray::try_new(offsets, data_buffer, null_buffer)?))
            }
            DataType::Binary => {
                let data_buffer = self.provider.finalize_data_buffer(column_index, data_position)?;
                let offsets_buffer = self.provider.finalize_offsets_buffer(column_index, (row_count + 1) * 4)?;
                let offsets = OffsetBuffer::new(ScalarBuffer::<i32>::from(offsets_buffer));
                Ok(Arc::new(BinaryArray::try_new(offsets, data_buffer, null_buffer)?))
            }
            DataType::FixedSizeBinary(size) => {
                let buffer = self.provider.finalize_data_buffer(column_index, row_count * (*size as usize))?;
                Ok(Arc::new(FixedSizeBinaryArray::try_new(*size, buffer, null_buffer)?))
            }
            DataType::Decimal128(precision, scale) => {
                let buffer = self.provider.finalize_data_buffer(column_index, row_count * 16)?;
                let values = ScalarBuffer::<i128>::from(buffer);
                Ok(Arc::new(Decimal128Array::new(values, null_buffer).with_precision_and_scale(*precision, *scale)?))
            }
            DataType::Decimal256(precision, scale) => {
                let buffer = self.provider.finalize_data_buffer(column_index, row_count * 32)?;
                let values = ScalarBuffer::<i256>::from(buffer);
                Ok(Arc::new(Decimal256Array::new(values, null_buffer).with_precision_and_scale(*precision, *scale)?))
            }
            DataType::Interval(IntervalUnit::MonthDayNano) => {
                let buffer = self.provider.finalize_data_buffer(column_index, row_count * 16)?;
                let values = ScalarBuffer::<IntervalMonthDayNano>::from(buffer);
                Ok(Arc::new(IntervalMonthDayNanoArray::new(values, null_buffer)))
            }
            DataType::Map(entries_field, keys_sorted) => {
                self.flush_map_column(col_idx, entries_field.clone(), *keys_sorted)
            }
            DataType::Union(fields, mode) => {
                self.flush_union_column(col_idx, fields.clone(), *mode)
            }
            DataType::List(child_field) => {
                self.flush_list_column(col_idx, child_field.clone())
            }
            DataType::Dictionary(key_type, value_type) => {
                self.flush_dictionary_column(col_idx, key_type.clone(), value_type.clone())
            }
            DataType::Struct(fields) => {
                self.flush_struct_column(col_idx, fields.clone())
            }
            other => Err(ArrowError::NotYetImplemented(format!(
                "Pooled flush not yet implemented for {:?}",
                other
            ))),
        }
    }

    /// Flush a Dictionary column (Avro enum) to Arrow DictionaryArray
    fn flush_dictionary_column(
        &mut self,
        col_idx: usize,
        key_type: Box<DataType>,
        value_type: Box<DataType>,
    ) -> Result<ArrayRef, ArrowError> {
        let col = &mut self.columns[col_idx];

        if let NestedState::Dictionary { indices, dictionary } = &mut col.nested {
            // Create dictionary values array (StringArray from enum symbols)
            let dict_values: ArrayRef = match value_type.as_ref() {
                DataType::Utf8 => Arc::new(StringArray::from(
                    dictionary.iter().map(|s| s.as_str()).collect::<Vec<_>>()
                )),
                other => {
                    return Err(ArrowError::NotYetImplemented(format!(
                        "Dictionary value type {:?} not supported",
                        other
                    )));
                }
            };

            // Create DictionaryArray based on key type
            // DictionaryArray::try_new expects a PrimitiveArray, not Arc<dyn Array>
            match key_type.as_ref() {
                DataType::Int32 => {
                    let buffer = Buffer::from_slice_ref(indices);
                    let values = ScalarBuffer::<i32>::from(buffer);
                    let keys_array = Int32Array::new(values, None);
                    let dict_array = DictionaryArray::try_new(keys_array, dict_values)?;
                    Ok(Arc::new(dict_array))
                }
                DataType::Int8 => {
                    let values: Vec<i8> = indices.iter().map(|&i| i as i8).collect();
                    let keys_array = Int8Array::from(values);
                    let dict_array = DictionaryArray::try_new(keys_array, dict_values)?;
                    Ok(Arc::new(dict_array))
                }
                DataType::Int16 => {
                    let values: Vec<i16> = indices.iter().map(|&i| i as i16).collect();
                    let keys_array = Int16Array::from(values);
                    let dict_array = DictionaryArray::try_new(keys_array, dict_values)?;
                    Ok(Arc::new(dict_array))
                }
                DataType::UInt8 => {
                    let values: Vec<u8> = indices.iter().map(|&i| i as u8).collect();
                    let keys_array = UInt8Array::from(values);
                    let dict_array = DictionaryArray::try_new(keys_array, dict_values)?;
                    Ok(Arc::new(dict_array))
                }
                other => {
                    Err(ArrowError::NotYetImplemented(format!(
                        "Dictionary key type {:?} not supported",
                        other
                    )))
                }
            }
        } else {
            Err(ArrowError::InvalidArgumentError(
                "Dictionary column missing nested state".to_string()
            ))
        }
    }

    /// Flush a Map column from nested state to Arrow MapArray (Vec mode only)
    fn flush_map_column(
        &mut self,
        col_idx: usize,
        entries_field: FieldRef,
        keys_sorted: bool,
    ) -> Result<ArrayRef, ArrowError> {
        let col = &mut self.columns[col_idx];
        let col_null_buffer = col.get_null_buffer();

        if let NestedState::Map {
            map_offsets, key_offsets, key_data, value_offsets, value_data, value_validity, value_type, value_nullable, ..
        } = &mut col.nested {
            let is_value_nullable = *value_nullable;
            let value_type_clone = value_type.clone();

            // Zero-copy: take ownership of Vecs and convert directly to Buffer
            let key_data_vec = std::mem::take(key_data);
            let key_offsets_vec = std::mem::take(key_offsets);
            let key_data_buffer = Buffer::from_vec(key_data_vec);
            let key_offsets_buffer = Buffer::from_vec(key_offsets_vec);

            // Create key array (StringArray) - zero-copy
            let key_array = StringArray::try_new(
                OffsetBuffer::new(ScalarBuffer::<i32>::from(key_offsets_buffer)),
                key_data_buffer,
                None,
            )?;

            // Create value array based on type and nullability
            let value_array: ArrayRef = if is_value_nullable {
                // Nullable values: use separate validity tracking (single pass, no markers)
                let value_data_vec = std::mem::take(value_data);
                let validity_vec = std::mem::take(value_validity);
                Self::build_array_from_vec_with_nulls_owned(value_type_clone.clone(), value_data_vec, validity_vec)?
            } else {
                let value_data_vec = std::mem::take(value_data);
                match &value_type_clone {
                    DataType::Utf8 => {
                        // Variable-length string values: zero-copy from offsets
                        let value_offsets_vec = std::mem::take(value_offsets);
                        let value_data_buffer = Buffer::from_vec(value_data_vec);
                        let value_offsets_buffer = Buffer::from_vec(value_offsets_vec);
                        Arc::new(StringArray::try_new(
                            OffsetBuffer::new(ScalarBuffer::<i32>::from(value_offsets_buffer)),
                            value_data_buffer,
                            None,
                        )?)
                    }
                    DataType::Binary => {
                        // Variable-length binary values: zero-copy from offsets
                        let value_offsets_vec = std::mem::take(value_offsets);
                        let value_data_buffer = Buffer::from_vec(value_data_vec);
                        let value_offsets_buffer = Buffer::from_vec(value_offsets_vec);
                        Arc::new(BinaryArray::try_new(
                            OffsetBuffer::new(ScalarBuffer::<i32>::from(value_offsets_buffer)),
                            value_data_buffer,
                            None,
                        )?)
                    }
                    _ => {
                        // Fixed-size values: zero-copy build
                        Self::build_array_from_vec_owned(value_type_clone.clone(), value_data_vec)?
                    }
                }
            };

            // Create entries struct array
            let entries_struct = StructArray::new(
                Fields::from(vec![
                    Arc::new(Field::new("key", DataType::Utf8, false)),
                    Arc::new(Field::new("value", value_type_clone, is_value_nullable)),
                ]),
                vec![Arc::new(key_array), value_array],
                None,
            );

            // Zero-copy map offsets
            let map_offsets_vec = std::mem::take(map_offsets);
            let map_offsets_buffer = Buffer::from_vec(map_offsets_vec);
            let map_offsets = OffsetBuffer::new(ScalarBuffer::<i32>::from(map_offsets_buffer));

            // Create MapArray with column-level null buffer if applicable
            let map_array = MapArray::new(
                entries_field,
                map_offsets,
                entries_struct,
                col_null_buffer,
                keys_sorted,
            );

            Ok(Arc::new(map_array))
        } else {
            Err(ArrowError::InvalidArgumentError(
                "Map column missing nested state".to_string()
            ))
        }
    }

    /// Flush a Union column from nested state to Arrow UnionArray (Vec mode only)
    /// Uses zero-copy Buffer conversion where possible
    fn flush_union_column(
        &mut self,
        col_idx: usize,
        fields: UnionFields,
        mode: UnionMode,
    ) -> Result<ArrayRef, ArrowError> {
        let col = &mut self.columns[col_idx];

        if let NestedState::Union {
            type_ids, offsets, child_data, child_counts, ..
        } = &mut col.nested {
            // Zero-copy: take ownership of type_ids Vec
            let type_ids_vec = std::mem::take(type_ids);
            let type_ids_buffer = ScalarBuffer::<i8>::from(Buffer::from_vec(type_ids_vec));

            // Build child arrays using zero-copy owned versions
            let mut children: Vec<ArrayRef> = Vec::with_capacity(fields.len());
            for (i, (_, field)) in fields.iter().enumerate() {
                let child_type = field.data_type().clone();
                match child_type {
                    DataType::Null => {
                        children.push(Arc::new(NullArray::new(child_counts[i])));
                    }
                    _ => {
                        // Zero-copy: take ownership of child data Vec
                        let child_data_vec = std::mem::take(&mut child_data[i]);
                        let child_array = Self::build_array_from_vec_owned(child_type, child_data_vec)?;
                        children.push(child_array);
                    }
                }
            }

            // Create union array based on mode
            let union_array = match mode {
                UnionMode::Dense => {
                    // Zero-copy offsets
                    let offsets_buffer = offsets.as_mut()
                        .map(|o| {
                            let offs_vec = std::mem::take(o);
                            ScalarBuffer::<i32>::from(Buffer::from_vec(offs_vec))
                        });
                    UnionArray::try_new(fields, type_ids_buffer, offsets_buffer, children)?
                }
                UnionMode::Sparse => {
                    UnionArray::try_new(fields, type_ids_buffer, None, children)?
                }
            };

            Ok(Arc::new(union_array))
        } else {
            Err(ArrowError::InvalidArgumentError(
                "Union column missing nested state".to_string()
            ))
        }
    }

    /// Flush a List column from nested state to Arrow ListArray
    fn flush_list_column(
        &mut self,
        col_idx: usize,
        child_field: FieldRef,
    ) -> Result<ArrayRef, ArrowError> {
        // Extract info we need first to avoid borrow conflicts
        let (offsets_clone, child_type_clone, child_data_position, child_offsets_position, use_prealloc, is_child_nullable, col_null_buffer) = {
            let col = &self.columns[col_idx];
            let null_buf = col.get_null_buffer();
            if let NestedState::List { offsets, child_type, child_data_position, child_offsets_position, use_preallocated, child_nullable, .. } = &col.nested {
                (offsets.clone(), child_type.clone(), *child_data_position, *child_offsets_position, *use_preallocated, *child_nullable, null_buf)
            } else {
                return Err(ArrowError::InvalidArgumentError(
                    "List column missing nested state".to_string()
                ));
            }
        };

        // Build child array based on mode
        let child_array = if use_prealloc {
            // Pre-allocated mode: get buffer from provider
            match &child_type_clone {
                DataType::Utf8 | DataType::Binary => {
                    // Variable-length types need both data and offsets buffers
                    let child_data_buffer = self.provider.finalize_child_data_buffer(col_idx, 0, child_data_position)?;
                    let child_offsets_buffer = self.provider.finalize_child_offsets_buffer(col_idx, 0, child_offsets_position)?;
                    Self::build_string_array_from_buffers(child_type_clone, child_data_buffer, child_offsets_buffer)?
                }
                _ => {
                    // Fixed-size types only need data buffer
                    let child_data_buffer = self.provider.finalize_child_data_buffer(col_idx, 0, child_data_position)?;
                    Self::build_array_from_buffer(child_type_clone, child_data_buffer)?
                }
            }
        } else {
            // Vec mode: handle nullable with separate validity tracking
            let col = &mut self.columns[col_idx];
            if let NestedState::List { child_data, child_validity, child_type, .. } = &mut col.nested {
                if is_child_nullable && !child_validity.is_empty() {
                    // Use separate validity tracking (single pass, no markers)
                    let child_data_vec = std::mem::take(child_data);
                    let validity_vec = std::mem::take(child_validity);
                    Self::build_array_from_vec_with_nulls_owned(child_type.clone(), child_data_vec, validity_vec)?
                } else {
                    let child_data_vec = std::mem::take(child_data);
                    Self::build_array_from_vec_owned(child_type.clone(), child_data_vec)?
                }
            } else {
                return Err(ArrowError::InvalidArgumentError(
                    "List column missing nested state".to_string()
                ));
            }
        };

        // Create offsets
        let offsets_buffer = Buffer::from_slice_ref(&offsets_clone);
        let list_offsets = OffsetBuffer::new(ScalarBuffer::<i32>::from(offsets_buffer));

        // Create ListArray with column-level null buffer if applicable
        let list_array = ListArray::new(child_field, list_offsets, child_array, col_null_buffer);

        Ok(Arc::new(list_array))
    }

    /// Build a string/binary array from pre-allocated data and offsets buffers
    fn build_string_array_from_buffers(
        data_type: DataType,
        data_buffer: Buffer,
        offsets_buffer: Buffer,
    ) -> Result<ArrayRef, ArrowError> {
        // Handle empty case
        if offsets_buffer.is_empty() {
            return match data_type {
                DataType::Utf8 => Ok(Arc::new(StringArray::new_null(0))),
                DataType::Binary => Ok(Arc::new(BinaryArray::new_null(0))),
                _ => Err(ArrowError::InvalidArgumentError(format!(
                    "build_string_array_from_buffers called with non-string type: {:?}",
                    data_type
                ))),
            };
        }

        // Create offsets from buffer
        let offsets = OffsetBuffer::new(ScalarBuffer::<i32>::from(offsets_buffer));

        match data_type {
            DataType::Utf8 => {
                let array = StringArray::new(offsets, data_buffer, None);
                Ok(Arc::new(array))
            }
            DataType::Binary => {
                let array = BinaryArray::new(offsets, data_buffer, None);
                Ok(Arc::new(array))
            }
            _ => Err(ArrowError::InvalidArgumentError(format!(
                "build_string_array_from_buffers called with non-string type: {:?}",
                data_type
            ))),
        }
    }
    /// Build an Arrow array from a pre-allocated buffer
    fn build_array_from_buffer(data_type: DataType, buffer: Buffer) -> Result<ArrayRef, ArrowError> {
        // Handle empty buffers specially - ScalarBuffer::from panics on unaligned empty buffers
        if buffer.is_empty() {
            return match data_type {
                DataType::Int32 | DataType::Date32 | DataType::Time32(_) => {
                    Ok(Arc::new(Int32Array::new_null(0)))
                }
                DataType::Int64 | DataType::Time64(_) | DataType::Timestamp(_, _) => {
                    Ok(Arc::new(Int64Array::new_null(0)))
                }
                DataType::Float32 => Ok(Arc::new(Float32Array::new_null(0))),
                DataType::Float64 => Ok(Arc::new(Float64Array::new_null(0))),
                DataType::Boolean => Ok(Arc::new(BooleanArray::new_null(0))),
                other => Err(ArrowError::NotYetImplemented(format!(
                    "build_array_from_buffer not implemented for {:?}",
                    other
                ))),
            };
        }

        match data_type {
            DataType::Int32 | DataType::Date32 | DataType::Time32(_) => {
                let scalar_buffer = ScalarBuffer::<i32>::from(buffer);
                let array = Int32Array::new(scalar_buffer, None);
                Ok(Arc::new(array))
            }
            DataType::Int64 | DataType::Time64(_) | DataType::Timestamp(_, _) => {
                let scalar_buffer = ScalarBuffer::<i64>::from(buffer);
                let array = Int64Array::new(scalar_buffer, None);
                Ok(Arc::new(array))
            }
            DataType::Float32 => {
                let scalar_buffer = ScalarBuffer::<f32>::from(buffer);
                let array = Float32Array::new(scalar_buffer, None);
                Ok(Arc::new(array))
            }
            DataType::Float64 => {
                let scalar_buffer = ScalarBuffer::<f64>::from(buffer);
                let array = Float64Array::new(scalar_buffer, None);
                Ok(Arc::new(array))
            }
            DataType::Boolean => {
                // For booleans in list, each value is stored as a byte (0 or 1)
                let values: Vec<bool> = buffer.iter().map(|&b| b != 0).collect();
                let array = BooleanArray::from(values);
                Ok(Arc::new(array))
            }
            other => Err(ArrowError::NotYetImplemented(format!(
                "build_array_from_buffer not implemented for {:?}",
                other
            ))),
        }
    }

    /// Flush a Struct column from nested state to Arrow StructArray
    fn flush_struct_column(
        &mut self,
        col_idx: usize,
        _fields: Fields,
    ) -> Result<ArrayRef, ArrowError> {
        let col = &mut self.columns[col_idx];
        let col_null_buffer = col.get_null_buffer();
        let provider_col_idx = col.column_index;

        if let NestedState::Struct {
            field_data, field_types, field_names, field_nullable, row_count,
            field_positions, field_offsets_positions, use_preallocated, record_decoder, ..
        } = &mut col.nested {
            let total_rows = *row_count;

            if *use_preallocated {
                // Pre-allocated path: finalize buffers from provider
                let mut child_arrays: Vec<ArrayRef> = Vec::with_capacity(field_types.len());
                for (field_idx, field_type) in field_types.iter().enumerate() {
                    let data_pos = field_positions[field_idx];
                    let child_array = match field_type {
                        DataType::Utf8 | DataType::Binary => {
                            // Variable-length: need both data and offsets buffers
                            let data_buffer = self.provider.finalize_child_data_buffer(provider_col_idx, field_idx, data_pos)?;
                            let offsets_pos = field_offsets_positions[field_idx];
                            let offsets_buffer = self.provider.finalize_child_offsets_buffer(provider_col_idx, field_idx, offsets_pos)?;
                            Self::build_string_array_from_buffers(field_type.clone(), data_buffer, offsets_buffer)?
                        }
                        _ => {
                            // Fixed-size: only data buffer
                            let data_buffer = self.provider.finalize_child_data_buffer(provider_col_idx, field_idx, data_pos)?;
                            Self::build_array_from_buffer(field_type.clone(), data_buffer)?
                        }
                    };
                    child_arrays.push(child_array);
                }

                // Validate all child arrays have same length
                for (i, child) in child_arrays.iter().enumerate() {
                    if child.len() != total_rows {
                        return Err(ArrowError::InvalidArgumentError(format!(
                            "Struct field {} has {} rows, expected {} (use_preallocated={})",
                            field_names[i], child.len(), total_rows, *use_preallocated
                        )));
                    }
                }

                // Build struct fields
                let struct_fields: Vec<FieldRef> = field_names.iter()
                    .zip(field_types.iter())
                    .zip(field_nullable.iter())
                    .map(|((name, dtype), nullable)| {
                        Arc::new(Field::new(name, dtype.clone(), *nullable))
                    })
                    .collect();

                // Create StructArray with column-level null buffer if applicable
                let struct_array = StructArray::new(
                    Fields::from(struct_fields),
                    child_arrays,
                    col_null_buffer,
                );

                Ok(Arc::new(struct_array))
            } else {
                // Vec fallback path: use Apache's Record Decoder flush directly
                // This returns a StructArray with all fields and their null buffers
                // The record_decoder handles nullable fields internally via Decoder::Nullable wrappers
                let struct_array = record_decoder.flush(col_null_buffer)?;

                // Clear field_data since we used record_decoder instead
                for data in field_data.iter_mut() {
                    data.clear();
                }

                Ok(struct_array)
            }
        } else {
            Err(ArrowError::InvalidArgumentError(
                "Struct column missing nested state".to_string()
            ))
        }
    }

    /// Build an Arrow array from a raw byte vector based on data type
    fn build_array_from_vec(data_type: DataType, data: &[u8]) -> Result<ArrayRef, ArrowError> {
        match data_type {
            DataType::Null => {
                // Count nulls - not stored in data
                Ok(Arc::new(NullArray::new(0)))
            }
            DataType::Int32 => {
                let buffer = Buffer::from_slice_ref(data);
                let values = ScalarBuffer::<i32>::from(buffer);
                Ok(Arc::new(Int32Array::new(values, None)))
            }
            DataType::Int64 => {
                let buffer = Buffer::from_slice_ref(data);
                let values = ScalarBuffer::<i64>::from(buffer);
                Ok(Arc::new(Int64Array::new(values, None)))
            }
            DataType::Float32 => {
                let buffer = Buffer::from_slice_ref(data);
                let values = ScalarBuffer::<f32>::from(buffer);
                Ok(Arc::new(Float32Array::new(values, None)))
            }
            DataType::Float64 => {
                let buffer = Buffer::from_slice_ref(data);
                let values = ScalarBuffer::<f64>::from(buffer);
                Ok(Arc::new(Float64Array::new(values, None)))
            }
            DataType::Boolean => {
                // Reconstruct boolean array from packed bytes
                let mut builder = BooleanArray::builder(data.len());
                for &byte in data {
                    builder.append_value(byte != 0);
                }
                Ok(Arc::new(builder.finish()))
            }
            DataType::Utf8 => {
                // Data is stored as: [len:i32][bytes:len]...
                let mut strings: Vec<&str> = Vec::new();
                let mut pos = 0;
                while pos + 4 <= data.len() {
                    let len = i32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
                    pos += 4;
                    if pos + len <= data.len() {
                        let s = std::str::from_utf8(&data[pos..pos+len])
                            .map_err(|e| ArrowError::ParseError(format!("Invalid UTF-8: {}", e)))?;
                        strings.push(s);
                        pos += len;
                    } else {
                        break;
                    }
                }
                Ok(Arc::new(StringArray::from(strings)))
            }
            DataType::Binary => {
                // Data is stored as: [len:i32][bytes:len]...
                let mut binaries: Vec<&[u8]> = Vec::new();
                let mut pos = 0;
                while pos + 4 <= data.len() {
                    let len = i32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
                    pos += 4;
                    if pos + len <= data.len() {
                        binaries.push(&data[pos..pos+len]);
                        pos += len;
                    } else {
                        break;
                    }
                }
                Ok(Arc::new(BinaryArray::from(binaries)))
            }
            DataType::FixedSizeBinary(size) => {
                let element_size = size as usize;
                let count = data.len() / element_size;
                let builder = FixedSizeBinaryArray::try_from_iter(
                    data.chunks_exact(element_size).take(count)
                )?;
                Ok(Arc::new(builder))
            }
            DataType::Decimal128(precision, scale) => {
                let buffer = Buffer::from_slice_ref(data);
                let values = ScalarBuffer::<i128>::from(buffer);
                Ok(Arc::new(Decimal128Array::new(values, None)
                    .with_precision_and_scale(precision, scale)?))
            }
            DataType::Date32 => {
                let buffer = Buffer::from_slice_ref(data);
                let values = ScalarBuffer::<i32>::from(buffer);
                Ok(Arc::new(Date32Array::new(values, None)))
            }
            DataType::Time32(unit) => {
                let buffer = Buffer::from_slice_ref(data);
                let values = ScalarBuffer::<i32>::from(buffer);
                match unit {
                    arrow_schema::TimeUnit::Second => {
                        Ok(Arc::new(Time32SecondArray::new(values, None)))
                    }
                    arrow_schema::TimeUnit::Millisecond => {
                        Ok(Arc::new(Time32MillisecondArray::new(values, None)))
                    }
                    _ => Err(ArrowError::InvalidArgumentError(format!(
                        "Invalid Time32 unit: {:?}",
                        unit
                    ))),
                }
            }
            DataType::Time64(unit) => {
                let buffer = Buffer::from_slice_ref(data);
                let values = ScalarBuffer::<i64>::from(buffer);
                match unit {
                    arrow_schema::TimeUnit::Microsecond => {
                        Ok(Arc::new(Time64MicrosecondArray::new(values, None)))
                    }
                    arrow_schema::TimeUnit::Nanosecond => {
                        Ok(Arc::new(Time64NanosecondArray::new(values, None)))
                    }
                    _ => Err(ArrowError::InvalidArgumentError(format!(
                        "Invalid Time64 unit: {:?}",
                        unit
                    ))),
                }
            }
            DataType::Timestamp(unit, tz) => {
                let buffer = Buffer::from_slice_ref(data);
                let values = ScalarBuffer::<i64>::from(buffer);
                match unit {
                    arrow_schema::TimeUnit::Second => {
                        let arr = TimestampSecondArray::new(values, None);
                        Ok(Arc::new(if let Some(tz) = tz { arr.with_timezone(tz.as_ref()) } else { arr }))
                    }
                    arrow_schema::TimeUnit::Millisecond => {
                        let arr = TimestampMillisecondArray::new(values, None);
                        Ok(Arc::new(if let Some(tz) = tz { arr.with_timezone(tz.as_ref()) } else { arr }))
                    }
                    arrow_schema::TimeUnit::Microsecond => {
                        let arr = TimestampMicrosecondArray::new(values, None);
                        Ok(Arc::new(if let Some(tz) = tz { arr.with_timezone(tz.as_ref()) } else { arr }))
                    }
                    arrow_schema::TimeUnit::Nanosecond => {
                        let arr = TimestampNanosecondArray::new(values, None);
                        Ok(Arc::new(if let Some(tz) = tz { arr.with_timezone(tz.as_ref()) } else { arr }))
                    }
                }
            }
            DataType::Interval(IntervalUnit::MonthDayNano) => {
                // Each interval is 16 bytes
                let count = data.len() / 16;
                let mut builder = IntervalMonthDayNanoArray::builder(count);
                for chunk in data.chunks_exact(16).take(count) {
                    let interval: IntervalMonthDayNano = unsafe {
                        std::ptr::read_unaligned(chunk.as_ptr() as *const IntervalMonthDayNano)
                    };
                    builder.append_value(interval);
                }
                Ok(Arc::new(builder.finish()))
            }
            // List: child data format depends on whether it's from nested decode or direct
            // From decode_value_to_vec for nested List: [count:i32][elem0][elem1]...[count:i32]...
            DataType::List(child_field) => {
                Self::build_list_from_vec(child_field.clone(), data)
            }
            // Map: [count:i32][key_len:i32][key][marker?][value]... per map, then next map
            DataType::Map(entries_field, keys_sorted) => {
                Self::build_map_from_vec(entries_field.clone(), keys_sorted, data)
            }
            // Struct: data = [field0][field1]...[field0][field1]... (row-major)
            DataType::Struct(fields) => {
                Self::build_struct_from_vec(&fields, data)
            }
            other => Err(ArrowError::NotYetImplemented(format!(
                "build_array_from_vec not implemented for {:?}",
                other
            ))),
        }
    }

    /// Zero-copy version: build Arrow array taking ownership of Vec<u8>
    /// Avoids memory copy by using Buffer::from_vec()
    fn build_array_from_vec_owned(data_type: DataType, data: Vec<u8>) -> Result<ArrayRef, ArrowError> {
        // Handle empty data case - return properly typed empty array
        if data.is_empty() {
            return Self::build_empty_array(&data_type);
        }

        match data_type {
            DataType::Null => Ok(Arc::new(NullArray::new(0))),
            DataType::Int32 => {
                let buffer = Buffer::from_vec(data);
                let values = ScalarBuffer::<i32>::from(buffer);
                Ok(Arc::new(Int32Array::new(values, None)))
            }
            DataType::Int64 => {
                let buffer = Buffer::from_vec(data);
                let values = ScalarBuffer::<i64>::from(buffer);
                Ok(Arc::new(Int64Array::new(values, None)))
            }
            DataType::Float32 => {
                let buffer = Buffer::from_vec(data);
                let values = ScalarBuffer::<f32>::from(buffer);
                Ok(Arc::new(Float32Array::new(values, None)))
            }
            DataType::Float64 => {
                let buffer = Buffer::from_vec(data);
                let values = ScalarBuffer::<f64>::from(buffer);
                Ok(Arc::new(Float64Array::new(values, None)))
            }
            DataType::Boolean => {
                let mut builder = BooleanArray::builder(data.len());
                for byte in &data {
                    builder.append_value(*byte != 0);
                }
                Ok(Arc::new(builder.finish()))
            }
            DataType::Decimal128(precision, scale) => {
                let buffer = Buffer::from_vec(data);
                let values = ScalarBuffer::<i128>::from(buffer);
                Ok(Arc::new(Decimal128Array::new(values, None)
                    .with_precision_and_scale(precision, scale)?))
            }
            DataType::Date32 => {
                let buffer = Buffer::from_vec(data);
                let values = ScalarBuffer::<i32>::from(buffer);
                Ok(Arc::new(Date32Array::new(values, None)))
            }
            DataType::Time32(unit) => {
                let buffer = Buffer::from_vec(data);
                let values = ScalarBuffer::<i32>::from(buffer);
                match unit {
                    arrow_schema::TimeUnit::Second => Ok(Arc::new(Time32SecondArray::new(values, None))),
                    arrow_schema::TimeUnit::Millisecond => Ok(Arc::new(Time32MillisecondArray::new(values, None))),
                    _ => Err(ArrowError::InvalidArgumentError(format!("Invalid Time32 unit: {:?}", unit))),
                }
            }
            DataType::Time64(unit) => {
                let buffer = Buffer::from_vec(data);
                let values = ScalarBuffer::<i64>::from(buffer);
                match unit {
                    arrow_schema::TimeUnit::Microsecond => Ok(Arc::new(Time64MicrosecondArray::new(values, None))),
                    arrow_schema::TimeUnit::Nanosecond => Ok(Arc::new(Time64NanosecondArray::new(values, None))),
                    _ => Err(ArrowError::InvalidArgumentError(format!("Invalid Time64 unit: {:?}", unit))),
                }
            }
            DataType::Timestamp(unit, tz) => {
                let buffer = Buffer::from_vec(data);
                let values = ScalarBuffer::<i64>::from(buffer);
                match unit {
                    arrow_schema::TimeUnit::Second => {
                        let arr = TimestampSecondArray::new(values, None);
                        Ok(Arc::new(if let Some(tz) = tz { arr.with_timezone(tz.as_ref()) } else { arr }))
                    }
                    arrow_schema::TimeUnit::Millisecond => {
                        let arr = TimestampMillisecondArray::new(values, None);
                        Ok(Arc::new(if let Some(tz) = tz { arr.with_timezone(tz.as_ref()) } else { arr }))
                    }
                    arrow_schema::TimeUnit::Microsecond => {
                        let arr = TimestampMicrosecondArray::new(values, None);
                        Ok(Arc::new(if let Some(tz) = tz { arr.with_timezone(tz.as_ref()) } else { arr }))
                    }
                    arrow_schema::TimeUnit::Nanosecond => {
                        let arr = TimestampNanosecondArray::new(values, None);
                        Ok(Arc::new(if let Some(tz) = tz { arr.with_timezone(tz.as_ref()) } else { arr }))
                    }
                }
            }
            // For complex types, fall back to slice version (they need parsing anyway)
            other => Self::build_array_from_vec(other, &data),
        }
    }

    /// Zero-copy version of build_array_from_vec_with_nulls
    fn build_array_from_vec_with_nulls_owned(data_type: DataType, data: Vec<u8>, validity: Vec<bool>) -> Result<ArrayRef, ArrowError> {
        let null_buffer = NullBuffer::from(validity.iter().copied().collect::<Vec<bool>>());

        match data_type {
            DataType::Int32 => {
                let buffer = Buffer::from_vec(data);
                let values = ScalarBuffer::<i32>::from(buffer);
                Ok(Arc::new(Int32Array::new(values, Some(null_buffer))))
            }
            DataType::Int64 => {
                let buffer = Buffer::from_vec(data);
                let values = ScalarBuffer::<i64>::from(buffer);
                Ok(Arc::new(Int64Array::new(values, Some(null_buffer))))
            }
            DataType::Float32 => {
                let buffer = Buffer::from_vec(data);
                let values = ScalarBuffer::<f32>::from(buffer);
                Ok(Arc::new(Float32Array::new(values, Some(null_buffer))))
            }
            DataType::Float64 => {
                let buffer = Buffer::from_vec(data);
                let values = ScalarBuffer::<f64>::from(buffer);
                Ok(Arc::new(Float64Array::new(values, Some(null_buffer))))
            }
            DataType::Boolean => {
                let mut builder = BooleanArray::builder(data.len());
                for (i, byte) in data.iter().enumerate() {
                    if validity.get(i).copied().unwrap_or(false) {
                        builder.append_value(*byte != 0);
                    } else {
                        builder.append_null();
                    }
                }
                Ok(Arc::new(builder.finish()))
            }
            DataType::Decimal128(precision, scale) => {
                let buffer = Buffer::from_vec(data);
                let values = ScalarBuffer::<i128>::from(buffer);
                Ok(Arc::new(Decimal128Array::new(values, Some(null_buffer))
                    .with_precision_and_scale(precision, scale)?))
            }
            DataType::Date32 => {
                let buffer = Buffer::from_vec(data);
                let values = ScalarBuffer::<i32>::from(buffer);
                Ok(Arc::new(Date32Array::new(values, Some(null_buffer))))
            }
            DataType::Time64(unit) => {
                let buffer = Buffer::from_vec(data);
                let values = ScalarBuffer::<i64>::from(buffer);
                match unit {
                    arrow_schema::TimeUnit::Microsecond => Ok(Arc::new(Time64MicrosecondArray::new(values, Some(null_buffer)))),
                    arrow_schema::TimeUnit::Nanosecond => Ok(Arc::new(Time64NanosecondArray::new(values, Some(null_buffer)))),
                    _ => Err(ArrowError::InvalidArgumentError(format!("Invalid Time64 unit: {:?}", unit))),
                }
            }
            DataType::Timestamp(unit, tz) => {
                let buffer = Buffer::from_vec(data);
                let values = ScalarBuffer::<i64>::from(buffer);
                match unit {
                    arrow_schema::TimeUnit::Second => {
                        let arr = TimestampSecondArray::new(values, Some(null_buffer));
                        Ok(Arc::new(if let Some(tz) = tz { arr.with_timezone(tz.as_ref()) } else { arr }))
                    }
                    arrow_schema::TimeUnit::Millisecond => {
                        let arr = TimestampMillisecondArray::new(values, Some(null_buffer));
                        Ok(Arc::new(if let Some(tz) = tz { arr.with_timezone(tz.as_ref()) } else { arr }))
                    }
                    arrow_schema::TimeUnit::Microsecond => {
                        let arr = TimestampMicrosecondArray::new(values, Some(null_buffer));
                        Ok(Arc::new(if let Some(tz) = tz { arr.with_timezone(tz.as_ref()) } else { arr }))
                    }
                    arrow_schema::TimeUnit::Nanosecond => {
                        let arr = TimestampNanosecondArray::new(values, Some(null_buffer));
                        Ok(Arc::new(if let Some(tz) = tz { arr.with_timezone(tz.as_ref()) } else { arr }))
                    }
                }
            }
            // For complex types fall back to slice version
            other => Self::build_array_from_vec_with_nulls(other, &data, &validity.iter().copied().collect::<Vec<_>>()),
        }
    }

    /// Build a ListArray from serialized Vec data
    /// Format: [count0:i32][marker?][elem0_0]...[count1:i32][marker?][elem1_0]...
    /// If child_field is nullable, each element has a 1-byte marker (0=null, 1=valid)
    fn build_list_from_vec(child_field: FieldRef, data: &[u8]) -> Result<ArrayRef, ArrowError> {
        let child_type = child_field.data_type();
        let is_nullable = child_field.is_nullable();

        if data.is_empty() {
            // Empty list array
            let empty_child = Self::build_empty_array(child_type)?;
            return Ok(Arc::new(ListArray::new(
                child_field,
                OffsetBuffer::new(ScalarBuffer::from(vec![0i32])),
                empty_child,
                None,
            )));
        }

        let mut offsets: Vec<i32> = vec![0];
        let mut all_child_data: Vec<u8> = Vec::new();
        let mut null_bitmap: Vec<bool> = Vec::new(); // true = valid, false = null
        let mut pos = 0;

        // Parse each list: [count:i32][marker?][elem0][marker?][elem1]...
        while pos + 4 <= data.len() {
            let count = i32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            pos += 4;

            // Parse 'count' elements
            for _ in 0..count {
                if is_nullable {
                    if pos >= data.len() {
                        return Err(ArrowError::ParseError("Truncated nullable marker in list".to_string()));
                    }
                    let marker = data[pos];
                    pos += 1;
                    if marker == 0 {
                        // Null element - add placeholder data and mark as null
                        null_bitmap.push(false);
                        Self::append_placeholder_to_vec(child_type, &mut all_child_data);
                    } else {
                        // Valid element
                        null_bitmap.push(true);
                        let elem_start = pos;
                        pos = Self::skip_serialized_value(child_type, data, pos)?;
                        all_child_data.extend_from_slice(&data[elem_start..pos]);
                    }
                } else {
                    let elem_start = pos;
                    pos = Self::skip_serialized_value(child_type, data, pos)?;
                    all_child_data.extend_from_slice(&data[elem_start..pos]);
                }
            }

            offsets.push(offsets.last().unwrap() + count);
        }

        // Build child array from collected data, with null buffer if needed
        let child_array = if is_nullable && !null_bitmap.is_empty() {
            Self::build_array_from_vec_with_nulls(child_type.clone(), &all_child_data, &null_bitmap)?
        } else {
            Self::build_array_from_vec(child_type.clone(), &all_child_data)?
        };

        Ok(Arc::new(ListArray::new(
            child_field,
            OffsetBuffer::new(ScalarBuffer::from(offsets)),
            child_array,
            None,
        )))
    }

    /// Build a MapArray from serialized Vec data
    /// Format: [count:i32][key_len:i32][key_bytes][marker?][value]... per map
    /// Keys are always strings; values may be nullable (marker = 0 for null, 1 for valid)
    fn build_map_from_vec(entries_field: FieldRef, keys_sorted: bool, data: &[u8]) -> Result<ArrayRef, ArrowError> {
        // Extract value type and nullability from entries struct
        let (value_type, value_nullable) = if let DataType::Struct(fields) = entries_field.data_type() {
            fields.iter()
                .find(|f| f.name() == "value")
                .map(|f| (f.data_type().clone(), f.is_nullable()))
                .unwrap_or((DataType::Utf8, false))
        } else {
            return Err(ArrowError::InvalidArgumentError(
                "Map entries field must be a Struct".to_string()
            ));
        };

        // Extract struct fields from entries_field
        let struct_fields = match entries_field.data_type() {
            DataType::Struct(fields) => fields.clone(),
            _ => return Err(ArrowError::InvalidArgumentError(
                "Map entries field must be a Struct".to_string()
            )),
        };

        if data.is_empty() {
            // Empty map array
            let empty_keys = Arc::new(StringArray::new_null(0)) as ArrayRef;
            let empty_values = Self::build_empty_array(&value_type)?;
            let entries_struct = StructArray::new(
                struct_fields.clone(),
                vec![empty_keys, empty_values],
                None,
            );
            return Ok(Arc::new(MapArray::new(
                entries_field,
                OffsetBuffer::new(ScalarBuffer::from(vec![0i32])),
                entries_struct,
                None,
                keys_sorted,
            )));
        }

        let mut offsets: Vec<i32> = vec![0];
        let mut all_keys: Vec<String> = Vec::new();
        let mut all_values_data: Vec<u8> = Vec::new();
        let mut value_nulls: Vec<bool> = Vec::new(); // true = valid
        let mut pos = 0;

        // Parse each map: [count:i32][key_len:i32][key][marker?][value]...
        while pos + 4 <= data.len() {
            let count = i32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            pos += 4;

            // Parse 'count' key-value pairs
            for _ in 0..count {
                // Read key (always string): [key_len:i32][key_bytes]
                if pos + 4 > data.len() {
                    return Err(ArrowError::ParseError("Truncated key length in map".to_string()));
                }
                let key_len = i32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
                pos += 4;

                if pos + key_len > data.len() {
                    return Err(ArrowError::ParseError("Truncated key data in map".to_string()));
                }
                let key = std::str::from_utf8(&data[pos..pos + key_len])
                    .map_err(|e| ArrowError::ParseError(format!("Invalid UTF-8 in map key: {}", e)))?;
                all_keys.push(key.to_string());
                pos += key_len;

                // Read value (may have nullable marker)
                if value_nullable {
                    if pos >= data.len() {
                        return Err(ArrowError::ParseError("Truncated value marker in map".to_string()));
                    }
                    let marker = data[pos];
                    pos += 1;
                    if marker == 0 {
                        // Null value
                        value_nulls.push(false);
                        Self::append_placeholder_to_vec(&value_type, &mut all_values_data);
                    } else {
                        // Valid value
                        value_nulls.push(true);
                        let value_start = pos;
                        pos = Self::skip_serialized_value(&value_type, data, pos)?;
                        all_values_data.extend_from_slice(&data[value_start..pos]);
                    }
                } else {
                    let value_start = pos;
                    pos = Self::skip_serialized_value(&value_type, data, pos)?;
                    all_values_data.extend_from_slice(&data[value_start..pos]);
                }
            }

            offsets.push(offsets.last().unwrap() + count);
        }

        // Build key array (always StringArray)
        let key_array: ArrayRef = Arc::new(StringArray::from(all_keys));

        // Build value array with null handling if needed
        let value_array = if value_nullable && !value_nulls.is_empty() {
            Self::build_array_from_vec_with_nulls(value_type, &all_values_data, &value_nulls)?
        } else {
            Self::build_array_from_vec(value_type, &all_values_data)?
        };

        // Create entries struct array
        let entries_struct = StructArray::new(
            struct_fields,
            vec![key_array, value_array],
            None,
        );

        Ok(Arc::new(MapArray::new(
            entries_field,
            OffsetBuffer::new(ScalarBuffer::from(offsets)),
            entries_struct,
            None,
            keys_sorted,
        )))
    }

    /// Append placeholder data for a null value (used when building arrays with null bitmap)
    fn append_placeholder_to_vec(data_type: &DataType, output: &mut Vec<u8>) {
        match data_type {
            DataType::Null => {}
            DataType::Boolean => output.push(0),
            DataType::Int32 | DataType::Date32 | DataType::Time32(_) | DataType::Float32 => {
                output.extend_from_slice(&[0u8; 4]);
            }
            DataType::Int64 | DataType::Time64(_) | DataType::Timestamp(_, _) | DataType::Float64 => {
                output.extend_from_slice(&[0u8; 8]);
            }
            DataType::Utf8 | DataType::Binary => {
                // Store 0 length
                output.extend_from_slice(&0i32.to_le_bytes());
            }
            DataType::Decimal128(_, _) | DataType::Interval(IntervalUnit::MonthDayNano) => {
                output.extend_from_slice(&[0u8; 16]);
            }
            DataType::FixedSizeBinary(size) => {
                output.extend(std::iter::repeat(0u8).take(*size as usize));
            }
            // Complex types: add minimal placeholder
            DataType::List(_) => {
                // Empty list: count = 0
                output.extend_from_slice(&0i32.to_le_bytes());
            }
            DataType::Struct(fields) => {
                // Add placeholders for each field
                for field in fields.iter() {
                    if field.is_nullable() {
                        output.push(0); // null marker
                    } else {
                        Self::append_placeholder_to_vec(field.data_type(), output);
                    }
                }
            }
            DataType::Map(_, _) => {
                // Empty map: count = 0
                output.extend_from_slice(&0i32.to_le_bytes());
            }
            _ => {
                // For unknown types, add nothing (will likely cause issues but better than wrong data)
            }
        }
    }

    /// Build an array from Vec data with explicit null bitmap
    fn build_array_from_vec_with_nulls(data_type: DataType, data: &[u8], nulls: &[bool]) -> Result<ArrayRef, ArrowError> {
        // Build the null buffer
        let null_buffer = NullBuffer::from_iter(nulls.iter().copied());

        match data_type {
            DataType::Int32 => {
                let buffer = Buffer::from_slice_ref(data);
                let values = ScalarBuffer::<i32>::from(buffer);
                Ok(Arc::new(Int32Array::new(values, Some(null_buffer))))
            }
            DataType::Int64 => {
                let buffer = Buffer::from_slice_ref(data);
                let values = ScalarBuffer::<i64>::from(buffer);
                Ok(Arc::new(Int64Array::new(values, Some(null_buffer))))
            }
            DataType::Float32 => {
                let buffer = Buffer::from_slice_ref(data);
                let values = ScalarBuffer::<f32>::from(buffer);
                Ok(Arc::new(Float32Array::new(values, Some(null_buffer))))
            }
            DataType::Float64 => {
                let buffer = Buffer::from_slice_ref(data);
                let values = ScalarBuffer::<f64>::from(buffer);
                Ok(Arc::new(Float64Array::new(values, Some(null_buffer))))
            }
            DataType::Boolean => {
                let mut builder = BooleanArray::builder(data.len());
                for (i, &byte) in data.iter().enumerate() {
                    if i < nulls.len() && !nulls[i] {
                        builder.append_null();
                    } else {
                        builder.append_value(byte != 0);
                    }
                }
                Ok(Arc::new(builder.finish()))
            }
            DataType::Utf8 => {
                // Parse length-prefixed strings
                let mut strings: Vec<Option<&str>> = Vec::new();
                let mut pos = 0;
                let mut idx = 0;
                while pos + 4 <= data.len() && idx < nulls.len() {
                    let len = i32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
                    pos += 4;
                    if !nulls[idx] {
                        strings.push(None);
                    } else if pos + len <= data.len() {
                        let s = std::str::from_utf8(&data[pos..pos+len])
                            .map_err(|e| ArrowError::ParseError(format!("Invalid UTF-8: {}", e)))?;
                        strings.push(Some(s));
                    } else {
                        break;
                    }
                    pos += len;
                    idx += 1;
                }
                Ok(Arc::new(StringArray::from(strings)))
            }
            // For other types, fall back to non-null version (not ideal but safe)
            other => Self::build_array_from_vec(other, data),
        }
    }

    /// Build a StructArray from serialized Vec data
    /// Format: [marker?][field0_row0][marker?][field1_row0]...[marker?][field0_row1]...
    /// Each nullable field is prefixed with a 1-byte marker (0=null, 1=valid)
    fn build_struct_from_vec(fields: &Fields, data: &[u8]) -> Result<ArrayRef, ArrowError> {
        if data.is_empty() || fields.is_empty() {
            let empty_children: Vec<ArrayRef> = fields
                .iter()
                .map(|f| Self::build_empty_array(f.data_type()))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(Arc::new(StructArray::new(fields.clone(), empty_children, None)));
        }

        // Collect data for each field separately, with null tracking
        let mut field_data: Vec<Vec<u8>> = vec![Vec::new(); fields.len()];
        let mut field_nulls: Vec<Vec<bool>> = vec![Vec::new(); fields.len()]; // true = valid
        let mut pos = 0;
        let mut row_count = 0;

        // Parse row-by-row
        while pos < data.len() {
            for (i, field) in fields.iter().enumerate() {
                if field.is_nullable() {
                    if pos >= data.len() {
                        return Err(ArrowError::ParseError("Truncated nullable marker in struct".to_string()));
                    }
                    let marker = data[pos];
                    pos += 1;
                    if marker == 0 {
                        // Null field
                        field_nulls[i].push(false);
                        Self::append_placeholder_to_vec(field.data_type(), &mut field_data[i]);
                    } else {
                        // Valid field
                        field_nulls[i].push(true);
                        let field_start = pos;
                        pos = Self::skip_serialized_value(field.data_type(), data, pos)?;
                        field_data[i].extend_from_slice(&data[field_start..pos]);
                    }
                } else {
                    let field_start = pos;
                    pos = Self::skip_serialized_value(field.data_type(), data, pos)?;
                    field_data[i].extend_from_slice(&data[field_start..pos]);
                }
            }
            row_count += 1;

            // Safety: don't loop forever on malformed data
            if row_count > 1_000_000 {
                return Err(ArrowError::ParseError(
                    "Struct deserialization exceeded maximum row count".to_string(),
                ));
            }
        }

        // Build each field's array with appropriate null handling
        let child_arrays: Vec<ArrayRef> = fields
            .iter()
            .enumerate()
            .map(|(i, field)| {
                if field.is_nullable() && !field_nulls[i].is_empty() {
                    Self::build_array_from_vec_with_nulls(field.data_type().clone(), &field_data[i], &field_nulls[i])
                } else {
                    Self::build_array_from_vec(field.data_type().clone(), &field_data[i])
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Arc::new(StructArray::new(fields.clone(), child_arrays, None)))
    }

    /// Skip over a serialized value, returning the new position
    /// This version is nullable-aware and handles the marker byte for nullable fields
    fn skip_serialized_field(field: &Field, data: &[u8], pos: usize) -> Result<usize, ArrowError> {
        if field.is_nullable() {
            // Nullable field has a marker byte first
            if pos >= data.len() {
                return Err(ArrowError::ParseError("Truncated nullable marker".to_string()));
            }
            let marker = data[pos];
            if marker == 0 {
                // Null value - just the marker, no data
                Ok(pos + 1)
            } else {
                // Valid value - marker + data
                Self::skip_serialized_value(field.data_type(), data, pos + 1)
            }
        } else {
            Self::skip_serialized_value(field.data_type(), data, pos)
        }
    }

    /// Skip over a serialized value, returning the new position
    fn skip_serialized_value(data_type: &DataType, data: &[u8], pos: usize) -> Result<usize, ArrowError> {
        match data_type {
            DataType::Null => Ok(pos),
            DataType::Boolean => Ok(pos + 1),
            DataType::Int32 | DataType::Date32 | DataType::Time32(_) => Ok(pos + 4),
            DataType::Int64 | DataType::Time64(_) | DataType::Timestamp(_, _) => Ok(pos + 8),
            DataType::Float32 => Ok(pos + 4),
            DataType::Float64 => Ok(pos + 8),
            DataType::Decimal128(_, _) => Ok(pos + 16),
            DataType::Interval(IntervalUnit::MonthDayNano) => Ok(pos + 16),
            DataType::FixedSizeBinary(size) => Ok(pos + *size as usize),
            DataType::Utf8 | DataType::Binary => {
                if pos + 4 > data.len() {
                    return Err(ArrowError::ParseError("Truncated string/binary length".to_string()));
                }
                let len = i32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
                Ok(pos + 4 + len)
            }
            DataType::List(child_field) => {
                if pos + 4 > data.len() {
                    return Err(ArrowError::ParseError("Truncated list count".to_string()));
                }
                let count = i32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
                let mut new_pos = pos + 4;
                for _ in 0..count {
                    // Use field-aware skip to handle nullable child elements
                    new_pos = Self::skip_serialized_field(child_field, data, new_pos)?;
                }
                Ok(new_pos)
            }
            DataType::Struct(fields) => {
                let mut new_pos = pos;
                for field in fields.iter() {
                    // Use field-aware skip to handle nullable struct fields
                    new_pos = Self::skip_serialized_field(field, data, new_pos)?;
                }
                Ok(new_pos)
            }
            DataType::Map(entries_field, _) => {
                // Map: [count:i32][key_len:i32][key][marker?][value]...
                if pos + 4 > data.len() {
                    return Err(ArrowError::ParseError("Truncated map count".to_string()));
                }
                let count = i32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
                let mut new_pos = pos + 4;
                let (value_type, value_nullable) = if let DataType::Struct(fields) = entries_field.data_type() {
                    fields.iter().nth(1)
                        .map(|f| (f.data_type().clone(), f.is_nullable()))
                        .unwrap_or((DataType::Utf8, false))
                } else {
                    (DataType::Utf8, false)
                };
                for _ in 0..count {
                    // Skip key (string)
                    if new_pos + 4 > data.len() {
                        return Err(ArrowError::ParseError("Truncated map key length".to_string()));
                    }
                    let key_len = i32::from_le_bytes(data[new_pos..new_pos + 4].try_into().unwrap()) as usize;
                    new_pos += 4 + key_len;
                    // Skip value with nullable awareness
                    if value_nullable {
                        if new_pos >= data.len() {
                            return Err(ArrowError::ParseError("Truncated map value marker".to_string()));
                        }
                        let marker = data[new_pos];
                        new_pos += 1;
                        if marker != 0 {
                            new_pos = Self::skip_serialized_value(&value_type, data, new_pos)?;
                        }
                    } else {
                        new_pos = Self::skip_serialized_value(&value_type, data, new_pos)?;
                    }
                }
                Ok(new_pos)
            }
            DataType::Union(union_fields, _) => {
                // Union: [tag:i32][value]
                if pos + 4 > data.len() {
                    return Err(ArrowError::ParseError("Truncated union tag".to_string()));
                }
                let tag = i32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                let new_pos = pos + 4;
                if let Some((_, field)) = union_fields.iter().find(|(id, _)| *id == tag as i8) {
                    if !matches!(field.data_type(), DataType::Null) {
                        return Self::skip_serialized_value(field.data_type(), data, new_pos);
                    }
                }
                Ok(new_pos)
            }
            DataType::Dictionary(_, _) => Ok(pos + 4), // index is i32
            _ => Err(ArrowError::NotYetImplemented(format!(
                "skip_serialized_value not implemented for {:?}",
                data_type
            ))),
        }
    }

    /// Build an empty array for the given data type
    fn build_empty_array(data_type: &DataType) -> Result<ArrayRef, ArrowError> {
        match data_type {
            DataType::Null => Ok(Arc::new(NullArray::new(0))),
            DataType::Int32 => Ok(Arc::new(Int32Array::from(Vec::<i32>::new()))),
            DataType::Int64 => Ok(Arc::new(Int64Array::from(Vec::<i64>::new()))),
            DataType::Float32 => Ok(Arc::new(Float32Array::from(Vec::<f32>::new()))),
            DataType::Float64 => Ok(Arc::new(Float64Array::from(Vec::<f64>::new()))),
            DataType::Boolean => Ok(Arc::new(BooleanArray::from(Vec::<bool>::new()))),
            DataType::Utf8 => Ok(Arc::new(StringArray::from(Vec::<&str>::new()))),
            DataType::Binary => Ok(Arc::new(BinaryArray::from(Vec::<&[u8]>::new()))),
            DataType::List(child_field) => {
                let empty_child = Self::build_empty_array(child_field.data_type())?;
                Ok(Arc::new(ListArray::new(
                    child_field.clone(),
                    OffsetBuffer::new(ScalarBuffer::from(vec![0i32])),
                    empty_child,
                    None,
                )))
            }
            DataType::Struct(fields) => {
                let empty_children: Vec<ArrayRef> = fields
                    .iter()
                    .map(|f| Self::build_empty_array(f.data_type()))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Arc::new(StructArray::new(fields.clone(), empty_children, None)))
            }
            _ => Ok(Arc::new(NullArray::new(0))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer_provider::{BufferLayout, ColumnRegion, DefaultBufferProvider, PooledBufferProvider};

    fn create_int64_layout(row_capacity: usize) -> BufferLayout {
        BufferLayout {
            columns: vec![ColumnRegion::fixed(0, row_capacity * 8)],
            total_size: row_capacity * 8,
            row_count: row_capacity,
        }
    }

    #[test]
    fn test_pooled_decoder_creation() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Float64, false),
        ]));

        let layout = BufferLayout {
            columns: vec![
                ColumnRegion::fixed(0, 1024),
                ColumnRegion::fixed(1024, 1024),
            ],
            total_size: 2048,
            row_count: 128,
        };
        let buffer = vec![0u8; layout.total_size];
        let provider = PooledBufferProvider::new(buffer, layout);

        let decoder = PooledRecordDecoder::try_new(schema.clone(), provider).unwrap();

        assert_eq!(decoder.schema().fields().len(), 2);
        assert_eq!(decoder.row_count(), 0);
    }

    #[test]
    fn test_decode_simple_int64() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));

        let layout = create_int64_layout(100);
        let buffer = vec![0u8; layout.total_size];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema.clone(), provider).unwrap();

        // Encode value 42 as zigzag varint: 42 -> 84
        let avro_bytes = vec![84u8];

        let consumed = decoder.decode(&avro_bytes, 1).unwrap();
        assert_eq!(consumed, 1);
        assert_eq!(decoder.row_count(), 1);
    }

    #[test]
    fn test_checkpoint_creation() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let layout = BufferLayout {
            columns: vec![
                ColumnRegion::fixed(0, 800),
                ColumnRegion::variable(800, 1000, 1800, 404), // 101 * 4 for offsets
            ],
            total_size: 2204,
            row_count: 100,
        };
        let buffer = vec![0u8; layout.total_size];
        let provider = PooledBufferProvider::new(buffer, layout);
        let decoder = PooledRecordDecoder::try_new(schema.clone(), provider).unwrap();

        let checkpoint = decoder.create_checkpoint(0);

        assert_eq!(checkpoint.cursor_position, 0);
        assert_eq!(checkpoint.data_positions.len(), 2);
        assert_eq!(checkpoint.row_counts.len(), 2);
        assert_eq!(checkpoint.nested_snapshots.len(), 2);
    }

    #[test]
    fn test_decode_with_capacity() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));

        // Only room for 2 rows
        let layout = create_int64_layout(2);
        let buffer = vec![0u8; layout.total_size];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema.clone(), provider).unwrap();

        // Encode 3 values as zigzag varint
        let avro_bytes = vec![84u8, 84u8, 84u8]; // 42, 42, 42

        let (consumed, rows, exceeded) = decoder.decode_with_capacity(&avro_bytes, 3).unwrap();

        // Should decode 2 rows successfully
        assert_eq!(rows, 2);
        assert_eq!(consumed, 2);
        assert!(exceeded);
        assert_eq!(decoder.row_count(), 2);
    }

    #[test]
    fn test_nested_snapshot_map() {
        let col = ColumnState {
            column_index: 0,
            data_type: DataType::Map(
                Arc::new(Field::new(
                    "entries",
                    DataType::Struct(Fields::from(vec![
                        Field::new("key", DataType::Utf8, false),
                        Field::new("value", DataType::Int32, false),
                    ])),
                    false,
                )),
                false,
            ),
            data_position: 0,
            row_count: 0,
            is_nullable: false,
            null_bitmap: Vec::new(),
            nested: NestedState::Map {
                map_offsets: vec![0, 2],
                key_offsets: vec![0, 3, 7],
                key_data: b"foobar!".to_vec(),
                value_offsets: vec![0],  // Not used for Int32 values
                value_data: vec![1, 2, 3, 4, 5, 6, 7, 8],
                value_validity: Vec::new(),
                value_type: DataType::Int32,
                value_nullable: false,
                total_entries: 2,
            },
        };

        let snapshot = col.create_snapshot();

        if let NestedSnapshot::Map {
            map_offsets_len,
            key_offsets_len,
            key_data_len,
            value_offsets_len,
            value_data_len,
            value_validity_len: _,
            total_entries,
        } = snapshot {
            assert_eq!(map_offsets_len, 2);
            assert_eq!(key_offsets_len, 3);
            assert_eq!(key_data_len, 7);
            assert_eq!(value_offsets_len, 1);
            assert_eq!(value_data_len, 8);
            assert_eq!(total_entries, 2);
        } else {
            panic!("Expected Map snapshot");
        }
    }

    #[test]
    fn test_nested_snapshot_restore() {
        let mut col = ColumnState {
            column_index: 0,
            data_type: DataType::List(Arc::new(Field::new("item", DataType::Int32, false))),
            data_position: 10,
            row_count: 2,
            is_nullable: false,
            null_bitmap: Vec::new(),
            nested: NestedState::List {
                offsets: vec![0, 3, 5],
                child_data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                child_validity: Vec::new(),
                child_data_position: 10,
                child_offsets_position: 0,
                child_type: DataType::Int32,
                child_nullable: false,
                total_elements: 5,
                use_preallocated: false,
            },
        };

        // Create snapshot
        let snapshot = col.create_snapshot();

        // Modify state (simulating partial decode)
        if let NestedState::List { offsets, child_data, total_elements, .. } = &mut col.nested {
            offsets.push(8);
            child_data.extend_from_slice(&[11, 12, 13]);
            *total_elements = 8;
        }
        col.data_position = 20;
        col.row_count = 3;

        // Restore
        col.restore_snapshot(&snapshot);
        col.data_position = 10;
        col.row_count = 2;

        // Verify restored state
        if let NestedState::List { offsets, child_data, total_elements, .. } = &col.nested {
            assert_eq!(offsets.len(), 3);
            assert_eq!(child_data.len(), 10);
            assert_eq!(*total_elements, 5);
        } else {
            panic!("Expected List nested state");
        }
    }

    #[test]
    fn test_enable_preallocated_list() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("list_col", DataType::List(Arc::new(Field::new("item", DataType::Int64, false))), false),
        ]));

        let layout = BufferLayout {
            columns: vec![ColumnRegion::fixed(0, 1024)],
            total_size: 1024,
            row_count: 100,
        };
        let buffer = vec![0u8; 1024];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        // Enable pre-allocated mode
        assert!(decoder.enable_preallocated_list(0).is_ok());

        // Verify the flag is set
        if let NestedState::List { use_preallocated, .. } = &decoder.columns[0].nested {
            assert!(*use_preallocated);
        } else {
            panic!("Expected List nested state");
        }
    }

    #[test]
    fn test_enable_preallocated_list_wrong_type() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("int_col", DataType::Int64, false),
        ]));

        let layout = BufferLayout {
            columns: vec![ColumnRegion::fixed(0, 1024)],
            total_size: 1024,
            row_count: 100,
        };
        let buffer = vec![0u8; 1024];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        // Should fail for non-List column
        let result = decoder.enable_preallocated_list(0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not a List type"));
    }

    #[test]
    fn test_enable_preallocated_list_out_of_bounds() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("list_col", DataType::List(Arc::new(Field::new("item", DataType::Int64, false))), false),
        ]));

        let layout = BufferLayout {
            columns: vec![ColumnRegion::fixed(0, 1024)],
            total_size: 1024,
            row_count: 100,
        };
        let buffer = vec![0u8; 1024];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        // Should fail for out of bounds
        let result = decoder.enable_preallocated_list(5);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out of bounds"));
    }

    #[test]
    fn test_list_child_data_position_tracking() {
        // Test that child_data_position is properly tracked in Vec mode
        let mut col = ColumnState {
            column_index: 0,
            data_type: DataType::List(Arc::new(Field::new("item", DataType::Int32, false))),
            data_position: 0,
            row_count: 0,
            is_nullable: false,
            null_bitmap: Vec::new(),
            nested: NestedState::List {
                offsets: vec![0],
                child_data: Vec::new(),
                child_validity: Vec::new(),
                child_data_position: 0,
                child_offsets_position: 0,
                child_type: DataType::Int32,
                child_nullable: false,
                total_elements: 0,
                use_preallocated: false,
            },
        };

        // Simulate adding data
        if let NestedState::List { child_data, child_data_position, total_elements, offsets, .. } = &mut col.nested {
            // Add 3 i32 values (12 bytes)
            child_data.extend_from_slice(&1i32.to_le_bytes());
            child_data.extend_from_slice(&2i32.to_le_bytes());
            child_data.extend_from_slice(&3i32.to_le_bytes());
            *child_data_position = child_data.len();
            *total_elements = 3;
            offsets.push(3);
        }

        // Create snapshot
        let snapshot = col.create_snapshot();

        if let NestedSnapshot::List { child_data_len, child_data_position, total_elements, .. } = snapshot {
            assert_eq!(child_data_len, 12);
            assert_eq!(child_data_position, 12);
            assert_eq!(total_elements, 3);
        } else {
            panic!("Expected List snapshot");
        }
    }

    #[test]
    fn test_list_snapshot_with_position_restore() {
        let mut col = ColumnState {
            column_index: 0,
            data_type: DataType::List(Arc::new(Field::new("item", DataType::Int32, false))),
            data_position: 0,
            row_count: 2,
            is_nullable: false,
            null_bitmap: Vec::new(),
            nested: NestedState::List {
                offsets: vec![0, 2, 5],
                child_data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], // 5 i32s
                child_validity: Vec::new(),
                child_data_position: 20,
                child_offsets_position: 0,
                child_type: DataType::Int32,
                child_nullable: false,
                total_elements: 5,
                use_preallocated: false,
            },
        };

        // Create snapshot at current state
        let snapshot = col.create_snapshot();

        // Simulate partial row decode (add more data)
        if let NestedState::List { child_data, child_data_position, total_elements, offsets, .. } = &mut col.nested {
            child_data.extend_from_slice(&21i32.to_le_bytes());
            child_data.extend_from_slice(&22i32.to_le_bytes());
            *child_data_position = child_data.len(); // 28 bytes now
            *total_elements = 7;
            offsets.push(7);
        }
        col.row_count = 3;

        // Verify modified state
        if let NestedState::List { child_data_position, total_elements, offsets, .. } = &col.nested {
            assert_eq!(*child_data_position, 28);
            assert_eq!(*total_elements, 7);
            assert_eq!(offsets.len(), 4);
        }

        // Restore snapshot
        col.restore_snapshot(&snapshot);

        // Verify restored state
        if let NestedState::List { child_data, child_data_position, total_elements, offsets, .. } = &col.nested {
            assert_eq!(child_data.len(), 20); // Truncated back
            assert_eq!(*child_data_position, 20); // Position restored
            assert_eq!(*total_elements, 5);
            assert_eq!(offsets.len(), 3);
        } else {
            panic!("Expected List nested state");
        }
    }

    #[test]
    fn test_peek_long_cursor() {
        use crate::reader::cursor::AvroCursor;

        // Test peek_long doesn't consume data
        // 42 in zigzag encoding is 84 (0x54)
        let data = vec![0x54u8];
        let cursor = AvroCursor::new(&data);

        // Peek should return 42
        let peeked = cursor.peek_long().unwrap();
        assert_eq!(peeked, 42);

        // Position should still be 0
        assert_eq!(cursor.position(), 0);

        // Create mutable cursor to actually read
        let mut cursor2 = AvroCursor::new(&data);
        let read = cursor2.get_long().unwrap();
        assert_eq!(read, 42);
        assert_eq!(cursor2.position(), 1);
    }

    #[test]
    fn test_decode_value_to_slice_int32() {
        use crate::reader::cursor::AvroCursor;

        // 42 in zigzag encoding
        let data = vec![0x54u8];
        let mut cursor = AvroCursor::new(&data);
        let mut output = [0u8; 4];

        let bytes_written = PooledRecordDecoder::<DefaultBufferProvider>::decode_value_to_slice(
            &DataType::Int32,
            &mut cursor,
            &mut output,
        ).unwrap();

        assert_eq!(bytes_written, 4);
        assert_eq!(i32::from_le_bytes(output), 42);
    }

    #[test]
    fn test_decode_value_to_slice_int64() {
        use crate::reader::cursor::AvroCursor;

        // 100 in zigzag encoding is 200 (0xC8 0x01)
        let data = vec![0xC8u8, 0x01];
        let mut cursor = AvroCursor::new(&data);
        let mut output = [0u8; 8];

        let bytes_written = PooledRecordDecoder::<DefaultBufferProvider>::decode_value_to_slice(
            &DataType::Int64,
            &mut cursor,
            &mut output,
        ).unwrap();

        assert_eq!(bytes_written, 8);
        assert_eq!(i64::from_le_bytes(output), 100);
    }

    #[test]
    fn test_decode_value_to_slice_float32() {
        use crate::reader::cursor::AvroCursor;

        let value: f32 = 3.14;
        let data = value.to_le_bytes().to_vec();
        let mut cursor = AvroCursor::new(&data);
        let mut output = [0u8; 4];

        let bytes_written = PooledRecordDecoder::<DefaultBufferProvider>::decode_value_to_slice(
            &DataType::Float32,
            &mut cursor,
            &mut output,
        ).unwrap();

        assert_eq!(bytes_written, 4);
        assert!((f32::from_le_bytes(output) - 3.14).abs() < 0.001);
    }

    #[test]
    fn test_decode_value_to_slice_float64() {
        use crate::reader::cursor::AvroCursor;

        let value: f64 = 2.71828;
        let data = value.to_le_bytes().to_vec();
        let mut cursor = AvroCursor::new(&data);
        let mut output = [0u8; 8];

        let bytes_written = PooledRecordDecoder::<DefaultBufferProvider>::decode_value_to_slice(
            &DataType::Float64,
            &mut cursor,
            &mut output,
        ).unwrap();

        assert_eq!(bytes_written, 8);
        assert!((f64::from_le_bytes(output) - 2.71828).abs() < 0.00001);
    }

    #[test]
    fn test_decode_value_to_slice_boolean() {
        use crate::reader::cursor::AvroCursor;

        // True
        let data_true = vec![1u8];
        let mut cursor = AvroCursor::new(&data_true);
        let mut output = [0u8; 1];

        let bytes_written = PooledRecordDecoder::<DefaultBufferProvider>::decode_value_to_slice(
            &DataType::Boolean,
            &mut cursor,
            &mut output,
        ).unwrap();

        assert_eq!(bytes_written, 1);
        assert_eq!(output[0], 1);

        // False
        let data_false = vec![0u8];
        let mut cursor = AvroCursor::new(&data_false);

        let bytes_written = PooledRecordDecoder::<DefaultBufferProvider>::decode_value_to_slice(
            &DataType::Boolean,
            &mut cursor,
            &mut output,
        ).unwrap();

        assert_eq!(bytes_written, 1);
        assert_eq!(output[0], 0);
    }

    #[test]
    fn test_get_value_size_fixed_types() {
        use crate::reader::cursor::AvroCursor;

        let dummy_data = vec![0u8; 100];

        // Null - 0 bytes
        let mut cursor = AvroCursor::new(&dummy_data);
        assert_eq!(PooledRecordDecoder::<DefaultBufferProvider>::get_value_size(&DataType::Null, &mut cursor).unwrap(), 0);

        // Boolean - 1 byte
        let mut cursor = AvroCursor::new(&dummy_data);
        assert_eq!(PooledRecordDecoder::<DefaultBufferProvider>::get_value_size(&DataType::Boolean, &mut cursor).unwrap(), 1);

        // Int32 - 4 bytes
        let mut cursor = AvroCursor::new(&dummy_data);
        assert_eq!(PooledRecordDecoder::<DefaultBufferProvider>::get_value_size(&DataType::Int32, &mut cursor).unwrap(), 4);

        // Int64 - 8 bytes
        let mut cursor = AvroCursor::new(&dummy_data);
        assert_eq!(PooledRecordDecoder::<DefaultBufferProvider>::get_value_size(&DataType::Int64, &mut cursor).unwrap(), 8);

        // Float32 - 4 bytes
        let mut cursor = AvroCursor::new(&dummy_data);
        assert_eq!(PooledRecordDecoder::<DefaultBufferProvider>::get_value_size(&DataType::Float32, &mut cursor).unwrap(), 4);

        // Float64 - 8 bytes
        let mut cursor = AvroCursor::new(&dummy_data);
        assert_eq!(PooledRecordDecoder::<DefaultBufferProvider>::get_value_size(&DataType::Float64, &mut cursor).unwrap(), 8);

        // FixedSizeBinary(16) - 16 bytes
        let mut cursor = AvroCursor::new(&dummy_data);
        assert_eq!(PooledRecordDecoder::<DefaultBufferProvider>::get_value_size(&DataType::FixedSizeBinary(16), &mut cursor).unwrap(), 16);
    }

    #[test]
    fn test_get_value_size_variable_types() {
        use crate::reader::cursor::AvroCursor;

        // String "hello" - length 5 encoded as varint (0x0A = 5 in zigzag)
        // Total size = 4 (length prefix in output) + 5 (data) = 9
        let data = vec![0x0Au8, b'h', b'e', b'l', b'l', b'o'];
        let mut cursor = AvroCursor::new(&data);

        let size = PooledRecordDecoder::<DefaultBufferProvider>::get_value_size(&DataType::Utf8, &mut cursor).unwrap();
        assert_eq!(size, 4 + 5); // 4 byte length prefix + 5 bytes data

        // Binary with length 10 - 0x14 = 10 in zigzag
        let data = vec![0x14u8; 11]; // Length byte + 10 data bytes
        let mut cursor = AvroCursor::new(&data);

        let size = PooledRecordDecoder::<DefaultBufferProvider>::get_value_size(&DataType::Binary, &mut cursor).unwrap();
        assert_eq!(size, 4 + 10); // 4 byte length prefix + 10 bytes data
    }

    #[test]
    fn test_build_array_from_buffer_int32() {
        // Create buffer with 3 i32 values: 1, 2, 3
        let mut data = Vec::new();
        data.extend_from_slice(&1i32.to_le_bytes());
        data.extend_from_slice(&2i32.to_le_bytes());
        data.extend_from_slice(&3i32.to_le_bytes());
        let buffer = Buffer::from(data);

        let array = PooledRecordDecoder::<DefaultBufferProvider>::build_array_from_buffer(
            DataType::Int32,
            buffer,
        ).unwrap();

        let int_array = array.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(int_array.len(), 3);
        assert_eq!(int_array.value(0), 1);
        assert_eq!(int_array.value(1), 2);
        assert_eq!(int_array.value(2), 3);
    }

    #[test]
    fn test_build_array_from_buffer_int64() {
        let mut data = Vec::new();
        data.extend_from_slice(&100i64.to_le_bytes());
        data.extend_from_slice(&200i64.to_le_bytes());
        let buffer = Buffer::from(data);

        let array = PooledRecordDecoder::<DefaultBufferProvider>::build_array_from_buffer(
            DataType::Int64,
            buffer,
        ).unwrap();

        let int_array = array.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(int_array.len(), 2);
        assert_eq!(int_array.value(0), 100);
        assert_eq!(int_array.value(1), 200);
    }

    #[test]
    fn test_build_array_from_buffer_float32() {
        let mut data = Vec::new();
        data.extend_from_slice(&1.5f32.to_le_bytes());
        data.extend_from_slice(&2.5f32.to_le_bytes());
        let buffer = Buffer::from(data);

        let array = PooledRecordDecoder::<DefaultBufferProvider>::build_array_from_buffer(
            DataType::Float32,
            buffer,
        ).unwrap();

        let float_array = array.as_any().downcast_ref::<Float32Array>().unwrap();
        assert_eq!(float_array.len(), 2);
        assert!((float_array.value(0) - 1.5).abs() < 0.001);
        assert!((float_array.value(1) - 2.5).abs() < 0.001);
    }

    #[test]
    fn test_build_array_from_buffer_float64() {
        let mut data = Vec::new();
        data.extend_from_slice(&3.14159f64.to_le_bytes());
        let buffer = Buffer::from(data);

        let array = PooledRecordDecoder::<DefaultBufferProvider>::build_array_from_buffer(
            DataType::Float64,
            buffer,
        ).unwrap();

        let float_array = array.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(float_array.len(), 1);
        assert!((float_array.value(0) - 3.14159).abs() < 0.00001);
    }

    #[test]
    fn test_build_array_from_buffer_boolean() {
        // Booleans stored as bytes (0 or 1)
        let data = vec![1u8, 0u8, 1u8, 1u8, 0u8];
        let buffer = Buffer::from(data);

        let array = PooledRecordDecoder::<DefaultBufferProvider>::build_array_from_buffer(
            DataType::Boolean,
            buffer,
        ).unwrap();

        let bool_array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(bool_array.len(), 5);
        assert!(bool_array.value(0));
        assert!(!bool_array.value(1));
        assert!(bool_array.value(2));
        assert!(bool_array.value(3));
        assert!(!bool_array.value(4));
    }

    #[test]
    fn test_build_array_from_buffer_empty() {
        let buffer = Buffer::from(Vec::<u8>::new());

        let array = PooledRecordDecoder::<DefaultBufferProvider>::build_array_from_buffer(
            DataType::Int32,
            buffer,
        ).unwrap();

        let int_array = array.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(int_array.len(), 0);
    }

    #[test]
    fn test_build_array_from_buffer_unsupported_type() {
        let buffer = Buffer::from(vec![0u8; 10]);

        let result = PooledRecordDecoder::<DefaultBufferProvider>::build_array_from_buffer(
            DataType::Utf8, // String not supported in buffer mode
            buffer,
        );

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not implemented"));
    }

    #[test]
    fn test_preallocated_list_with_child_region() {
        // Create a schema with a List<Int32> column
        let schema = Arc::new(Schema::new(vec![
            Field::new("list_col", DataType::List(Arc::new(Field::new("item", DataType::Int32, false))), false),
        ]));

        // Create layout with child region for list elements
        let child_region = ColumnRegion::fixed(1024, 512); // 512 bytes for child data
        let parent_region = ColumnRegion::fixed(0, 1024).with_child(child_region);

        let layout = BufferLayout {
            columns: vec![parent_region],
            total_size: 1536,
            row_count: 100,
        };

        let buffer = vec![0u8; 1536];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        // Enable pre-allocated mode
        decoder.enable_preallocated_list(0).unwrap();

        // Verify child region is accessible
        assert!(decoder.provider.has_capacity_for_child_data(0, 0, 100));
        assert!(decoder.provider.has_capacity_for_child_data(0, 0, 512));
        // Should still have capacity at exactly 512
        assert!(!decoder.provider.has_capacity_for_child_data(0, 0, 513));
    }

    #[test]
    fn test_child_region_capacity_check() {
        let child_region = ColumnRegion::fixed(100, 50); // 50 bytes capacity
        let parent_region = ColumnRegion::fixed(0, 100).with_child(child_region);

        let layout = BufferLayout {
            columns: vec![parent_region],
            total_size: 150,
            row_count: 10,
        };

        let buffer = vec![0u8; 150];
        let provider = PooledBufferProvider::new(buffer, layout);

        // Should have capacity for up to 50 bytes
        assert!(provider.has_capacity_for_child_data(0, 0, 50));
        assert!(!provider.has_capacity_for_child_data(0, 0, 51));

        // Invalid column index
        assert!(!provider.has_capacity_for_child_data(5, 0, 10));

        // Invalid child index
        assert!(!provider.has_capacity_for_child_data(0, 1, 10));
    }

    // =============================================================================
    // Comprehensive roundtrip tests for ALL supported Avro types
    // =============================================================================

    /// Helper to create Avro varint encoding for i32
    fn encode_varint_i32(val: i32) -> Vec<u8> {
        let zigzag = ((val << 1) ^ (val >> 31)) as u32;
        let mut result = Vec::new();
        let mut n = zigzag;
        loop {
            if n & !0x7F == 0 {
                result.push(n as u8);
                break;
            }
            result.push((n as u8 & 0x7F) | 0x80);
            n >>= 7;
        }
        result
    }

    /// Helper to create Avro varint encoding for i64
    fn encode_varint_i64(val: i64) -> Vec<u8> {
        let zigzag = ((val << 1) ^ (val >> 63)) as u64;
        let mut result = Vec::new();
        let mut n = zigzag;
        loop {
            if n & !0x7F == 0 {
                result.push(n as u8);
                break;
            }
            result.push((n as u8 & 0x7F) | 0x80);
            n >>= 7;
        }
        result
    }

    /// Helper to encode Avro string/bytes (length-prefixed)
    fn encode_bytes(data: &[u8]) -> Vec<u8> {
        let mut result = encode_varint_i64(data.len() as i64);
        result.extend_from_slice(data);
        result
    }

    #[test]
    fn test_roundtrip_flat_schema() {
        // Test flat schema: Int64, Float64, Int32, Float32
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Float64, false),
            Field::new("count", DataType::Int32, false),
            Field::new("ratio", DataType::Float32, false),
        ]));

        let rows = 100;
        let layout = BufferLayout {
            columns: vec![
                ColumnRegion::fixed(0, rows * 8),        // id: Int64
                ColumnRegion::fixed(rows * 8, rows * 8), // value: Float64
                ColumnRegion::fixed(rows * 16, rows * 4),// count: Int32
                ColumnRegion::fixed(rows * 20, rows * 4),// ratio: Float32
            ],
            total_size: rows * 24,
            row_count: rows,
        };

        let buffer = vec![0u8; layout.total_size];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        // Encode test data
        let mut avro_data = Vec::new();
        for i in 0..rows {
            avro_data.extend(encode_varint_i64(i as i64));      // id
            avro_data.extend((i as f64 * 0.1).to_le_bytes());    // value (fixed 8 bytes)
            avro_data.extend(encode_varint_i32((i % 1000) as i32)); // count
            avro_data.extend((i as f32 / 100.0).to_le_bytes());  // ratio (fixed 4 bytes)
        }

        // Decode
        let bytes_consumed = decoder.decode(&avro_data, rows).unwrap();
        assert!(bytes_consumed > 0);

        // Flush and verify
        let batch = decoder.flush().unwrap();
        assert_eq!(batch.num_rows(), rows);
        assert_eq!(batch.num_columns(), 4);

        // Verify values
        let id_col = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(id_col.value(0), 0);
        assert_eq!(id_col.value(99), 99);

        let count_col = batch.column(2).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(count_col.value(0), 0);
        assert_eq!(count_col.value(99), 99);
    }

    #[test]
    fn test_roundtrip_strings_schema() {
        // Test strings-only schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("email", DataType::Utf8, false),
        ]));

        let rows = 50;
        let avg_len = 20;
        let string_capacity = rows * (avg_len + 10);
        let offsets_capacity = (rows + 1) * 4;

        let mut offset = 0;
        let layout = BufferLayout {
            columns: vec![
                ColumnRegion {
                    data_offset: offset,
                    data_capacity: string_capacity,
                    offsets_offset: Some(offset + string_capacity),
                    offsets_capacity: Some(offsets_capacity),
                    null_bitmap_offset: None,
                    null_bitmap_capacity: None,
                    child_regions: vec![],
                },
                {
                    offset += string_capacity + offsets_capacity;
                    ColumnRegion {
                        data_offset: offset,
                        data_capacity: string_capacity,
                        offsets_offset: Some(offset + string_capacity),
                        offsets_capacity: Some(offsets_capacity),
                        null_bitmap_offset: None,
                        null_bitmap_capacity: None,
                        child_regions: vec![],
                    }
                },
            ],
            total_size: (string_capacity + offsets_capacity) * 2,
            row_count: rows,
        };

        let buffer = vec![0u8; layout.total_size];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        // Encode test data
        let mut avro_data = Vec::new();
        for i in 0..rows {
            let name = format!("name_{:05}", i);
            let email = format!("user{}@test.com", i);
            avro_data.extend(encode_bytes(name.as_bytes()));
            avro_data.extend(encode_bytes(email.as_bytes()));
        }

        // Decode
        let bytes_consumed = decoder.decode(&avro_data, rows).unwrap();
        assert!(bytes_consumed > 0);

        // Flush and verify
        let batch = decoder.flush().unwrap();
        assert_eq!(batch.num_rows(), rows);
        assert_eq!(batch.num_columns(), 2);

        let name_col = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(name_col.value(0), "name_00000");
        assert_eq!(name_col.value(49), "name_00049");
    }

    #[test]
    fn test_roundtrip_mixed_schema() {
        // Test mixed schema: fixed + variable types
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("value", DataType::Float64, false),
        ]));

        let rows = 30;
        let string_capacity = rows * 30;
        let offsets_capacity = (rows + 1) * 4;

        let layout = BufferLayout {
            columns: vec![
                ColumnRegion::fixed(0, rows * 8), // id
                ColumnRegion {
                    data_offset: rows * 8,
                    data_capacity: string_capacity,
                    offsets_offset: Some(rows * 8 + string_capacity),
                    offsets_capacity: Some(offsets_capacity),
                    null_bitmap_offset: None,
                    null_bitmap_capacity: None,
                    child_regions: vec![],
                },
                ColumnRegion::fixed(rows * 8 + string_capacity + offsets_capacity, rows * 8), // value
            ],
            total_size: rows * 16 + string_capacity + offsets_capacity,
            row_count: rows,
        };

        let buffer = vec![0u8; layout.total_size];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        // Encode test data
        let mut avro_data = Vec::new();
        for i in 0..rows {
            avro_data.extend(encode_varint_i64(i as i64 * 100));
            let name = format!("item_{}", i);
            avro_data.extend(encode_bytes(name.as_bytes()));
            avro_data.extend((i as f64 * 1.5).to_le_bytes());
        }

        // Decode
        let bytes_consumed = decoder.decode(&avro_data, rows).unwrap();
        assert!(bytes_consumed > 0);

        // Flush and verify
        let batch = decoder.flush().unwrap();
        assert_eq!(batch.num_rows(), rows);

        let id_col = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(id_col.value(0), 0);
        assert_eq!(id_col.value(10), 1000);
    }

    #[test]
    fn test_roundtrip_dictionary_schema() {
        use arrow_array::DictionaryArray;
        use arrow_array::types::Int32Type;

        // Test dictionary (enum) schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("status", DataType::Dictionary(
                Box::new(DataType::Int32),
                Box::new(DataType::Utf8)
            ), false),
        ]));

        let rows = 100;
        let layout = BufferLayout {
            columns: vec![
                ColumnRegion::fixed(0, 0), // Dictionary uses Vec internally
            ],
            total_size: 0,
            row_count: rows,
        };

        let buffer = vec![0u8; 1]; // Minimal buffer
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        // Set dictionary values (enum symbols)
        decoder.set_dictionary_values(0, vec![
            "ACTIVE".to_string(),
            "INACTIVE".to_string(),
            "PENDING".to_string(),
        ]).unwrap();

        // Encode test data - just varint indices
        let mut avro_data = Vec::new();
        for i in 0..rows {
            avro_data.extend(encode_varint_i32((i % 3) as i32)); // cycle through 0, 1, 2
        }

        // Decode
        let bytes_consumed = decoder.decode(&avro_data, rows).unwrap();
        assert!(bytes_consumed > 0);

        // Flush and verify
        let batch = decoder.flush().unwrap();
        assert_eq!(batch.num_rows(), rows);

        let dict_col = batch.column(0).as_any()
            .downcast_ref::<DictionaryArray<Int32Type>>().unwrap();

        // Check that keys cycle through 0, 1, 2
        let keys = dict_col.keys();
        assert_eq!(keys.value(0), 0);
        assert_eq!(keys.value(1), 1);
        assert_eq!(keys.value(2), 2);
        assert_eq!(keys.value(3), 0);
    }

    #[test]
    fn test_roundtrip_all_types_combined() {

        // Test comprehensive schema with all types
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("value", DataType::Float64, false),
            Field::new("status", DataType::Dictionary(
                Box::new(DataType::Int32),
                Box::new(DataType::Utf8)
            ), false),
            Field::new("count", DataType::Int32, false),
        ]));

        let rows = 50;
        let string_capacity = rows * 30;
        let offsets_capacity = (rows + 1) * 4;

        let layout = BufferLayout {
            columns: vec![
                ColumnRegion::fixed(0, rows * 8),  // id
                ColumnRegion {                      // name
                    data_offset: rows * 8,
                    data_capacity: string_capacity,
                    offsets_offset: Some(rows * 8 + string_capacity),
                    offsets_capacity: Some(offsets_capacity),
                    null_bitmap_offset: None,
                    null_bitmap_capacity: None,
                    child_regions: vec![],
                },
                ColumnRegion::fixed(rows * 8 + string_capacity + offsets_capacity, rows * 8), // value
                ColumnRegion::fixed(0, 0),  // status (Dictionary uses Vec)
                ColumnRegion::fixed(rows * 16 + string_capacity + offsets_capacity, rows * 4), // count
            ],
            total_size: rows * 20 + string_capacity + offsets_capacity,
            row_count: rows,
        };

        let buffer = vec![0u8; layout.total_size.max(1)];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        // Set dictionary values
        decoder.set_dictionary_values(3, vec![
            "ACTIVE".to_string(),
            "INACTIVE".to_string(),
            "PENDING".to_string(),
        ]).unwrap();

        // Encode test data
        let mut avro_data = Vec::new();
        for i in 0..rows {
            avro_data.extend(encode_varint_i64(i as i64));                // id
            let name = format!("item_{}", i);
            avro_data.extend(encode_bytes(name.as_bytes()));              // name
            avro_data.extend((i as f64 * 0.5).to_le_bytes());             // value (fixed 8 bytes)
            avro_data.extend(encode_varint_i32((i % 3) as i32));          // status (enum index)
            avro_data.extend(encode_varint_i32((i * 10) as i32));         // count
        }

        // Decode
        let bytes_consumed = decoder.decode(&avro_data, rows).unwrap();
        assert!(bytes_consumed > 0);

        // Flush and verify
        let batch = decoder.flush().unwrap();
        assert_eq!(batch.num_rows(), rows);
        assert_eq!(batch.num_columns(), 5);

        // Verify id column
        let id_col = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(id_col.value(0), 0);
        assert_eq!(id_col.value(25), 25);

        // Verify name column
        let name_col = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(name_col.value(0), "item_0");

        // Verify count column
        let count_col = batch.column(4).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(count_col.value(0), 0);
        assert_eq!(count_col.value(5), 50);
    }

    #[test]
    fn test_batch_decode_path_selection() {
        // Verify that different schemas use the correct batch decode path

        // Flat schema should use decode_batch_flat
        let flat_schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let layout = BufferLayout {
            columns: vec![
                ColumnRegion::fixed(0, 80),
                ColumnRegion::fixed(80, 40),
            ],
            total_size: 120,
            row_count: 10,
        };
        let buffer = vec![0u8; 120];
        let provider = PooledBufferProvider::new(buffer, layout);
        let decoder = PooledRecordDecoder::try_new(flat_schema, provider).unwrap();
        assert!(decoder.is_flat_schema());

        // Strings schema should use decode_batch_strings_only
        let strings_schema = Arc::new(Schema::new(vec![
            Field::new("s1", DataType::Utf8, false),
            Field::new("s2", DataType::Utf8, false),
        ]));
        let layout2 = BufferLayout {
            columns: vec![
                ColumnRegion {
                    data_offset: 0,
                    data_capacity: 100,
                    offsets_offset: Some(100),
                    offsets_capacity: Some(44),
                    null_bitmap_offset: None,
                    null_bitmap_capacity: None,
                    child_regions: vec![],
                },
                ColumnRegion {
                    data_offset: 144,
                    data_capacity: 100,
                    offsets_offset: Some(244),
                    offsets_capacity: Some(44),
                    null_bitmap_offset: None,
                    null_bitmap_capacity: None,
                    child_regions: vec![],
                },
            ],
            total_size: 288,
            row_count: 10,
        };
        let buffer2 = vec![0u8; 288];
        let provider2 = PooledBufferProvider::new(buffer2, layout2);
        let decoder2 = PooledRecordDecoder::try_new(strings_schema, provider2).unwrap();
        assert!(decoder2.is_strings_only_schema());
    }

    #[test]
    fn test_roundtrip_list_schema() {
        use arrow_array::ListArray;

        // Test List<Int32> schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("scores", DataType::List(Arc::new(Field::new("item", DataType::Int32, false))), false),
        ]));

        let rows = 10;
        let layout = BufferLayout {
            columns: vec![
                ColumnRegion::fixed(0, rows * 8), // id
                ColumnRegion::fixed(0, 0),        // scores (List uses Vec internally)
            ],
            total_size: rows * 8,
            row_count: rows,
        };

        let buffer = vec![0u8; layout.total_size.max(1)];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        // Encode test data - Avro array format: block_count, elements..., 0 (end)
        let mut avro_data = Vec::new();
        for i in 0..rows {
            // id
            avro_data.extend(encode_varint_i64(i as i64));
            // scores array: [i*10, i*10+1, i*10+2] (3 elements)
            avro_data.extend(encode_varint_i64(3)); // block count
            avro_data.extend(encode_varint_i32((i * 10) as i32));
            avro_data.extend(encode_varint_i32((i * 10 + 1) as i32));
            avro_data.extend(encode_varint_i32((i * 10 + 2) as i32));
            avro_data.extend(encode_varint_i64(0)); // end of array
        }

        // Decode
        let bytes_consumed = decoder.decode(&avro_data, rows).unwrap();
        assert!(bytes_consumed > 0);

        // Flush and verify
        let batch = decoder.flush().unwrap();
        assert_eq!(batch.num_rows(), rows);
        assert_eq!(batch.num_columns(), 2);

        // Verify id column
        let id_col = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(id_col.value(0), 0);
        assert_eq!(id_col.value(5), 5);

        // Verify list column
        let list_col = batch.column(1).as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(list_col.len(), rows);

        // Check first row's list: [0, 1, 2]
        let first_list = list_col.value(0);
        let first_values = first_list.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(first_values.len(), 3);
        assert_eq!(first_values.value(0), 0);
        assert_eq!(first_values.value(1), 1);
        assert_eq!(first_values.value(2), 2);

        // Check row 5's list: [50, 51, 52]
        let fifth_list = list_col.value(5);
        let fifth_values = fifth_list.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(fifth_values.value(0), 50);
    }

    #[test]
    fn test_roundtrip_map_schema() {
        use arrow_array::MapArray;

        // Test Map<String, Int32> schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("metadata", DataType::Map(
                Arc::new(Field::new("entries", DataType::Struct(
                    vec![
                        Field::new("key", DataType::Utf8, false),
                        Field::new("value", DataType::Int32, true),
                    ].into()
                ), false)),
                false
            ), false),
        ]));

        let rows = 5;
        let layout = BufferLayout {
            columns: vec![
                ColumnRegion::fixed(0, rows * 8), // id
                ColumnRegion::fixed(0, 0),        // metadata (Map uses Vec internally)
            ],
            total_size: rows * 8,
            row_count: rows,
        };

        let buffer = vec![0u8; layout.total_size.max(1)];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        // Encode test data - Avro map format: block_count, (key, value)..., 0 (end)
        // Value is nullable, so we need union index (1=non-null, zigzag: 2) before each value
        let mut avro_data = Vec::new();
        for i in 0..rows {
            // id
            avro_data.extend(encode_varint_i64(i as i64));
            // map with 2 entries: {"a": i*10, "b": i*10+1}
            avro_data.extend(encode_varint_i64(2)); // block count
            avro_data.extend(encode_bytes(b"a"));
            avro_data.extend(encode_varint_i64(1)); // union index: non-null (zigzag: 2)
            avro_data.extend(encode_varint_i32((i * 10) as i32));
            avro_data.extend(encode_bytes(b"b"));
            avro_data.extend(encode_varint_i64(1)); // union index: non-null (zigzag: 2)
            avro_data.extend(encode_varint_i32((i * 10 + 1) as i32));
            avro_data.extend(encode_varint_i64(0)); // end of map
        }

        // Decode
        let bytes_consumed = decoder.decode(&avro_data, rows).unwrap();
        assert!(bytes_consumed > 0);

        // Flush and verify
        let batch = decoder.flush().unwrap();
        assert_eq!(batch.num_rows(), rows);
        assert_eq!(batch.num_columns(), 2);

        // Verify id column
        let id_col = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(id_col.value(0), 0);

        // Verify map column structure
        let map_col = batch.column(1).as_any().downcast_ref::<MapArray>().unwrap();
        assert_eq!(map_col.len(), rows);

        // Each row should have 2 entries
        assert_eq!(map_col.value_length(0), 2);
        assert_eq!(map_col.value_length(1), 2);
    }

    #[test]
    fn test_roundtrip_map_preallocated() {
        use arrow_array::MapArray;

        // Test Map<String, Int32> schema with pre-allocated child regions
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("metadata", DataType::Map(
                Arc::new(Field::new("entries", DataType::Struct(
                    vec![
                        Field::new("key", DataType::Utf8, false),
                        Field::new("value", DataType::Int32, true),
                    ].into()
                ), false)),
                false
            ), false),
        ]));

        let rows = 5;
        let avg_entries = 2;
        let total_entries = rows * avg_entries;
        let avg_key_len = 5;

        // Calculate buffer sizes
        let id_cap = rows * 8;
        let map_off_cap = (rows + 1) * 4;
        let key_data_cap = total_entries * avg_key_len;
        let key_off_cap = (total_entries + 1) * 4; // key string offsets
        let value_data_cap = total_entries * 4; // Int32 values (fixed size, no offsets needed)

        let layout = BufferLayout {
            columns: vec![
                ColumnRegion::fixed(0, id_cap), // id
                ColumnRegion {
                    data_offset: id_cap,
                    data_capacity: 0,
                    offsets_offset: Some(id_cap),
                    offsets_capacity: Some(map_off_cap),
                    null_bitmap_offset: None,
                    null_bitmap_capacity: None,
                    child_regions: vec![
                        // child_index 0: key data (strings) + offsets
                        ColumnRegion {
                            data_offset: id_cap + map_off_cap,
                            data_capacity: key_data_cap,
                            offsets_offset: Some(id_cap + map_off_cap + key_data_cap),
                            offsets_capacity: Some(key_off_cap),
                            null_bitmap_offset: None,
                            null_bitmap_capacity: None,
                            child_regions: vec![],
                        },
                        // child_index 1: value data (Int32 - no offsets needed)
                        ColumnRegion {
                            data_offset: id_cap + map_off_cap + key_data_cap + key_off_cap,
                            data_capacity: value_data_cap,
                            offsets_offset: None,
                            offsets_capacity: None,
                            null_bitmap_offset: None,
                            null_bitmap_capacity: None,
                            child_regions: vec![],
                        },
                    ],
                },
            ],
            total_size: id_cap + map_off_cap + key_data_cap + key_off_cap + value_data_cap,
            row_count: rows,
        };

        let buffer = vec![0u8; layout.total_size.max(1)];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        // Encode test data - Avro map format: block_count, (key, value)..., 0 (end)
        // Value is nullable, so we need union index (1=non-null, zigzag: 2) before each value
        let mut avro_data = Vec::new();
        for i in 0..rows {
            // id
            avro_data.extend(encode_varint_i64(i as i64));
            // map with 2 entries: {"a": i*10, "b": i*10+1}
            avro_data.extend(encode_varint_i64(2)); // block count
            avro_data.extend(encode_bytes(b"a"));
            avro_data.extend(encode_varint_i64(1)); // union index: non-null (zigzag: 2)
            avro_data.extend(encode_varint_i32((i * 10) as i32));
            avro_data.extend(encode_bytes(b"b"));
            avro_data.extend(encode_varint_i64(1)); // union index: non-null (zigzag: 2)
            avro_data.extend(encode_varint_i32((i * 10 + 1) as i32));
            avro_data.extend(encode_varint_i64(0)); // end of map
        }

        // Decode
        let bytes_consumed = decoder.decode(&avro_data, rows).unwrap();
        assert!(bytes_consumed > 0);

        // Flush and verify
        let batch = decoder.flush().unwrap();
        assert_eq!(batch.num_rows(), rows);
        assert_eq!(batch.num_columns(), 2);

        // Verify id column
        let id_col = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(id_col.value(0), 0);

        // Verify map column structure
        let map_col = batch.column(1).as_any().downcast_ref::<MapArray>().unwrap();
        assert_eq!(map_col.len(), rows);

        // Each row should have 2 entries
        assert_eq!(map_col.value_length(0), 2);
        assert_eq!(map_col.value_length(1), 2);
    }

    // NOTE: test_roundtrip_map_string_string_preallocated was removed because
    // preallocated Map mode was removed in favor of Vec-only mode.
    // Map functionality is covered by test_roundtrip_map_string_int.

    #[test]
    fn test_roundtrip_union_schema() {
        use arrow_array::UnionArray;

        // Test Union schema: ["null", "string"]
        let union_fields = UnionFields::new(
            vec![0, 1],
            vec![
                Field::new("null", DataType::Null, true),
                Field::new("string", DataType::Utf8, false),
            ],
        );

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("optional_name", DataType::Union(union_fields, UnionMode::Dense), false),
        ]));

        let rows = 10;
        let layout = BufferLayout {
            columns: vec![
                ColumnRegion::fixed(0, rows * 8), // id
                ColumnRegion::fixed(0, 0),        // optional_name (Union uses Vec internally)
            ],
            total_size: rows * 8,
            row_count: rows,
        };

        let buffer = vec![0u8; layout.total_size.max(1)];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        // Encode test data - Avro union format: type_index (varint), then value
        let mut avro_data = Vec::new();
        for i in 0..rows {
            // id
            avro_data.extend(encode_varint_i64(i as i64));
            // optional_name: alternate between null (index 0) and string (index 1)
            if i % 2 == 0 {
                avro_data.extend(encode_varint_i64(0)); // null type
                // no value for null
            } else {
                avro_data.extend(encode_varint_i64(1)); // string type
                let name = format!("name_{}", i);
                avro_data.extend(encode_bytes(name.as_bytes()));
            }
        }

        // Decode
        let bytes_consumed = decoder.decode(&avro_data, rows).unwrap();
        assert!(bytes_consumed > 0);

        // Flush and verify
        let batch = decoder.flush().unwrap();
        assert_eq!(batch.num_rows(), rows);
        assert_eq!(batch.num_columns(), 2);

        // Verify id column
        let id_col = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(id_col.value(0), 0);
        assert_eq!(id_col.value(5), 5);

        // Verify union column
        let union_col = batch.column(1).as_any().downcast_ref::<UnionArray>().unwrap();
        assert_eq!(union_col.len(), rows);

        // Check type IDs alternate: 0 (null), 1 (string), 0, 1, ...
        assert_eq!(union_col.type_id(0), 0); // null
        assert_eq!(union_col.type_id(1), 1); // string
        assert_eq!(union_col.type_id(2), 0); // null
        assert_eq!(union_col.type_id(3), 1); // string
    }

    #[test]
    fn test_roundtrip_struct_schema() {
        use arrow_array::StructArray;

        // Test nested struct schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("address", DataType::Struct(
                vec![
                    Field::new("street", DataType::Utf8, false),
                    Field::new("city", DataType::Utf8, false),
                    Field::new("zip", DataType::Int32, false),
                ].into()
            ), false),
        ]));

        let rows = 5;
        let layout = BufferLayout {
            columns: vec![
                ColumnRegion::fixed(0, rows * 8), // id
                ColumnRegion::fixed(0, 0),        // address (Struct uses Vec internally)
            ],
            total_size: rows * 8,
            row_count: rows,
        };

        let buffer = vec![0u8; layout.total_size.max(1)];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        // Encode test data - Avro record format: fields in order, no type tags
        let mut avro_data = Vec::new();
        for i in 0..rows {
            // id
            avro_data.extend(encode_varint_i64(i as i64));
            // address struct: street, city, zip
            let street = format!("{} Main St", i * 100);
            let city = format!("City{}", i);
            avro_data.extend(encode_bytes(street.as_bytes()));
            avro_data.extend(encode_bytes(city.as_bytes()));
            avro_data.extend(encode_varint_i32((10000 + i) as i32));
        }

        // Decode
        let bytes_consumed = decoder.decode(&avro_data, rows).unwrap();
        assert!(bytes_consumed > 0);

        // Flush and verify
        let batch = decoder.flush().unwrap();
        assert_eq!(batch.num_rows(), rows);
        assert_eq!(batch.num_columns(), 2);

        // Verify id column
        let id_col = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(id_col.value(0), 0);
        assert_eq!(id_col.value(4), 4);

        // Verify struct column
        let struct_col = batch.column(1).as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(struct_col.len(), rows);
        assert_eq!(struct_col.num_columns(), 3);

        // Check street field
        let street_col = struct_col.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(street_col.value(0), "0 Main St");
        assert_eq!(street_col.value(3), "300 Main St");

        // Check city field
        let city_col = struct_col.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(city_col.value(0), "City0");

        // Check zip field
        let zip_col = struct_col.column(2).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(zip_col.value(0), 10000);
        assert_eq!(zip_col.value(4), 10004);
    }

    #[test]
    fn test_roundtrip_list_of_strings() {
        use arrow_array::ListArray;

        // Test List<String> schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("tags", DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))), false),
        ]));

        let rows = 5;
        let layout = BufferLayout {
            columns: vec![
                ColumnRegion::fixed(0, 0), // tags (List uses Vec internally)
            ],
            total_size: 1,
            row_count: rows,
        };

        let buffer = vec![0u8; layout.total_size];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        // Encode test data
        let mut avro_data = Vec::new();
        for i in 0..rows {
            // tags array: ["tag_i_0", "tag_i_1"]
            avro_data.extend(encode_varint_i64(2)); // block count
            avro_data.extend(encode_bytes(format!("tag_{}_0", i).as_bytes()));
            avro_data.extend(encode_bytes(format!("tag_{}_1", i).as_bytes()));
            avro_data.extend(encode_varint_i64(0)); // end of array
        }

        // Decode
        let bytes_consumed = decoder.decode(&avro_data, rows).unwrap();
        assert!(bytes_consumed > 0);

        // Flush and verify
        let batch = decoder.flush().unwrap();
        assert_eq!(batch.num_rows(), rows);

        // Verify list column
        let list_col = batch.column(0).as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(list_col.len(), rows);

        // Check first row
        let first_list = list_col.value(0);
        let first_values = first_list.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(first_values.len(), 2);
        assert_eq!(first_values.value(0), "tag_0_0");
        assert_eq!(first_values.value(1), "tag_0_1");
    }

    #[test]
    fn test_roundtrip_boolean() {
        use arrow_array::BooleanArray;

        // Test Boolean schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("active", DataType::Boolean, false),
            Field::new("verified", DataType::Boolean, false),
        ]));

        let rows = 10;
        // Boolean in universal decode path uses 1 byte per value
        let layout = BufferLayout {
            columns: vec![
                ColumnRegion::fixed(0, rows),      // active
                ColumnRegion::fixed(rows, rows),   // verified
            ],
            total_size: rows * 2,
            row_count: rows,
        };

        let buffer = vec![0u8; layout.total_size];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        // Encode test data - Avro boolean is 1 byte: 0 = false, 1 = true
        let mut avro_data = Vec::new();
        for i in 0..rows {
            avro_data.push(if i % 2 == 0 { 1 } else { 0 }); // active: alternate
            avro_data.push(if i % 3 == 0 { 1 } else { 0 }); // verified: every 3rd
        }

        // Decode
        let bytes_consumed = decoder.decode(&avro_data, rows).unwrap();
        assert_eq!(bytes_consumed, rows * 2);

        // Flush and verify
        let batch = decoder.flush().unwrap();
        assert_eq!(batch.num_rows(), rows);

        let active_col = batch.column(0).as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(active_col.value(0));  // true
        assert!(!active_col.value(1)); // false
        assert!(active_col.value(2));  // true

        let verified_col = batch.column(1).as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(verified_col.value(0));  // true (0 % 3 == 0)
        assert!(!verified_col.value(1)); // false
        assert!(!verified_col.value(2)); // false
        assert!(verified_col.value(3));  // true (3 % 3 == 0)
    }

    #[test]
    fn test_roundtrip_binary_and_fixed() {
        use arrow_array::{BinaryArray, FixedSizeBinaryArray};

        // Test Binary and FixedSizeBinary schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("data", DataType::Binary, false),
            Field::new("checksum", DataType::FixedSizeBinary(4), false),
        ]));

        let rows = 5;
        let binary_capacity = rows * 20;
        let offsets_capacity = (rows + 1) * 4;
        let fixed_capacity = rows * 4;

        let layout = BufferLayout {
            columns: vec![
                ColumnRegion {
                    data_offset: 0,
                    data_capacity: binary_capacity,
                    offsets_offset: Some(binary_capacity),
                    offsets_capacity: Some(offsets_capacity),
                    null_bitmap_offset: None,
                    null_bitmap_capacity: None,
                    child_regions: vec![],
                },
                ColumnRegion::fixed(binary_capacity + offsets_capacity, fixed_capacity),
            ],
            total_size: binary_capacity + offsets_capacity + fixed_capacity,
            row_count: rows,
        };

        let buffer = vec![0u8; layout.total_size];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        // Encode test data
        let mut avro_data = Vec::new();
        for i in 0..rows {
            // data: variable length bytes
            let data: Vec<u8> = (0..10).map(|j| ((i + j) % 256) as u8).collect();
            avro_data.extend(encode_bytes(&data));
            // checksum: fixed 4 bytes
            avro_data.extend(&[(i as u8), (i as u8) + 1, (i as u8) + 2, (i as u8) + 3]);
        }

        // Decode
        let bytes_consumed = decoder.decode(&avro_data, rows).unwrap();
        assert!(bytes_consumed > 0);

        // Flush and verify
        let batch = decoder.flush().unwrap();
        assert_eq!(batch.num_rows(), rows);

        // Verify binary column
        let binary_col = batch.column(0).as_any().downcast_ref::<BinaryArray>().unwrap();
        assert_eq!(binary_col.value(0).len(), 10);
        assert_eq!(binary_col.value(0)[0], 0);

        // Verify fixed binary column
        let fixed_col = batch.column(1).as_any().downcast_ref::<FixedSizeBinaryArray>().unwrap();
        assert_eq!(fixed_col.value(0), &[0, 1, 2, 3]);
        assert_eq!(fixed_col.value(1), &[1, 2, 3, 4]);
    }

    // ==========================================================================
    // DEEPLY NESTED TYPE TESTS (verifying recursive decode_value_to_vec)
    // ==========================================================================

    #[test]
    fn test_roundtrip_list_of_structs() {
        use arrow_array::{ListArray, StructArray};

        // Schema: List<Struct{name: String, value: Int32}>
        let struct_fields: Fields = vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("value", DataType::Int32, false),
        ].into();
        let schema = Arc::new(Schema::new(vec![
            Field::new("items", DataType::List(Arc::new(
                Field::new("item", DataType::Struct(struct_fields), false)
            )), false),
        ]));

        let rows = 3;
        let layout = BufferLayout {
            columns: vec![
                ColumnRegion::fixed(0, 0), // List uses Vec internally
            ],
            total_size: 1,
            row_count: rows,
        };

        let buffer = vec![0u8; layout.total_size.max(1)];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        // Encode: row0=[{name:"a",value:1},{name:"b",value:2}], row1=[], row2=[{name:"c",value:3}]
        let mut avro_data = Vec::new();
        // Row 0: 2 items
        avro_data.extend(encode_varint_i64(2)); // block count
        avro_data.extend(encode_bytes(b"a"));
        avro_data.extend(encode_varint_i32(1));
        avro_data.extend(encode_bytes(b"b"));
        avro_data.extend(encode_varint_i32(2));
        avro_data.extend(encode_varint_i64(0)); // end of array
        // Row 1: 0 items (empty)
        avro_data.extend(encode_varint_i64(0)); // end of array
        // Row 2: 1 item
        avro_data.extend(encode_varint_i64(1)); // block count
        avro_data.extend(encode_bytes(b"c"));
        avro_data.extend(encode_varint_i32(3));
        avro_data.extend(encode_varint_i64(0)); // end of array

        let bytes_consumed = decoder.decode(&avro_data, rows).unwrap();
        assert!(bytes_consumed > 0);

        let batch = decoder.flush().unwrap();
        assert_eq!(batch.num_rows(), rows);

        let list_col = batch.column(0).as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(list_col.len(), 3);
        assert_eq!(list_col.value_length(0), 2); // row0 has 2 items
        assert_eq!(list_col.value_length(1), 0); // row1 is empty
        assert_eq!(list_col.value_length(2), 1); // row2 has 1 item

        // Get first row's struct array
        let row0_items = list_col.value(0);
        let struct_arr = row0_items.as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(struct_arr.len(), 2);

        let names = struct_arr.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(names.value(0), "a");
        assert_eq!(names.value(1), "b");

        let values = struct_arr.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(values.value(0), 1);
        assert_eq!(values.value(1), 2);
    }

    #[test]
    fn test_roundtrip_nested_struct() {
        use arrow_array::StructArray;

        // Schema: Struct{id: Int64, inner: Struct{x: Int32, y: Int32}}
        let inner_fields: Fields = vec![
            Field::new("x", DataType::Int32, false),
            Field::new("y", DataType::Int32, false),
        ].into();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("inner", DataType::Struct(inner_fields), false),
        ]));

        let rows = 3;
        let layout = BufferLayout {
            columns: vec![
                ColumnRegion::fixed(0, rows * 8), // id
                ColumnRegion::fixed(0, 0), // inner struct uses Vec
            ],
            total_size: rows * 8,
            row_count: rows,
        };

        let buffer = vec![0u8; layout.total_size.max(1)];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        let mut avro_data = Vec::new();
        for i in 0..rows {
            avro_data.extend(encode_varint_i64(i as i64));      // id
            avro_data.extend(encode_varint_i32((i * 10) as i32)); // x
            avro_data.extend(encode_varint_i32((i * 20) as i32)); // y
        }

        let bytes_consumed = decoder.decode(&avro_data, rows).unwrap();
        assert!(bytes_consumed > 0);

        let batch = decoder.flush().unwrap();
        assert_eq!(batch.num_rows(), rows);

        let id_col = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(id_col.value(0), 0);
        assert_eq!(id_col.value(2), 2);

        let inner_col = batch.column(1).as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(inner_col.len(), rows);

        let x_col = inner_col.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let y_col = inner_col.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(x_col.value(0), 0);
        assert_eq!(y_col.value(0), 0);
        assert_eq!(x_col.value(2), 20);
        assert_eq!(y_col.value(2), 40);
    }

    #[test]
    fn test_roundtrip_list_of_lists() {
        use arrow_array::ListArray;

        // Schema: List<List<Int32>> (2D array)
        let schema = Arc::new(Schema::new(vec![
            Field::new("matrix", DataType::List(Arc::new(
                Field::new("row", DataType::List(Arc::new(
                    Field::new("item", DataType::Int32, false)
                )), false)
            )), false),
        ]));

        let rows = 2;
        let layout = BufferLayout {
            columns: vec![
                ColumnRegion::fixed(0, 0), // List uses Vec
            ],
            total_size: 1,
            row_count: rows,
        };

        let buffer = vec![0u8; layout.total_size.max(1)];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        // Row 0: [[1,2],[3]]
        // Row 1: [[4,5,6]]
        let mut avro_data = Vec::new();
        // Row 0: 2 inner lists
        avro_data.extend(encode_varint_i64(2)); // outer block count
        // Inner list 0: [1,2]
        avro_data.extend(encode_varint_i64(2));
        avro_data.extend(encode_varint_i32(1));
        avro_data.extend(encode_varint_i32(2));
        avro_data.extend(encode_varint_i64(0)); // end inner
        // Inner list 1: [3]
        avro_data.extend(encode_varint_i64(1));
        avro_data.extend(encode_varint_i32(3));
        avro_data.extend(encode_varint_i64(0)); // end inner
        avro_data.extend(encode_varint_i64(0)); // end outer
        // Row 1: 1 inner list
        avro_data.extend(encode_varint_i64(1)); // outer block count
        // Inner list 0: [4,5,6]
        avro_data.extend(encode_varint_i64(3));
        avro_data.extend(encode_varint_i32(4));
        avro_data.extend(encode_varint_i32(5));
        avro_data.extend(encode_varint_i32(6));
        avro_data.extend(encode_varint_i64(0)); // end inner
        avro_data.extend(encode_varint_i64(0)); // end outer

        let bytes_consumed = decoder.decode(&avro_data, rows).unwrap();
        assert!(bytes_consumed > 0);

        let batch = decoder.flush().unwrap();
        assert_eq!(batch.num_rows(), rows);

        let outer_list = batch.column(0).as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(outer_list.len(), 2);
        assert_eq!(outer_list.value_length(0), 2); // row0 has 2 inner lists
        assert_eq!(outer_list.value_length(1), 1); // row1 has 1 inner list

        // Check first row's inner lists
        let row0 = outer_list.value(0);
        let inner_lists = row0.as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(inner_lists.len(), 2);
        assert_eq!(inner_lists.value_length(0), 2); // [1,2]
        assert_eq!(inner_lists.value_length(1), 1); // [3]

        let first_inner = inner_lists.value(0);
        let int_arr = first_inner.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(int_arr.value(0), 1);
        assert_eq!(int_arr.value(1), 2);
    }

    #[test]
    fn test_roundtrip_map_of_structs() {
        use arrow_array::{MapArray, StructArray};

        // Schema: Map<String, Struct{x: Int32, y: Int32}>
        let value_fields: Fields = vec![
            Field::new("x", DataType::Int32, false),
            Field::new("y", DataType::Int32, false),
        ].into();
        let schema = Arc::new(Schema::new(vec![
            Field::new("coords", DataType::Map(
                Arc::new(Field::new("entries", DataType::Struct(vec![
                    Field::new("key", DataType::Utf8, false),
                    Field::new("value", DataType::Struct(value_fields), true),
                ].into()), false)),
                false
            ), false),
        ]));

        let rows = 2;
        let layout = BufferLayout {
            columns: vec![
                ColumnRegion::fixed(0, 0), // Map uses Vec
            ],
            total_size: 1,
            row_count: rows,
        };

        let buffer = vec![0u8; layout.total_size.max(1)];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        // Row 0: {"a": {x:1, y:2}, "b": {x:3, y:4}}
        // Row 1: {"c": {x:5, y:6}}
        // Value (Struct) is nullable, so need union index (1=non-null) before each struct
        let mut avro_data = Vec::new();
        // Row 0: 2 entries
        avro_data.extend(encode_varint_i64(2));
        avro_data.extend(encode_bytes(b"a"));
        avro_data.extend(encode_varint_i64(1)); // union index: non-null (zigzag: 2)
        avro_data.extend(encode_varint_i32(1)); // x
        avro_data.extend(encode_varint_i32(2)); // y
        avro_data.extend(encode_bytes(b"b"));
        avro_data.extend(encode_varint_i64(1)); // union index: non-null (zigzag: 2)
        avro_data.extend(encode_varint_i32(3)); // x
        avro_data.extend(encode_varint_i32(4)); // y
        avro_data.extend(encode_varint_i64(0)); // end map
        // Row 1: 1 entry
        avro_data.extend(encode_varint_i64(1));
        avro_data.extend(encode_bytes(b"c"));
        avro_data.extend(encode_varint_i64(1)); // union index: non-null (zigzag: 2)
        avro_data.extend(encode_varint_i32(5)); // x
        avro_data.extend(encode_varint_i32(6)); // y
        avro_data.extend(encode_varint_i64(0)); // end map

        let bytes_consumed = decoder.decode(&avro_data, rows).unwrap();
        assert!(bytes_consumed > 0);

        let batch = decoder.flush().unwrap();
        assert_eq!(batch.num_rows(), rows);

        let map_col = batch.column(0).as_any().downcast_ref::<MapArray>().unwrap();
        assert_eq!(map_col.len(), 2);
        assert_eq!(map_col.value_length(0), 2); // row0 has 2 entries
        assert_eq!(map_col.value_length(1), 1); // row1 has 1 entry

        // Get keys from first row
        let row0 = map_col.value(0);
        let keys = row0.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(keys.value(0), "a");
        assert_eq!(keys.value(1), "b");

        // Get struct values from first row
        let values = row0.column(1).as_any().downcast_ref::<StructArray>().unwrap();
        let x_col = values.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let y_col = values.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(x_col.value(0), 1);
        assert_eq!(y_col.value(0), 2);
        assert_eq!(x_col.value(1), 3);
        assert_eq!(y_col.value(1), 4);
    }

    #[test]
    fn test_roundtrip_struct_with_list_field() {
        use arrow_array::{StructArray, ListArray};

        // Schema: Struct{id: Int64, tags: List<String>}
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("data", DataType::Struct(vec![
                Field::new("name", DataType::Utf8, false),
                Field::new("tags", DataType::List(Arc::new(
                    Field::new("item", DataType::Utf8, false)
                )), false),
            ].into()), false),
        ]));

        let rows = 2;
        let layout = BufferLayout {
            columns: vec![
                ColumnRegion::fixed(0, rows * 8), // id
                ColumnRegion::fixed(0, 0), // struct uses Vec
            ],
            total_size: rows * 8,
            row_count: rows,
        };

        let buffer = vec![0u8; layout.total_size.max(1)];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        // Row 0: {id: 1, data: {name: "alice", tags: ["a", "b"]}}
        // Row 1: {id: 2, data: {name: "bob", tags: []}}
        let mut avro_data = Vec::new();
        // Row 0
        avro_data.extend(encode_varint_i64(1)); // id
        avro_data.extend(encode_bytes(b"alice")); // name
        avro_data.extend(encode_varint_i64(2)); // tags block
        avro_data.extend(encode_bytes(b"a"));
        avro_data.extend(encode_bytes(b"b"));
        avro_data.extend(encode_varint_i64(0)); // end tags
        // Row 1
        avro_data.extend(encode_varint_i64(2)); // id
        avro_data.extend(encode_bytes(b"bob")); // name
        avro_data.extend(encode_varint_i64(0)); // empty tags

        let bytes_consumed = decoder.decode(&avro_data, rows).unwrap();
        assert!(bytes_consumed > 0);

        let batch = decoder.flush().unwrap();
        assert_eq!(batch.num_rows(), rows);

        let id_col = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(id_col.value(0), 1);
        assert_eq!(id_col.value(1), 2);

        let struct_col = batch.column(1).as_any().downcast_ref::<StructArray>().unwrap();
        let name_col = struct_col.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(name_col.value(0), "alice");
        assert_eq!(name_col.value(1), "bob");

        let tags_col = struct_col.column(1).as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(tags_col.value_length(0), 2); // alice has 2 tags
        assert_eq!(tags_col.value_length(1), 0); // bob has 0 tags

        let alice_tags = tags_col.value(0);
        let tags_arr = alice_tags.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(tags_arr.value(0), "a");
        assert_eq!(tags_arr.value(1), "b");
    }

    #[test]
    fn test_list_utf8_preallocated() {
        use arrow_array::ListArray;
        use crate::buffer_provider::align_to;

        // Test List<String> with pre-allocated buffers (not Vec mode)
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("tags", DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))), false),
        ]));

        let rows = 5;
        let max_tags = 3; // max tags per row
        let max_tag_len = 20;
        let total_items = rows * max_tags;

        // Calculate aligned layout for List<Utf8>
        let mut offset = 0;

        // id: Int64 (8-byte aligned)
        offset = align_to(offset, 8);
        let id_offset = offset;
        let id_cap = rows * 8;
        offset += id_cap;

        // tags: List<Utf8>
        // List offsets: (rows + 1) * 4
        offset = align_to(offset, 4);
        let list_off_offset = offset;
        let list_off_cap = (rows + 1) * 4;
        offset += list_off_cap;

        // Child string data
        let child_data_offset = offset;
        let child_data_cap = total_items * max_tag_len;
        offset += child_data_cap;

        // Child string offsets: (total_items + 1) * 4
        offset = align_to(offset, 4);
        let child_off_offset = offset;
        let child_off_cap = (total_items + 1) * 4;
        offset += child_off_cap;

        let total_size = align_to(offset, 8);

        let layout = BufferLayout {
            columns: vec![
                ColumnRegion::fixed(id_offset, id_cap),
                ColumnRegion {
                    data_offset: list_off_offset,
                    data_capacity: 0, // List itself has no data
                    offsets_offset: Some(list_off_offset),
                    offsets_capacity: Some(list_off_cap),
                    null_bitmap_offset: None,
                    null_bitmap_capacity: None,
                    child_regions: vec![
                        ColumnRegion {
                            data_offset: child_data_offset,
                            data_capacity: child_data_cap,
                            offsets_offset: Some(child_off_offset),
                            offsets_capacity: Some(child_off_cap),
                            null_bitmap_offset: None,
                            null_bitmap_capacity: None,
                            child_regions: vec![],
                        },
                    ],
                },
            ],
            total_size,
            row_count: rows,
        };

        let buffer = vec![0u8; layout.total_size];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        // Encode test data
        let mut avro_data = Vec::new();
        for i in 0..rows {
            // id
            avro_data.extend(encode_varint_i64(i as i64 * 100));
            // tags array
            let num_tags = (i % 3) + 1; // 1-3 tags
            avro_data.extend(encode_varint_i64(num_tags as i64)); // block count
            for j in 0..num_tags {
                avro_data.extend(encode_bytes(format!("tag_{}_{}", i, j).as_bytes()));
            }
            avro_data.extend(encode_varint_i64(0)); // end of array
        }

        // Decode
        let bytes_consumed = decoder.decode(&avro_data, rows).unwrap();
        assert!(bytes_consumed > 0);

        // Flush and verify
        let batch = decoder.flush().unwrap();
        assert_eq!(batch.num_rows(), rows);

        // Verify id column
        let id_col = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(id_col.value(0), 0);
        assert_eq!(id_col.value(1), 100);
        assert_eq!(id_col.value(2), 200);

        // Verify list column
        let list_col = batch.column(1).as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(list_col.len(), rows);

        // Check row 0: 1 tag
        let row0 = list_col.value(0);
        let row0_strings = row0.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(row0_strings.len(), 1);
        assert_eq!(row0_strings.value(0), "tag_0_0");

        // Check row 1: 2 tags
        let row1 = list_col.value(1);
        let row1_strings = row1.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(row1_strings.len(), 2);
        assert_eq!(row1_strings.value(0), "tag_1_0");
        assert_eq!(row1_strings.value(1), "tag_1_1");

        // Check row 2: 3 tags
        let row2 = list_col.value(2);
        let row2_strings = row2.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(row2_strings.len(), 3);
        assert_eq!(row2_strings.value(0), "tag_2_0");
        assert_eq!(row2_strings.value(1), "tag_2_1");
        assert_eq!(row2_strings.value(2), "tag_2_2");
    }

    #[test]
    fn test_list_utf8_parity_pooled_vs_vec() {
        use arrow_array::ListArray;
        use crate::buffer_provider::align_to;

        // Test that pre-allocated List<Utf8> produces same results as Vec mode
        let schema = Arc::new(Schema::new(vec![
            Field::new("tags", DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))), false),
        ]));

        let rows = 10;

        // Encode test data (same for both decoders)
        let mut avro_data = Vec::new();
        for i in 0..rows {
            let num_tags = (i % 4) + 1; // 1-4 tags
            avro_data.extend(encode_varint_i64(num_tags as i64));
            for j in 0..num_tags {
                avro_data.extend(encode_bytes(format!("item_{}_{}", i, j).as_bytes()));
            }
            avro_data.extend(encode_varint_i64(0));
        }

        // === Vec mode decoder ===
        let vec_layout = BufferLayout {
            columns: vec![ColumnRegion::fixed(0, 0)], // Vec mode
            total_size: 1,
            row_count: rows,
        };
        let vec_buffer = vec![0u8; vec_layout.total_size];
        let vec_provider = PooledBufferProvider::new(vec_buffer, vec_layout);
        let mut vec_decoder = PooledRecordDecoder::try_new(schema.clone(), vec_provider).unwrap();

        let _ = vec_decoder.decode(&avro_data, rows).unwrap();
        let vec_batch = vec_decoder.flush().unwrap();

        // === Pre-allocated mode decoder ===
        let max_items = rows * 4;
        let max_item_len = 20;

        let list_off_cap = (rows + 1) * 4;
        let child_data_cap = max_items * max_item_len;
        let child_off_cap = (max_items + 1) * 4;

        let mut offset = 0;
        offset = align_to(offset, 4);
        let list_off_offset = offset;
        offset += list_off_cap;

        let child_data_offset = offset;
        offset += child_data_cap;

        offset = align_to(offset, 4);
        let child_off_offset = offset;
        offset += child_off_cap;

        let total_size = align_to(offset, 8);

        let pooled_layout = BufferLayout {
            columns: vec![
                ColumnRegion {
                    data_offset: list_off_offset,
                    data_capacity: 0,
                    offsets_offset: Some(list_off_offset),
                    offsets_capacity: Some(list_off_cap),
                    null_bitmap_offset: None,
                    null_bitmap_capacity: None,
                    child_regions: vec![
                        ColumnRegion {
                            data_offset: child_data_offset,
                            data_capacity: child_data_cap,
                            offsets_offset: Some(child_off_offset),
                            offsets_capacity: Some(child_off_cap),
                            null_bitmap_offset: None,
                            null_bitmap_capacity: None,
                            child_regions: vec![],
                        },
                    ],
                },
            ],
            total_size,
            row_count: rows,
        };

        let pooled_buffer = vec![0u8; pooled_layout.total_size];
        let pooled_provider = PooledBufferProvider::new(pooled_buffer, pooled_layout);
        let mut pooled_decoder = PooledRecordDecoder::try_new(schema, pooled_provider).unwrap();

        let _ = pooled_decoder.decode(&avro_data, rows).unwrap();
        let pooled_batch = pooled_decoder.flush().unwrap();

        // === Compare results ===
        assert_eq!(vec_batch.num_rows(), pooled_batch.num_rows());

        let vec_list = vec_batch.column(0).as_any().downcast_ref::<ListArray>().unwrap();
        let pooled_list = pooled_batch.column(0).as_any().downcast_ref::<ListArray>().unwrap();

        assert_eq!(vec_list.len(), pooled_list.len());

        for row in 0..rows {
            let vec_row = vec_list.value(row);
            let pooled_row = pooled_list.value(row);

            let vec_strings = vec_row.as_any().downcast_ref::<StringArray>().unwrap();
            let pooled_strings = pooled_row.as_any().downcast_ref::<StringArray>().unwrap();

            assert_eq!(vec_strings.len(), pooled_strings.len(), "Row {} length mismatch", row);

            for i in 0..vec_strings.len() {
                assert_eq!(
                    vec_strings.value(i),
                    pooled_strings.value(i),
                    "Row {} item {} mismatch",
                    row,
                    i
                );
            }
        }
    }

    #[test]
    fn test_simple_utf8_preallocated() {
        use crate::buffer_provider::align_to;

        // Test simple String field with pre-allocated buffers
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let rows = 5;
        let max_name_len = 30;

        // Calculate aligned layout
        let mut offset = 0;

        // id: Int64 (8-byte aligned)
        offset = align_to(offset, 8);
        let id_offset = offset;
        let id_cap = rows * 8;
        offset += id_cap;

        // name: Utf8
        let name_data_offset = offset;
        let name_data_cap = rows * max_name_len;
        offset += name_data_cap;

        offset = align_to(offset, 4);
        let name_off_offset = offset;
        let name_off_cap = (rows + 1) * 4;
        offset += name_off_cap;

        let total_size = align_to(offset, 8);

        let layout = BufferLayout {
            columns: vec![
                ColumnRegion::fixed(id_offset, id_cap),
                ColumnRegion {
                    data_offset: name_data_offset,
                    data_capacity: name_data_cap,
                    offsets_offset: Some(name_off_offset),
                    offsets_capacity: Some(name_off_cap),
                    null_bitmap_offset: None,
                    null_bitmap_capacity: None,
                    child_regions: vec![],
                },
            ],
            total_size,
            row_count: rows,
        };

        let buffer = vec![0u8; layout.total_size];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        // Encode test data
        let mut avro_data = Vec::new();
        let names = ["alice", "bob", "charlie", "david", "eve"];
        for (i, name) in names.iter().enumerate() {
            avro_data.extend(encode_varint_i64(i as i64));
            avro_data.extend(encode_bytes(name.as_bytes()));
        }

        // Decode
        let bytes_consumed = decoder.decode(&avro_data, rows).unwrap();
        assert!(bytes_consumed > 0);

        // Flush and verify
        let batch = decoder.flush().unwrap();
        assert_eq!(batch.num_rows(), rows);

        let id_col = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let name_col = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();

        for (i, name) in names.iter().enumerate() {
            assert_eq!(id_col.value(i), i as i64);
            assert_eq!(name_col.value(i), *name);
        }
    }

    #[test]
    fn test_utf8_multiple_columns() {
        use crate::buffer_provider::align_to;

        // Test multiple Utf8 columns with pre-allocated buffers
        // Note: strings-only schema always uses pre-allocated path (no Vec fallback)
        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("desc", DataType::Utf8, false),
        ]));

        let rows = 10;
        let max_len = 50;

        // Calculate aligned layout
        let mut offset = 0;

        let name_data_offset = offset;
        let name_data_cap = rows * max_len;
        offset += name_data_cap;

        offset = align_to(offset, 4);
        let name_off_offset = offset;
        let name_off_cap = (rows + 1) * 4;
        offset += name_off_cap;

        let desc_data_offset = offset;
        let desc_data_cap = rows * max_len;
        offset += desc_data_cap;

        offset = align_to(offset, 4);
        let desc_off_offset = offset;
        let desc_off_cap = (rows + 1) * 4;
        offset += desc_off_cap;

        let total_size = align_to(offset, 8);

        let layout = BufferLayout {
            columns: vec![
                ColumnRegion {
                    data_offset: name_data_offset,
                    data_capacity: name_data_cap,
                    offsets_offset: Some(name_off_offset),
                    offsets_capacity: Some(name_off_cap),
                    null_bitmap_offset: None,
                    null_bitmap_capacity: None,
                    child_regions: vec![],
                },
                ColumnRegion {
                    data_offset: desc_data_offset,
                    data_capacity: desc_data_cap,
                    offsets_offset: Some(desc_off_offset),
                    offsets_capacity: Some(desc_off_cap),
                    null_bitmap_offset: None,
                    null_bitmap_capacity: None,
                    child_regions: vec![],
                },
            ],
            total_size,
            row_count: rows,
        };

        let buffer = vec![0u8; layout.total_size];
        let provider = PooledBufferProvider::new(buffer, layout);
        let mut decoder = PooledRecordDecoder::try_new(schema, provider).unwrap();

        // Encode test data
        let mut avro_data = Vec::new();
        for i in 0..rows {
            avro_data.extend(encode_bytes(format!("name_{}", i).as_bytes()));
            avro_data.extend(encode_bytes(format!("description_for_{}", i).as_bytes()));
        }

        // Decode
        let _ = decoder.decode(&avro_data, rows).unwrap();
        let batch = decoder.flush().unwrap();

        // Verify
        assert_eq!(batch.num_rows(), rows);

        let name_col = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let desc_col = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();

        for i in 0..rows {
            assert_eq!(name_col.value(i), format!("name_{}", i));
            assert_eq!(desc_col.value(i), format!("description_for_{}", i));
        }
    }
}

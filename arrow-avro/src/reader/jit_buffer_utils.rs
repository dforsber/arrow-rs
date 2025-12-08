//! Buffer-to-Array conversion utilities for JIT decoder output
//!
//! These utilities convert raw decoded buffers from `JitDecoder::decode_checked()`
//! into Arrow arrays and RecordBatches. The conversion is zero-copy when buffers
//! are properly aligned.
//!
//! # Usage
//! ```ignore
//! use arrow_avro::reader::jit_buffer_utils::{buffers_to_batch, BufferSet};
//!
//! // After decode_checked fills buffers...
//! let batch = buffers_to_batch(&schema, &buffer_sets, row_count, &decoder)?;
//! ```

use arrow_array::{
    types::{IntervalDayTime, IntervalMonthDayNano},
    ArrayRef, BinaryArray, BooleanArray, Date32Array, Date64Array, Decimal128Array,
    Decimal256Array, DurationMicrosecondArray, DurationMillisecondArray,
    DurationNanosecondArray, DurationSecondArray, FixedSizeBinaryArray, Float32Array,
    Float64Array, Int16Array, Int32Array, Int64Array, Int8Array, IntervalDayTimeArray,
    IntervalMonthDayNanoArray, IntervalYearMonthArray, LargeBinaryArray, LargeStringArray,
    RecordBatch, StringArray, Time32MillisecondArray, Time32SecondArray,
    Time64MicrosecondArray, Time64NanosecondArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt16Array,
    UInt32Array, UInt64Array, UInt8Array,
};
use arrow_buffer::{i256, BooleanBuffer, Buffer, NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{ArrowError, DataType, Field, IntervalUnit, SchemaRef, TimeUnit};
use bytes::BytesMut;
use std::sync::Arc;

use super::jit_decoder::JitDecoder;

/// A set of buffers for a single column
#[derive(Debug)]
pub struct BufferSet {
    /// Primary buffer (data for fixed-size, offsets for variable-size)
    pub primary: Vec<u8>,
    /// Secondary buffer (data for variable-size columns, None for fixed-size)
    pub secondary: Option<Vec<u8>>,
    /// Validity bitmap (for nullable columns)
    pub validity: Option<Vec<u8>>,
    /// Actual number of bytes used in primary buffer
    pub primary_len: usize,
    /// Actual number of bytes used in secondary buffer
    pub secondary_len: usize,
}

impl BufferSet {
    /// Convert validity to NullBuffer (zero-copy, consumes validity)
    fn into_null_buffer(validity: Option<Vec<u8>>, row_count: usize) -> Option<NullBuffer> {
        validity.map(|v| {
            let buffer = Buffer::from_vec(v);
            NullBuffer::new(BooleanBuffer::new(buffer, 0, row_count))
        })
    }
}

impl BufferSet {
    /// Create a new buffer set for fixed-size column
    pub fn new_fixed(capacity: usize) -> Self {
        Self {
            primary: vec![0u8; capacity],
            secondary: None,
            validity: None,
            primary_len: 0,
            secondary_len: 0,
        }
    }

    /// Create a new buffer set for variable-size column
    pub fn new_variable(offsets_capacity: usize, data_capacity: usize) -> Self {
        Self {
            primary: vec![0u8; offsets_capacity],
            secondary: Some(vec![0u8; data_capacity]),
            validity: None,
            primary_len: 0,
            secondary_len: 0,
        }
    }

    /// Create a new buffer set with validity bitmap
    pub fn with_validity(mut self, validity_bytes: usize) -> Self {
        self.validity = Some(vec![0u8; validity_bytes]);
        self
    }

    /// Get mutable pointer to primary buffer
    pub fn primary_ptr(&mut self) -> *mut u8 {
        self.primary.as_mut_ptr()
    }

    /// Get mutable pointer to secondary buffer (panics if None)
    pub fn secondary_ptr(&mut self) -> *mut u8 {
        self.secondary.as_mut().unwrap().as_mut_ptr()
    }

    /// Get mutable pointer to validity buffer (panics if None)
    pub fn validity_ptr(&mut self) -> *mut u8 {
        self.validity.as_mut().unwrap().as_mut_ptr()
    }
}

/// A set of pooled buffers for a single column using BytesMut for zero-copy conversion
///
/// This struct uses `BytesMut` from the `bytes` crate instead of `Vec<u8>`,
/// enabling zero-copy conversion to Arrow buffers via `freeze()` + `Buffer::from()`.
/// Designed for use with a buffer pool to eliminate per-decode heap allocations.
#[derive(Debug)]
pub struct PooledBufferSet {
    /// Primary buffer (data for fixed-size, offsets for variable-size)
    pub primary: BytesMut,
    /// Secondary buffer (data for variable-size columns, None for fixed-size)
    pub secondary: Option<BytesMut>,
    /// Validity bitmap (for nullable columns)
    pub validity: Option<BytesMut>,
    /// Actual number of bytes used in primary buffer
    pub primary_len: usize,
    /// Actual number of bytes used in secondary buffer
    pub secondary_len: usize,
}

impl PooledBufferSet {
    /// Create a new pooled buffer set from pre-allocated BytesMut buffers
    ///
    /// This constructor takes ownership of pre-allocated buffers from a pool.
    /// The buffers should already have the required capacity.
    #[inline]
    pub fn from_pool(
        primary: BytesMut,
        secondary: Option<BytesMut>,
        validity: Option<BytesMut>,
    ) -> Self {
        Self {
            primary,
            secondary,
            validity,
            primary_len: 0,
            secondary_len: 0,
        }
    }

    /// Create a new buffer set for fixed-size column (allocates BytesMut)
    pub fn new_fixed(capacity: usize) -> Self {
        let mut primary = BytesMut::with_capacity(capacity);
        primary.resize(capacity, 0);
        Self {
            primary,
            secondary: None,
            validity: None,
            primary_len: 0,
            secondary_len: 0,
        }
    }

    /// Create a new buffer set for variable-size column (allocates BytesMut)
    pub fn new_variable(offsets_capacity: usize, data_capacity: usize) -> Self {
        let mut primary = BytesMut::with_capacity(offsets_capacity);
        primary.resize(offsets_capacity, 0);
        let mut secondary = BytesMut::with_capacity(data_capacity);
        secondary.resize(data_capacity, 0);
        Self {
            primary,
            secondary: Some(secondary),
            validity: None,
            primary_len: 0,
            secondary_len: 0,
        }
    }

    /// Add validity bitmap
    pub fn with_validity(mut self, validity_bytes: usize) -> Self {
        let mut validity = BytesMut::with_capacity(validity_bytes);
        validity.resize(validity_bytes, 0);
        self.validity = Some(validity);
        self
    }

    /// Get mutable pointer to primary buffer
    #[inline]
    pub fn primary_ptr(&mut self) -> *mut u8 {
        self.primary.as_mut_ptr()
    }

    /// Get capacity of primary buffer
    #[inline]
    pub fn primary_capacity(&self) -> usize {
        self.primary.len()
    }

    /// Get mutable pointer to secondary buffer (panics if None)
    #[inline]
    pub fn secondary_ptr(&mut self) -> *mut u8 {
        self.secondary.as_mut().unwrap().as_mut_ptr()
    }

    /// Get capacity of secondary buffer (panics if None)
    #[inline]
    pub fn secondary_capacity(&self) -> usize {
        self.secondary.as_ref().unwrap().len()
    }

    /// Get mutable pointer to validity buffer (panics if None)
    #[inline]
    pub fn validity_ptr(&mut self) -> *mut u8 {
        self.validity.as_mut().unwrap().as_mut_ptr()
    }

    /// Get capacity of validity buffer (panics if None)
    #[inline]
    pub fn validity_capacity(&self) -> usize {
        self.validity.as_ref().unwrap().len()
    }

    /// Convert validity to NullBuffer (zero-copy, consumes validity)
    fn into_null_buffer(validity: Option<BytesMut>, row_count: usize) -> Option<NullBuffer> {
        validity.map(|v| {
            let buffer = Buffer::from(v.freeze());
            NullBuffer::new(BooleanBuffer::new(buffer, 0, row_count))
        })
    }
}

/// Convert a single column's pooled buffers to an Arrow ArrayRef (zero-copy, consumes PooledBufferSet)
///
/// This version uses BytesMut.freeze() for zero-copy conversion to Arrow buffers.
///
/// # Arguments
/// * `field` - Arrow field describing the column type
/// * `buffer_set` - The decoded buffer data (consumed)
/// * `row_count` - Number of rows decoded
///
/// # Returns
/// Arrow ArrayRef wrapping the buffer data (zero-copy)
pub fn pooled_buffer_to_array(
    field: &Field,
    buffer_set: PooledBufferSet,
    row_count: usize,
) -> Result<ArrayRef, ArrowError> {
    let data_type = field.data_type();

    // Extract validity first (before consuming other buffers)
    let validity = buffer_set.validity;

    match data_type {
        // Boolean (1 bit per value, packed)
        DataType::Boolean => {
            let buffer = Buffer::from(buffer_set.primary.freeze());
            let bool_buffer = BooleanBuffer::new(buffer, 0, row_count);
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(BooleanArray::new(bool_buffer, null_buffer)))
        }

        // Signed integers
        DataType::Int8 => {
            let buffer = ScalarBuffer::<i8>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(Int8Array::new(buffer, null_buffer)))
        }
        DataType::Int16 => {
            let buffer = ScalarBuffer::<i16>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(Int16Array::new(buffer, null_buffer)))
        }
        DataType::Int32 => {
            let buffer = ScalarBuffer::<i32>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(Int32Array::new(buffer, null_buffer)))
        }
        DataType::Int64 => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(Int64Array::new(buffer, null_buffer)))
        }

        // Unsigned integers
        DataType::UInt8 => {
            let buffer = ScalarBuffer::<u8>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(UInt8Array::new(buffer, null_buffer)))
        }
        DataType::UInt16 => {
            let buffer = ScalarBuffer::<u16>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(UInt16Array::new(buffer, null_buffer)))
        }
        DataType::UInt32 => {
            let buffer = ScalarBuffer::<u32>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(UInt32Array::new(buffer, null_buffer)))
        }
        DataType::UInt64 => {
            let buffer = ScalarBuffer::<u64>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(UInt64Array::new(buffer, null_buffer)))
        }

        // Floating point
        DataType::Float32 => {
            let buffer = ScalarBuffer::<f32>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(Float32Array::new(buffer, null_buffer)))
        }
        DataType::Float64 => {
            let buffer = ScalarBuffer::<f64>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(Float64Array::new(buffer, null_buffer)))
        }

        // Date types
        DataType::Date32 => {
            let buffer = ScalarBuffer::<i32>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(Date32Array::new(buffer, null_buffer)))
        }
        DataType::Date64 => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(Date64Array::new(buffer, null_buffer)))
        }

        // Time types
        DataType::Time32(TimeUnit::Second) => {
            let buffer = ScalarBuffer::<i32>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(Time32SecondArray::new(buffer, null_buffer)))
        }
        DataType::Time32(TimeUnit::Millisecond) => {
            let buffer = ScalarBuffer::<i32>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(Time32MillisecondArray::new(buffer, null_buffer)))
        }
        DataType::Time64(TimeUnit::Microsecond) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(Time64MicrosecondArray::new(buffer, null_buffer)))
        }
        DataType::Time64(TimeUnit::Nanosecond) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(Time64NanosecondArray::new(buffer, null_buffer)))
        }

        // Timestamp types
        DataType::Timestamp(TimeUnit::Second, tz) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(TimestampSecondArray::new(buffer, null_buffer).with_timezone_opt(tz.clone())))
        }
        DataType::Timestamp(TimeUnit::Millisecond, tz) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(TimestampMillisecondArray::new(buffer, null_buffer).with_timezone_opt(tz.clone())))
        }
        DataType::Timestamp(TimeUnit::Microsecond, tz) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(TimestampMicrosecondArray::new(buffer, null_buffer).with_timezone_opt(tz.clone())))
        }
        DataType::Timestamp(TimeUnit::Nanosecond, tz) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(TimestampNanosecondArray::new(buffer, null_buffer).with_timezone_opt(tz.clone())))
        }

        // Duration types
        DataType::Duration(TimeUnit::Second) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(DurationSecondArray::new(buffer, null_buffer)))
        }
        DataType::Duration(TimeUnit::Millisecond) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(DurationMillisecondArray::new(buffer, null_buffer)))
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(DurationMicrosecondArray::new(buffer, null_buffer)))
        }
        DataType::Duration(TimeUnit::Nanosecond) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(DurationNanosecondArray::new(buffer, null_buffer)))
        }

        // Interval types
        DataType::Interval(IntervalUnit::YearMonth) => {
            let buffer = ScalarBuffer::<i32>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(IntervalYearMonthArray::new(buffer, null_buffer)))
        }
        DataType::Interval(IntervalUnit::DayTime) => {
            let buffer = ScalarBuffer::<IntervalDayTime>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(IntervalDayTimeArray::new(buffer, null_buffer)))
        }
        DataType::Interval(IntervalUnit::MonthDayNano) => {
            let buffer = ScalarBuffer::<IntervalMonthDayNano>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(IntervalMonthDayNanoArray::new(buffer, null_buffer)))
        }

        // Decimal types
        DataType::Decimal128(precision, scale) => {
            let buffer = ScalarBuffer::<i128>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(
                Decimal128Array::new(buffer, null_buffer)
                    .with_precision_and_scale(*precision, *scale)?,
            ))
        }
        DataType::Decimal256(precision, scale) => {
            let buffer = ScalarBuffer::<i256>::from(Buffer::from(buffer_set.primary.freeze()));
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(
                Decimal256Array::new(buffer, null_buffer)
                    .with_precision_and_scale(*precision, *scale)?,
            ))
        }

        // Fixed-size binary
        DataType::FixedSizeBinary(size) => {
            let buffer = Buffer::from(buffer_set.primary.freeze());
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(FixedSizeBinaryArray::new(*size, buffer, null_buffer)))
        }

        // Variable-length string (UTF-8)
        DataType::Utf8 => {
            let offsets_buffer = Buffer::from(buffer_set.primary.freeze());
            let data_buffer = Buffer::from(
                buffer_set
                    .secondary
                    .ok_or_else(|| ArrowError::InvalidArgumentError("Utf8 requires data buffer".into()))?
                    .freeze(),
            );
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            let offsets = OffsetBuffer::new(ScalarBuffer::<i32>::from(offsets_buffer));
            Ok(Arc::new(StringArray::new(offsets, data_buffer, null_buffer)))
        }

        // Large string (i64 offsets)
        DataType::LargeUtf8 => {
            let offsets_buffer = Buffer::from(buffer_set.primary.freeze());
            let data_buffer = Buffer::from(
                buffer_set
                    .secondary
                    .ok_or_else(|| ArrowError::InvalidArgumentError("LargeUtf8 requires data buffer".into()))?
                    .freeze(),
            );
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            let offsets = OffsetBuffer::new(ScalarBuffer::<i64>::from(offsets_buffer));
            Ok(Arc::new(LargeStringArray::new(offsets, data_buffer, null_buffer)))
        }

        // Variable-length binary
        DataType::Binary => {
            let offsets_buffer = Buffer::from(buffer_set.primary.freeze());
            let data_buffer = Buffer::from(
                buffer_set
                    .secondary
                    .ok_or_else(|| ArrowError::InvalidArgumentError("Binary requires data buffer".into()))?
                    .freeze(),
            );
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            let offsets = OffsetBuffer::new(ScalarBuffer::<i32>::from(offsets_buffer));
            Ok(Arc::new(BinaryArray::new(offsets, data_buffer, null_buffer)))
        }

        // Large binary (i64 offsets)
        DataType::LargeBinary => {
            let offsets_buffer = Buffer::from(buffer_set.primary.freeze());
            let data_buffer = Buffer::from(
                buffer_set
                    .secondary
                    .ok_or_else(|| ArrowError::InvalidArgumentError("LargeBinary requires data buffer".into()))?
                    .freeze(),
            );
            let null_buffer = PooledBufferSet::into_null_buffer(validity, row_count);
            let offsets = OffsetBuffer::new(ScalarBuffer::<i64>::from(offsets_buffer));
            Ok(Arc::new(LargeBinaryArray::new(offsets, data_buffer, null_buffer)))
        }

        // Unsupported types
        _ => Err(ArrowError::NotYetImplemented(format!(
            "pooled_buffer_to_array does not yet support {:?}",
            data_type
        ))),
    }
}

/// Convert multiple pooled buffer sets to a RecordBatch (zero-copy, consumes buffer sets)
///
/// This version uses BytesMut.freeze() for zero-copy conversion to Arrow buffers.
///
/// # Arguments
/// * `schema` - Arrow schema for the batch
/// * `buffer_sets` - One PooledBufferSet per column (consumed)
/// * `row_count` - Number of rows decoded
///
/// # Returns
/// A RecordBatch containing all columns (zero-copy from buffers)
pub fn pooled_buffers_to_batch_owned(
    schema: &SchemaRef,
    buffer_sets: Vec<PooledBufferSet>,
    row_count: usize,
) -> Result<RecordBatch, ArrowError> {
    if buffer_sets.len() != schema.fields().len() {
        return Err(ArrowError::InvalidArgumentError(format!(
            "Expected {} buffer sets for schema, got {}",
            schema.fields().len(),
            buffer_sets.len()
        )));
    }

    let arrays: Result<Vec<ArrayRef>, ArrowError> = schema
        .fields()
        .iter()
        .zip(buffer_sets.into_iter())
        .map(|(field, buffer_set)| pooled_buffer_to_array(field, buffer_set, row_count))
        .collect();

    RecordBatch::try_new(schema.clone(), arrays?)
}

/// Allocate pooled buffer sets for a schema based on JitDecoder metadata
///
/// # Arguments
/// * `decoder` - JitDecoder with schema and buffer slot info
/// * `row_count` - Expected number of rows
/// * `avg_var_bytes` - Average bytes per variable-length value (for sizing)
///
/// # Returns
/// Vector of PooledBufferSets, one per column
pub fn allocate_pooled_buffer_sets(
    decoder: &JitDecoder,
    row_count: usize,
    avg_var_bytes: usize,
) -> Vec<PooledBufferSet> {
    let schema = decoder.schema();
    let mut buffer_sets = Vec::with_capacity(schema.fields().len());

    for (col_idx, field) in schema.fields().iter().enumerate() {
        let slot = decoder.buffer_slot(col_idx).unwrap();
        let nullable = field.is_nullable();
        let validity_bytes = if nullable { (row_count + 7) / 8 } else { 0 };

        let buffer_set = if slot.secondary.is_some() {
            // Variable-length column
            let offsets_size = (row_count + 1) * 4; // i32 offsets
            let data_size = row_count * avg_var_bytes;
            let mut bs = PooledBufferSet::new_variable(offsets_size, data_size);
            if nullable {
                bs = bs.with_validity(validity_bytes);
            }
            bs
        } else {
            // Fixed-size column
            let elem_size = JitDecoder::fixed_size_for_type(field.data_type()).unwrap_or(8);
            let mut bs = PooledBufferSet::new_fixed(row_count * elem_size);
            if nullable {
                bs = bs.with_validity(validity_bytes);
            }
            bs
        };

        buffer_sets.push(buffer_set);
    }

    buffer_sets
}

/// Convert a single column's buffers to an Arrow ArrayRef
///
/// # Arguments
/// * `field` - Arrow field describing the column type
/// * `buffer_set` - The decoded buffer data
/// * `row_count` - Number of rows decoded
///
/// # Returns
/// Arrow ArrayRef wrapping the buffer data (zero-copy for aligned data)
pub fn buffer_to_array(
    field: &Field,
    buffer_set: &BufferSet,
    row_count: usize,
) -> Result<ArrayRef, ArrowError> {
    let data_type = field.data_type();
    let nullable = field.is_nullable();

    // Build validity buffer if nullable
    let null_buffer = if nullable {
        buffer_set.validity.as_ref().map(|v| {
            let buffer = Buffer::from(v.clone());
            NullBuffer::new(BooleanBuffer::new(buffer, 0, row_count))
        })
    } else {
        None
    };

    match data_type {
        // Boolean (1 bit per value, packed)
        DataType::Boolean => {
            let buffer = Buffer::from(buffer_set.primary.clone());
            let bool_buffer = BooleanBuffer::new(buffer, 0, row_count);
            Ok(Arc::new(BooleanArray::new(bool_buffer, null_buffer)))
        }

        // Signed integers
        DataType::Int8 => {
            let buffer = ScalarBuffer::<i8>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(Int8Array::new(buffer, null_buffer)))
        }
        DataType::Int16 => {
            let buffer = ScalarBuffer::<i16>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(Int16Array::new(buffer, null_buffer)))
        }
        DataType::Int32 => {
            let buffer = ScalarBuffer::<i32>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(Int32Array::new(buffer, null_buffer)))
        }
        DataType::Int64 => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(Int64Array::new(buffer, null_buffer)))
        }

        // Unsigned integers
        DataType::UInt8 => {
            let buffer = ScalarBuffer::<u8>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(UInt8Array::new(buffer, null_buffer)))
        }
        DataType::UInt16 => {
            let buffer = ScalarBuffer::<u16>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(UInt16Array::new(buffer, null_buffer)))
        }
        DataType::UInt32 => {
            let buffer = ScalarBuffer::<u32>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(UInt32Array::new(buffer, null_buffer)))
        }
        DataType::UInt64 => {
            let buffer = ScalarBuffer::<u64>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(UInt64Array::new(buffer, null_buffer)))
        }

        // Floating point
        DataType::Float32 => {
            let buffer = ScalarBuffer::<f32>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(Float32Array::new(buffer, null_buffer)))
        }
        DataType::Float64 => {
            let buffer = ScalarBuffer::<f64>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(Float64Array::new(buffer, null_buffer)))
        }

        // Date types
        DataType::Date32 => {
            let buffer = ScalarBuffer::<i32>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(Date32Array::new(buffer, null_buffer)))
        }
        DataType::Date64 => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(Date64Array::new(buffer, null_buffer)))
        }

        // Time types
        DataType::Time32(TimeUnit::Second) => {
            let buffer = ScalarBuffer::<i32>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(Time32SecondArray::new(buffer, null_buffer)))
        }
        DataType::Time32(TimeUnit::Millisecond) => {
            let buffer = ScalarBuffer::<i32>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(Time32MillisecondArray::new(buffer, null_buffer)))
        }
        DataType::Time64(TimeUnit::Microsecond) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(Time64MicrosecondArray::new(buffer, null_buffer)))
        }
        DataType::Time64(TimeUnit::Nanosecond) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(Time64NanosecondArray::new(buffer, null_buffer)))
        }

        // Timestamp types
        DataType::Timestamp(TimeUnit::Second, tz) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(TimestampSecondArray::new(buffer, null_buffer).with_timezone_opt(tz.clone())))
        }
        DataType::Timestamp(TimeUnit::Millisecond, tz) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(TimestampMillisecondArray::new(buffer, null_buffer).with_timezone_opt(tz.clone())))
        }
        DataType::Timestamp(TimeUnit::Microsecond, tz) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(TimestampMicrosecondArray::new(buffer, null_buffer).with_timezone_opt(tz.clone())))
        }
        DataType::Timestamp(TimeUnit::Nanosecond, tz) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(TimestampNanosecondArray::new(buffer, null_buffer).with_timezone_opt(tz.clone())))
        }

        // Duration types
        DataType::Duration(TimeUnit::Second) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(DurationSecondArray::new(buffer, null_buffer)))
        }
        DataType::Duration(TimeUnit::Millisecond) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(DurationMillisecondArray::new(buffer, null_buffer)))
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(DurationMicrosecondArray::new(buffer, null_buffer)))
        }
        DataType::Duration(TimeUnit::Nanosecond) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(DurationNanosecondArray::new(buffer, null_buffer)))
        }

        // Interval types
        DataType::Interval(IntervalUnit::YearMonth) => {
            let buffer = ScalarBuffer::<i32>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(IntervalYearMonthArray::new(buffer, null_buffer)))
        }
        DataType::Interval(IntervalUnit::DayTime) => {
            let buffer = ScalarBuffer::<IntervalDayTime>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(IntervalDayTimeArray::new(buffer, null_buffer)))
        }
        DataType::Interval(IntervalUnit::MonthDayNano) => {
            // MonthDayNano is 16 bytes: i32 months, i32 days, i64 nanoseconds
            let buffer = ScalarBuffer::<IntervalMonthDayNano>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(IntervalMonthDayNanoArray::new(buffer, null_buffer)))
        }

        // Decimal types
        DataType::Decimal128(precision, scale) => {
            let buffer = ScalarBuffer::<i128>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(
                Decimal128Array::new(buffer, null_buffer)
                    .with_precision_and_scale(*precision, *scale)?,
            ))
        }
        DataType::Decimal256(precision, scale) => {
            let buffer = ScalarBuffer::<i256>::from(Buffer::from(buffer_set.primary.clone()));
            Ok(Arc::new(
                Decimal256Array::new(buffer, null_buffer)
                    .with_precision_and_scale(*precision, *scale)?,
            ))
        }

        // Fixed-size binary
        DataType::FixedSizeBinary(size) => {
            let buffer = Buffer::from(buffer_set.primary.clone());
            Ok(Arc::new(FixedSizeBinaryArray::new(*size, buffer, null_buffer)))
        }

        // Variable-length string (UTF-8)
        DataType::Utf8 => {
            let offsets_buffer = Buffer::from(buffer_set.primary.clone());
            let data_buffer = Buffer::from(
                buffer_set
                    .secondary
                    .as_ref()
                    .ok_or_else(|| ArrowError::InvalidArgumentError("Utf8 requires data buffer".into()))?
                    .clone(),
            );

            // Build OffsetBuffer from i32 offsets
            let offsets = OffsetBuffer::new(ScalarBuffer::<i32>::from(offsets_buffer));

            Ok(Arc::new(StringArray::new(offsets, data_buffer, null_buffer)))
        }

        // Large string (i64 offsets)
        DataType::LargeUtf8 => {
            let offsets_buffer = Buffer::from(buffer_set.primary.clone());
            let data_buffer = Buffer::from(
                buffer_set
                    .secondary
                    .as_ref()
                    .ok_or_else(|| ArrowError::InvalidArgumentError("LargeUtf8 requires data buffer".into()))?
                    .clone(),
            );

            let offsets = OffsetBuffer::new(ScalarBuffer::<i64>::from(offsets_buffer));

            Ok(Arc::new(LargeStringArray::new(offsets, data_buffer, null_buffer)))
        }

        // Variable-length binary
        DataType::Binary => {
            let offsets_buffer = Buffer::from(buffer_set.primary.clone());
            let data_buffer = Buffer::from(
                buffer_set
                    .secondary
                    .as_ref()
                    .ok_or_else(|| ArrowError::InvalidArgumentError("Binary requires data buffer".into()))?
                    .clone(),
            );

            let offsets = OffsetBuffer::new(ScalarBuffer::<i32>::from(offsets_buffer));

            Ok(Arc::new(BinaryArray::new(offsets, data_buffer, null_buffer)))
        }

        // Large binary (i64 offsets)
        DataType::LargeBinary => {
            let offsets_buffer = Buffer::from(buffer_set.primary.clone());
            let data_buffer = Buffer::from(
                buffer_set
                    .secondary
                    .as_ref()
                    .ok_or_else(|| ArrowError::InvalidArgumentError("LargeBinary requires data buffer".into()))?
                    .clone(),
            );

            let offsets = OffsetBuffer::new(ScalarBuffer::<i64>::from(offsets_buffer));

            Ok(Arc::new(LargeBinaryArray::new(offsets, data_buffer, null_buffer)))
        }

        // Unsupported types
        _ => Err(ArrowError::NotYetImplemented(format!(
            "buffer_to_array does not yet support {:?}",
            data_type
        ))),
    }
}

/// Convert a single column's buffers to an Arrow ArrayRef (zero-copy, consumes BufferSet)
///
/// This version takes ownership of the BufferSet and converts it to Arrow arrays
/// without copying the underlying data. Use this when you don't need to reuse the buffers.
///
/// # Arguments
/// * `field` - Arrow field describing the column type
/// * `buffer_set` - The decoded buffer data (consumed)
/// * `row_count` - Number of rows decoded
///
/// # Returns
/// Arrow ArrayRef wrapping the buffer data (zero-copy)
pub fn buffer_to_array_owned(
    field: &Field,
    buffer_set: BufferSet,
    row_count: usize,
) -> Result<ArrayRef, ArrowError> {
    let data_type = field.data_type();

    // Extract validity first (before consuming other buffers)
    let validity = buffer_set.validity;

    match data_type {
        // Boolean (1 bit per value, packed)
        DataType::Boolean => {
            let buffer = Buffer::from_vec(buffer_set.primary);
            let bool_buffer = BooleanBuffer::new(buffer, 0, row_count);
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(BooleanArray::new(bool_buffer, null_buffer)))
        }

        // Signed integers
        DataType::Int8 => {
            let buffer = ScalarBuffer::<i8>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(Int8Array::new(buffer, null_buffer)))
        }
        DataType::Int16 => {
            let buffer = ScalarBuffer::<i16>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(Int16Array::new(buffer, null_buffer)))
        }
        DataType::Int32 => {
            let buffer = ScalarBuffer::<i32>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(Int32Array::new(buffer, null_buffer)))
        }
        DataType::Int64 => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(Int64Array::new(buffer, null_buffer)))
        }

        // Unsigned integers
        DataType::UInt8 => {
            let buffer = ScalarBuffer::<u8>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(UInt8Array::new(buffer, null_buffer)))
        }
        DataType::UInt16 => {
            let buffer = ScalarBuffer::<u16>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(UInt16Array::new(buffer, null_buffer)))
        }
        DataType::UInt32 => {
            let buffer = ScalarBuffer::<u32>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(UInt32Array::new(buffer, null_buffer)))
        }
        DataType::UInt64 => {
            let buffer = ScalarBuffer::<u64>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(UInt64Array::new(buffer, null_buffer)))
        }

        // Floating point
        DataType::Float32 => {
            let buffer = ScalarBuffer::<f32>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(Float32Array::new(buffer, null_buffer)))
        }
        DataType::Float64 => {
            let buffer = ScalarBuffer::<f64>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(Float64Array::new(buffer, null_buffer)))
        }

        // Date types
        DataType::Date32 => {
            let buffer = ScalarBuffer::<i32>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(Date32Array::new(buffer, null_buffer)))
        }
        DataType::Date64 => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(Date64Array::new(buffer, null_buffer)))
        }

        // Time types
        DataType::Time32(TimeUnit::Second) => {
            let buffer = ScalarBuffer::<i32>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(Time32SecondArray::new(buffer, null_buffer)))
        }
        DataType::Time32(TimeUnit::Millisecond) => {
            let buffer = ScalarBuffer::<i32>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(Time32MillisecondArray::new(buffer, null_buffer)))
        }
        DataType::Time64(TimeUnit::Microsecond) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(Time64MicrosecondArray::new(buffer, null_buffer)))
        }
        DataType::Time64(TimeUnit::Nanosecond) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(Time64NanosecondArray::new(buffer, null_buffer)))
        }

        // Timestamp types
        DataType::Timestamp(TimeUnit::Second, tz) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(TimestampSecondArray::new(buffer, null_buffer).with_timezone_opt(tz.clone())))
        }
        DataType::Timestamp(TimeUnit::Millisecond, tz) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(TimestampMillisecondArray::new(buffer, null_buffer).with_timezone_opt(tz.clone())))
        }
        DataType::Timestamp(TimeUnit::Microsecond, tz) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(TimestampMicrosecondArray::new(buffer, null_buffer).with_timezone_opt(tz.clone())))
        }
        DataType::Timestamp(TimeUnit::Nanosecond, tz) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(TimestampNanosecondArray::new(buffer, null_buffer).with_timezone_opt(tz.clone())))
        }

        // Duration types
        DataType::Duration(TimeUnit::Second) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(DurationSecondArray::new(buffer, null_buffer)))
        }
        DataType::Duration(TimeUnit::Millisecond) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(DurationMillisecondArray::new(buffer, null_buffer)))
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(DurationMicrosecondArray::new(buffer, null_buffer)))
        }
        DataType::Duration(TimeUnit::Nanosecond) => {
            let buffer = ScalarBuffer::<i64>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(DurationNanosecondArray::new(buffer, null_buffer)))
        }

        // Interval types
        DataType::Interval(IntervalUnit::YearMonth) => {
            let buffer = ScalarBuffer::<i32>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(IntervalYearMonthArray::new(buffer, null_buffer)))
        }
        DataType::Interval(IntervalUnit::DayTime) => {
            let buffer = ScalarBuffer::<IntervalDayTime>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(IntervalDayTimeArray::new(buffer, null_buffer)))
        }
        DataType::Interval(IntervalUnit::MonthDayNano) => {
            let buffer = ScalarBuffer::<IntervalMonthDayNano>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(IntervalMonthDayNanoArray::new(buffer, null_buffer)))
        }

        // Decimal types
        DataType::Decimal128(precision, scale) => {
            let buffer = ScalarBuffer::<i128>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(
                Decimal128Array::new(buffer, null_buffer)
                    .with_precision_and_scale(*precision, *scale)?,
            ))
        }
        DataType::Decimal256(precision, scale) => {
            let buffer = ScalarBuffer::<i256>::from(Buffer::from_vec(buffer_set.primary));
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(
                Decimal256Array::new(buffer, null_buffer)
                    .with_precision_and_scale(*precision, *scale)?,
            ))
        }

        // Fixed-size binary
        DataType::FixedSizeBinary(size) => {
            let buffer = Buffer::from_vec(buffer_set.primary);
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            Ok(Arc::new(FixedSizeBinaryArray::new(*size, buffer, null_buffer)))
        }

        // Variable-length string (UTF-8)
        DataType::Utf8 => {
            let offsets_buffer = Buffer::from_vec(buffer_set.primary);
            let data_buffer = Buffer::from_vec(
                buffer_set
                    .secondary
                    .ok_or_else(|| ArrowError::InvalidArgumentError("Utf8 requires data buffer".into()))?,
            );
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            let offsets = OffsetBuffer::new(ScalarBuffer::<i32>::from(offsets_buffer));
            Ok(Arc::new(StringArray::new(offsets, data_buffer, null_buffer)))
        }

        // Large string (i64 offsets)
        DataType::LargeUtf8 => {
            let offsets_buffer = Buffer::from_vec(buffer_set.primary);
            let data_buffer = Buffer::from_vec(
                buffer_set
                    .secondary
                    .ok_or_else(|| ArrowError::InvalidArgumentError("LargeUtf8 requires data buffer".into()))?,
            );
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            let offsets = OffsetBuffer::new(ScalarBuffer::<i64>::from(offsets_buffer));
            Ok(Arc::new(LargeStringArray::new(offsets, data_buffer, null_buffer)))
        }

        // Variable-length binary
        DataType::Binary => {
            let offsets_buffer = Buffer::from_vec(buffer_set.primary);
            let data_buffer = Buffer::from_vec(
                buffer_set
                    .secondary
                    .ok_or_else(|| ArrowError::InvalidArgumentError("Binary requires data buffer".into()))?,
            );
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            let offsets = OffsetBuffer::new(ScalarBuffer::<i32>::from(offsets_buffer));
            Ok(Arc::new(BinaryArray::new(offsets, data_buffer, null_buffer)))
        }

        // Large binary (i64 offsets)
        DataType::LargeBinary => {
            let offsets_buffer = Buffer::from_vec(buffer_set.primary);
            let data_buffer = Buffer::from_vec(
                buffer_set
                    .secondary
                    .ok_or_else(|| ArrowError::InvalidArgumentError("LargeBinary requires data buffer".into()))?,
            );
            let null_buffer = BufferSet::into_null_buffer(validity, row_count);
            let offsets = OffsetBuffer::new(ScalarBuffer::<i64>::from(offsets_buffer));
            Ok(Arc::new(LargeBinaryArray::new(offsets, data_buffer, null_buffer)))
        }

        // Unsupported types
        _ => Err(ArrowError::NotYetImplemented(format!(
            "buffer_to_array_owned does not yet support {:?}",
            data_type
        ))),
    }
}

/// Convert multiple buffer sets to a RecordBatch
///
/// # Arguments
/// * `schema` - Arrow schema for the batch
/// * `buffer_sets` - One BufferSet per column
/// * `row_count` - Number of rows decoded
///
/// # Returns
/// A RecordBatch containing all columns
pub fn buffers_to_batch(
    schema: &SchemaRef,
    buffer_sets: &[BufferSet],
    row_count: usize,
) -> Result<RecordBatch, ArrowError> {
    if buffer_sets.len() != schema.fields().len() {
        return Err(ArrowError::InvalidArgumentError(format!(
            "Expected {} buffer sets for schema, got {}",
            schema.fields().len(),
            buffer_sets.len()
        )));
    }

    let arrays: Result<Vec<ArrayRef>, ArrowError> = schema
        .fields()
        .iter()
        .zip(buffer_sets.iter())
        .map(|(field, buffer_set)| buffer_to_array(field, buffer_set, row_count))
        .collect();

    RecordBatch::try_new(schema.clone(), arrays?)
}

/// Convert multiple buffer sets to a RecordBatch (zero-copy, consumes buffer sets)
///
/// This version takes ownership of the buffer sets and converts them to Arrow arrays
/// without copying the underlying data. Use this when you don't need to reuse the buffers.
///
/// # Arguments
/// * `schema` - Arrow schema for the batch
/// * `buffer_sets` - One BufferSet per column (consumed)
/// * `row_count` - Number of rows decoded
///
/// # Returns
/// A RecordBatch containing all columns (zero-copy from buffers)
pub fn buffers_to_batch_owned(
    schema: &SchemaRef,
    buffer_sets: Vec<BufferSet>,
    row_count: usize,
) -> Result<RecordBatch, ArrowError> {
    if buffer_sets.len() != schema.fields().len() {
        return Err(ArrowError::InvalidArgumentError(format!(
            "Expected {} buffer sets for schema, got {}",
            schema.fields().len(),
            buffer_sets.len()
        )));
    }

    let arrays: Result<Vec<ArrayRef>, ArrowError> = schema
        .fields()
        .iter()
        .zip(buffer_sets.into_iter())
        .map(|(field, buffer_set)| buffer_to_array_owned(field, buffer_set, row_count))
        .collect();

    RecordBatch::try_new(schema.clone(), arrays?)
}

/// Allocate buffer sets for a schema based on JitDecoder metadata
///
/// # Arguments
/// * `decoder` - JitDecoder with schema and buffer slot info
/// * `row_count` - Expected number of rows
/// * `avg_var_bytes` - Average bytes per variable-length value (for sizing)
///
/// # Returns
/// Vector of BufferSets, one per column
pub fn allocate_buffer_sets(
    decoder: &JitDecoder,
    row_count: usize,
    avg_var_bytes: usize,
) -> Vec<BufferSet> {
    let schema = decoder.schema();
    let mut buffer_sets = Vec::with_capacity(schema.fields().len());

    for (col_idx, field) in schema.fields().iter().enumerate() {
        let slot = decoder.buffer_slot(col_idx).unwrap();
        let nullable = field.is_nullable();
        let validity_bytes = if nullable { (row_count + 7) / 8 } else { 0 };

        let buffer_set = if slot.secondary.is_some() {
            // Variable-length column
            let offsets_size = (row_count + 1) * 4; // i32 offsets
            let data_size = row_count * avg_var_bytes;
            let mut bs = BufferSet::new_variable(offsets_size, data_size);
            if nullable {
                bs = bs.with_validity(validity_bytes);
            }
            bs
        } else {
            // Fixed-size column
            let elem_size = JitDecoder::fixed_size_for_type(field.data_type()).unwrap_or(8);
            let mut bs = BufferSet::new_fixed(row_count * elem_size);
            if nullable {
                bs = bs.with_validity(validity_bytes);
            }
            bs
        };

        buffer_sets.push(buffer_set);
    }

    buffer_sets
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Array;
    use arrow_schema::{Field, Schema};

    #[test]
    fn test_buffer_to_array_int64() {
        let field = Field::new("id", DataType::Int64, false);
        let mut buffer_set = BufferSet::new_fixed(24); // 3 values * 8 bytes

        // Write 3 i64 values
        let values: [i64; 3] = [1, 2, 3];
        for (i, v) in values.iter().enumerate() {
            buffer_set.primary[i * 8..(i + 1) * 8].copy_from_slice(&v.to_le_bytes());
        }

        let array = buffer_to_array(&field, &buffer_set, 3).unwrap();
        let int_array = array.as_any().downcast_ref::<Int64Array>().unwrap();

        assert_eq!(int_array.len(), 3);
        assert_eq!(int_array.value(0), 1);
        assert_eq!(int_array.value(1), 2);
        assert_eq!(int_array.value(2), 3);
    }

    #[test]
    fn test_buffer_to_array_string() {
        let field = Field::new("name", DataType::Utf8, false);
        let mut buffer_set = BufferSet::new_variable(16, 64); // 4 offsets, 64 bytes data

        // Write offsets: [0, 5, 10, 15]
        let offsets: [i32; 4] = [0, 5, 10, 15];
        for (i, off) in offsets.iter().enumerate() {
            buffer_set.primary[i * 4..(i + 1) * 4].copy_from_slice(&off.to_le_bytes());
        }

        // Write data: "helloworld!!!!!" (15 bytes for offsets [0,5,10,15])
        let data = b"helloworld!!!!!";
        buffer_set.secondary.as_mut().unwrap()[..data.len()].copy_from_slice(data);

        let array = buffer_to_array(&field, &buffer_set, 3).unwrap();
        let str_array = array.as_any().downcast_ref::<StringArray>().unwrap();

        assert_eq!(str_array.len(), 3);
        assert_eq!(str_array.value(0), "hello");
        assert_eq!(str_array.value(1), "world");
        assert_eq!(str_array.value(2), "!!!!!");
    }

    #[test]
    fn test_buffers_to_batch() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Float64, false),
        ]));

        let mut buffer_sets = vec![
            BufferSet::new_fixed(16), // 2 i64
            BufferSet::new_fixed(16), // 2 f64
        ];

        // Write values
        buffer_sets[0].primary[0..8].copy_from_slice(&1i64.to_le_bytes());
        buffer_sets[0].primary[8..16].copy_from_slice(&2i64.to_le_bytes());
        buffer_sets[1].primary[0..8].copy_from_slice(&1.5f64.to_le_bytes());
        buffer_sets[1].primary[8..16].copy_from_slice(&2.5f64.to_le_bytes());

        let batch = buffers_to_batch(&schema, &buffer_sets, 2).unwrap();

        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 2);
    }

    // PooledBufferSet tests

    #[test]
    fn test_pooled_buffer_set_creation() {
        // Test from_pool constructor
        let primary = BytesMut::zeroed(24);
        let pbs = PooledBufferSet::from_pool(primary, None, None);
        assert_eq!(pbs.primary.len(), 24);
        assert!(pbs.secondary.is_none());
        assert!(pbs.validity.is_none());
        assert_eq!(pbs.primary_len, 0);
        assert_eq!(pbs.secondary_len, 0);
    }

    #[test]
    fn test_pooled_buffer_set_new_fixed() {
        let pbs = PooledBufferSet::new_fixed(100);
        assert_eq!(pbs.primary_capacity(), 100);
        assert!(pbs.secondary.is_none());
    }

    #[test]
    fn test_pooled_buffer_set_new_variable() {
        let pbs = PooledBufferSet::new_variable(16, 64);
        assert_eq!(pbs.primary_capacity(), 16);
        assert_eq!(pbs.secondary_capacity(), 64);
    }

    #[test]
    fn test_pooled_buffer_set_with_validity() {
        let pbs = PooledBufferSet::new_fixed(100).with_validity(13); // 100 rows / 8 bits
        assert!(pbs.validity.is_some());
        assert_eq!(pbs.validity.as_ref().unwrap().len(), 13);
    }

    #[test]
    fn test_pooled_buffer_set_pointers() {
        let mut pbs = PooledBufferSet::new_variable(16, 64).with_validity(4);

        // Get pointers and verify they're valid
        let p_ptr = pbs.primary_ptr();
        let s_ptr = pbs.secondary_ptr();
        let v_ptr = pbs.validity_ptr();

        assert!(!p_ptr.is_null());
        assert!(!s_ptr.is_null());
        assert!(!v_ptr.is_null());

        // Write through pointers
        unsafe {
            *p_ptr = 42;
            *s_ptr = 99;
            *v_ptr = 0xFF;
        }

        assert_eq!(pbs.primary[0], 42);
        assert_eq!(pbs.secondary.as_ref().unwrap()[0], 99);
        assert_eq!(pbs.validity.as_ref().unwrap()[0], 0xFF);
    }

    #[test]
    fn test_pooled_buffer_to_array_int64() {
        let field = Field::new("id", DataType::Int64, false);
        let mut buffer_set = PooledBufferSet::new_fixed(24); // 3 values * 8 bytes

        // Write 3 i64 values
        let values: [i64; 3] = [100, 200, 300];
        for (i, v) in values.iter().enumerate() {
            buffer_set.primary[i * 8..(i + 1) * 8].copy_from_slice(&v.to_le_bytes());
        }

        let array = pooled_buffer_to_array(&field, buffer_set, 3).unwrap();
        let int_array = array.as_any().downcast_ref::<Int64Array>().unwrap();

        assert_eq!(int_array.len(), 3);
        assert_eq!(int_array.value(0), 100);
        assert_eq!(int_array.value(1), 200);
        assert_eq!(int_array.value(2), 300);
    }

    #[test]
    fn test_pooled_buffer_to_array_string() {
        let field = Field::new("name", DataType::Utf8, false);
        let mut buffer_set = PooledBufferSet::new_variable(16, 64); // 4 offsets, 64 bytes data

        // Write offsets: [0, 4, 9, 14]
        let offsets: [i32; 4] = [0, 4, 9, 14];
        for (i, off) in offsets.iter().enumerate() {
            buffer_set.primary[i * 4..(i + 1) * 4].copy_from_slice(&off.to_le_bytes());
        }

        // Write data: "testpooledarray"
        let data = b"testpooledarray";
        buffer_set.secondary.as_mut().unwrap()[..data.len()].copy_from_slice(data);

        let array = pooled_buffer_to_array(&field, buffer_set, 3).unwrap();
        let str_array = array.as_any().downcast_ref::<StringArray>().unwrap();

        assert_eq!(str_array.len(), 3);
        assert_eq!(str_array.value(0), "test");
        assert_eq!(str_array.value(1), "poole");
        assert_eq!(str_array.value(2), "darra");
    }

    #[test]
    fn test_pooled_buffers_to_batch_owned() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("score", DataType::Float64, false),
        ]));

        let mut id_buf = PooledBufferSet::new_fixed(24); // 3 i64
        let mut score_buf = PooledBufferSet::new_fixed(24); // 3 f64

        // Write id values
        id_buf.primary[0..8].copy_from_slice(&10i64.to_le_bytes());
        id_buf.primary[8..16].copy_from_slice(&20i64.to_le_bytes());
        id_buf.primary[16..24].copy_from_slice(&30i64.to_le_bytes());

        // Write score values
        score_buf.primary[0..8].copy_from_slice(&1.1f64.to_le_bytes());
        score_buf.primary[8..16].copy_from_slice(&2.2f64.to_le_bytes());
        score_buf.primary[16..24].copy_from_slice(&3.3f64.to_le_bytes());

        let buffer_sets = vec![id_buf, score_buf];
        let batch = pooled_buffers_to_batch_owned(&schema, buffer_sets, 3).unwrap();

        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 2);

        let id_array = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(id_array.value(0), 10);
        assert_eq!(id_array.value(1), 20);
        assert_eq!(id_array.value(2), 30);

        let score_array = batch.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert!((score_array.value(0) - 1.1).abs() < f64::EPSILON);
        assert!((score_array.value(1) - 2.2).abs() < f64::EPSILON);
        assert!((score_array.value(2) - 3.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pooled_buffer_set_zero_copy_check() {
        // Verify that freeze() + Buffer::from() doesn't copy
        let mut pbs = PooledBufferSet::new_fixed(8);
        pbs.primary[0..8].copy_from_slice(&42i64.to_le_bytes());

        // Get pointer before freeze
        let ptr_before = pbs.primary.as_ptr();

        // Freeze and convert to Arrow buffer
        let frozen = pbs.primary.freeze();
        let arrow_buffer = Buffer::from(frozen);

        // The underlying pointer should be the same (zero-copy)
        let ptr_after = arrow_buffer.as_ptr();
        assert_eq!(ptr_before, ptr_after, "Buffer should be zero-copy");
    }
}

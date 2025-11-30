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

use crate::buffer_provider::BufferProvider;
use crate::codec::{AvroDataType, Codec};
use super::cursor::AvroCursor;
use arrow_array::*;
use arrow_buffer::{i256, Buffer, IntervalMonthDayNano, OffsetBuffer, ScalarBuffer};
use arrow_schema::{ArrowError, DataType, Field, FieldRef, Fields, IntervalUnit, Schema, SchemaRef, UnionFields, UnionMode};
use std::cmp::Ordering;
use std::sync::Arc;

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

/// Nested type state for complex types (Map, List, Union, Struct, Dictionary)
#[derive(Debug)]
enum NestedState {
    /// No nested state (primitive types)
    None,
    /// Map type: keys (strings) and values with their nested state
    Map {
        /// Map offsets (one per row, tracking number of entries)
        map_offsets: Vec<i32>,
        /// Key offsets for string keys
        key_offsets: Vec<i32>,
        /// Key data bytes
        key_data: Vec<u8>,
        /// Value data - stored as raw bytes for primitive values
        value_data: Vec<u8>,
        /// Value type
        value_type: DataType,
        /// Current total entries
        total_entries: usize,
    },
    /// Union type: type IDs and child data
    Union {
        /// Type IDs for each row
        type_ids: Vec<i8>,
        /// Offsets for dense union (optional)
        offsets: Option<Vec<i32>>,
        /// Child data per variant (index by type_id)
        child_data: Vec<Vec<u8>>,
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
        /// Child element data
        child_data: Vec<u8>,
        /// Child element type
        child_type: DataType,
        /// Total elements
        total_elements: usize,
    },
    /// Dictionary type: indices and dictionary values (for Avro enum)
    Dictionary {
        /// Indices (varint encoded enum values become i32 indices)
        indices: Vec<i32>,
        /// Dictionary values (enum symbols from schema)
        dictionary: Vec<String>,
    },
}

/// Column state for pooled decoding
#[derive(Debug)]
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
    /// Nested type state for complex types
    nested: NestedState,
}

impl ColumnState {
    fn new(column_index: usize, field: &Field) -> Self {
        let nested = Self::create_nested_state(field.data_type());
        Self {
            column_index,
            data_type: field.data_type().clone(),
            data_position: 0,
            row_count: 0,
            is_nullable: field.is_nullable(),
            nested,
        }
    }

    fn create_nested_state(data_type: &DataType) -> NestedState {
        match data_type {
            DataType::Map(entries_field, _) => {
                // Extract value type from map entries struct
                let value_type = if let DataType::Struct(fields) = entries_field.data_type() {
                    fields.iter()
                        .find(|f| f.name() == "value")
                        .map(|f| f.data_type().clone())
                        .unwrap_or(DataType::Utf8)
                } else {
                    DataType::Utf8
                };
                NestedState::Map {
                    map_offsets: vec![0],
                    key_offsets: vec![0],
                    key_data: Vec::new(),
                    value_data: Vec::new(),
                    value_type,
                    total_entries: 0,
                }
            }
            DataType::Union(fields, mode) => {
                let num_variants = fields.len();
                NestedState::Union {
                    type_ids: Vec::new(),
                    offsets: if *mode == UnionMode::Dense { Some(Vec::new()) } else { None },
                    child_data: (0..num_variants).map(|_| Vec::new()).collect(),
                    child_counts: vec![0; num_variants],
                    fields: fields.clone(),
                    mode: *mode,
                }
            }
            DataType::List(child_field) => {
                NestedState::List {
                    offsets: vec![0],
                    child_data: Vec::new(),
                    child_type: child_field.data_type().clone(),
                    total_elements: 0,
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
            _ => NestedState::None,
        }
    }

    fn reset(&mut self) {
        self.data_position = 0;
        self.row_count = 0;
        // Reset nested state
        match &mut self.nested {
            NestedState::None => {}
            NestedState::Map { map_offsets, key_offsets, key_data, value_data, total_entries, .. } => {
                map_offsets.clear();
                map_offsets.push(0);
                key_offsets.clear();
                key_offsets.push(0);
                key_data.clear();
                value_data.clear();
                *total_entries = 0;
            }
            NestedState::Union { type_ids, offsets, child_data, child_counts, .. } => {
                type_ids.clear();
                if let Some(off) = offsets {
                    off.clear();
                }
                for data in child_data.iter_mut() {
                    data.clear();
                }
                for count in child_counts.iter_mut() {
                    *count = 0;
                }
            }
            NestedState::List { offsets, child_data, total_elements, .. } => {
                offsets.clear();
                offsets.push(0);
                child_data.clear();
                *total_elements = 0;
            }
            NestedState::Dictionary { indices, dictionary: _ } => {
                // Keep dictionary values (they come from schema), clear indices
                indices.clear();
            }
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
            .map(|(idx, field)| ColumnState::new(idx, field.as_ref()))
            .collect();

        Ok(Self {
            schema,
            columns,
            provider,
            row_count: 0,
            max_rows,
        })
    }

    /// Create from an AvroDataType (for compatibility with existing API)
    pub fn try_from_avro(data_type: &AvroDataType, provider: P) -> Result<Self, ArrowError> {
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

    /// Decode a single row from the cursor
    pub fn decode_row(&mut self, cursor: &mut AvroCursor<'_>) -> Result<(), ArrowError> {
        if self.row_count >= self.max_rows {
            return Err(ArrowError::InvalidArgumentError(
                "Decoder at capacity".to_string(),
            ));
        }

        let num_columns = self.columns.len();
        for col_idx in 0..num_columns {
            self.decode_column_value(col_idx, cursor)?;
        }

        self.row_count += 1;
        Ok(())
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

    /// Decode a Map value from Avro block encoding
    /// Avro maps: { key: string, value: T } with block encoding
    fn decode_map_value(
        &mut self,
        col_idx: usize,
        cursor: &mut AvroCursor<'_>,
    ) -> Result<(), ArrowError> {
        let col = &mut self.columns[col_idx];

        if let NestedState::Map {
            map_offsets, key_offsets, key_data, value_data, value_type, total_entries
        } = &mut col.nested {
            let value_type_clone = value_type.clone();
            let mut entries_in_row = 0usize;

            // Read blocks until we get a 0 count
            loop {
                let block_count = cursor.get_long()?;
                match block_count.cmp(&0) {
                    Ordering::Equal => break,
                    Ordering::Less => {
                        let count = (-block_count) as usize;
                        let _size_in_bytes = cursor.get_long()?; // Skip size hint
                        for _ in 0..count {
                            // Read key (string)
                            let key_bytes = cursor.get_bytes()?;
                            key_data.extend_from_slice(key_bytes);
                            let new_offset = key_data.len() as i32;
                            key_offsets.push(new_offset);

                            // Read value based on type
                            Self::decode_value_to_vec(&value_type_clone, cursor, value_data)?;
                        }
                        entries_in_row += count;
                    }
                    Ordering::Greater => {
                        let count = block_count as usize;
                        for _ in 0..count {
                            // Read key (string)
                            let key_bytes = cursor.get_bytes()?;
                            key_data.extend_from_slice(key_bytes);
                            let new_offset = key_data.len() as i32;
                            key_offsets.push(new_offset);

                            // Read value based on type
                            Self::decode_value_to_vec(&value_type_clone, cursor, value_data)?;
                        }
                        entries_in_row += count;
                    }
                }
            }

            *total_entries += entries_in_row;
            let new_map_offset = *total_entries as i32;
            map_offsets.push(new_map_offset);
            col.row_count += 1;
            Ok(())
        } else {
            Err(ArrowError::InvalidArgumentError(
                "Map column missing nested state".to_string()
            ))
        }
    }

    /// Decode a Union value (type tag + value)
    fn decode_union_value(
        &mut self,
        col_idx: usize,
        cursor: &mut AvroCursor<'_>,
    ) -> Result<(), ArrowError> {
        let col = &mut self.columns[col_idx];

        if let NestedState::Union {
            type_ids, offsets, child_data, child_counts, fields, mode: _
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

            // Decode the value into the appropriate child data vector
            let child_type = field.data_type().clone();
            Self::decode_value_to_vec(&child_type, cursor, &mut child_data[type_index])?;
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
        let col = &mut self.columns[col_idx];

        if let NestedState::List { offsets, child_data, child_type, total_elements } = &mut col.nested {
            let child_type_clone = child_type.clone();
            let mut elements_in_row = 0usize;

            // Read blocks until we get a 0 count
            loop {
                let block_count = cursor.get_long()?;
                match block_count.cmp(&0) {
                    Ordering::Equal => break,
                    Ordering::Less => {
                        let count = (-block_count) as usize;
                        let _size_in_bytes = cursor.get_long()?;
                        for _ in 0..count {
                            Self::decode_value_to_vec(&child_type_clone, cursor, child_data)?;
                        }
                        elements_in_row += count;
                    }
                    Ordering::Greater => {
                        let count = block_count as usize;
                        for _ in 0..count {
                            Self::decode_value_to_vec(&child_type_clone, cursor, child_data)?;
                        }
                        elements_in_row += count;
                    }
                }
            }

            *total_elements += elements_in_row;
            let new_offset = *total_elements as i32;
            offsets.push(new_offset);
            col.row_count += 1;
            Ok(())
        } else {
            Err(ArrowError::InvalidArgumentError(
                "List column missing nested state".to_string()
            ))
        }
    }

    /// Helper to decode a primitive value directly into a Vec<u8>
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
            other => {
                return Err(ArrowError::NotYetImplemented(format!(
                    "Nested value decoding not implemented for {:?}",
                    other
                )));
            }
        }
        Ok(())
    }

    /// Decode multiple rows from a buffer
    pub fn decode(&mut self, buf: &[u8], count: usize) -> Result<usize, ArrowError> {
        let mut cursor = AvroCursor::new(buf);
        let rows_to_decode = count.min(self.max_rows - self.row_count);

        for _ in 0..rows_to_decode {
            self.decode_row(&mut cursor)?;
        }

        Ok(cursor.position())
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

        match &data_type {
            DataType::Null => {
                Ok(Arc::new(NullArray::new(row_count)))
            }
            DataType::Int32 => {
                let buffer = self.provider.finalize_data_buffer(column_index, row_count * 4)?;
                let values = ScalarBuffer::<i32>::from(buffer);
                Ok(Arc::new(Int32Array::new(values, None)))
            }
            DataType::Int64 => {
                let buffer = self.provider.finalize_data_buffer(column_index, row_count * 8)?;
                let values = ScalarBuffer::<i64>::from(buffer);
                Ok(Arc::new(Int64Array::new(values, None)))
            }
            DataType::Float32 => {
                let buffer = self.provider.finalize_data_buffer(column_index, row_count * 4)?;
                let values = ScalarBuffer::<f32>::from(buffer);
                Ok(Arc::new(Float32Array::new(values, None)))
            }
            DataType::Float64 => {
                let buffer = self.provider.finalize_data_buffer(column_index, row_count * 8)?;
                let values = ScalarBuffer::<f64>::from(buffer);
                Ok(Arc::new(Float64Array::new(values, None)))
            }
            DataType::Date32 => {
                let buffer = self.provider.finalize_data_buffer(column_index, row_count * 4)?;
                let values = ScalarBuffer::<i32>::from(buffer);
                Ok(Arc::new(Date32Array::new(values, None)))
            }
            DataType::Time32(unit) => {
                let buffer = self.provider.finalize_data_buffer(column_index, row_count * 4)?;
                let values = ScalarBuffer::<i32>::from(buffer);
                let array: ArrayRef = match unit {
                    arrow_schema::TimeUnit::Second => {
                        Arc::new(Time32SecondArray::new(values, None))
                    }
                    arrow_schema::TimeUnit::Millisecond => {
                        Arc::new(Time32MillisecondArray::new(values, None))
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
                        Arc::new(Time64MicrosecondArray::new(values, None))
                    }
                    arrow_schema::TimeUnit::Nanosecond => {
                        Arc::new(Time64NanosecondArray::new(values, None))
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
                        Arc::new(TimestampSecondArray::new(values, None).with_timezone_opt(tz.clone()))
                    }
                    arrow_schema::TimeUnit::Millisecond => {
                        Arc::new(TimestampMillisecondArray::new(values, None).with_timezone_opt(tz.clone()))
                    }
                    arrow_schema::TimeUnit::Microsecond => {
                        Arc::new(TimestampMicrosecondArray::new(values, None).with_timezone_opt(tz.clone()))
                    }
                    arrow_schema::TimeUnit::Nanosecond => {
                        Arc::new(TimestampNanosecondArray::new(values, None).with_timezone_opt(tz.clone()))
                    }
                };
                Ok(array)
            }
            DataType::Boolean => {
                let bitmap_bytes = (row_count + 7) / 8;
                let buffer = self.provider.finalize_data_buffer(column_index, bitmap_bytes)?;
                let bool_buffer = arrow_buffer::BooleanBuffer::new(buffer, 0, row_count);
                Ok(Arc::new(BooleanArray::new(bool_buffer, None)))
            }
            DataType::Utf8 => {
                let data_buffer = self.provider.finalize_data_buffer(column_index, data_position)?;
                let offsets_buffer = self.provider.finalize_offsets_buffer(column_index, (row_count + 1) * 4)?;
                let offsets = OffsetBuffer::new(ScalarBuffer::<i32>::from(offsets_buffer));
                Ok(Arc::new(StringArray::try_new(offsets, data_buffer, None)?))
            }
            DataType::Binary => {
                let data_buffer = self.provider.finalize_data_buffer(column_index, data_position)?;
                let offsets_buffer = self.provider.finalize_offsets_buffer(column_index, (row_count + 1) * 4)?;
                let offsets = OffsetBuffer::new(ScalarBuffer::<i32>::from(offsets_buffer));
                Ok(Arc::new(BinaryArray::try_new(offsets, data_buffer, None)?))
            }
            DataType::FixedSizeBinary(size) => {
                let buffer = self.provider.finalize_data_buffer(column_index, row_count * (*size as usize))?;
                Ok(Arc::new(FixedSizeBinaryArray::try_new(*size, buffer, None)?))
            }
            DataType::Decimal128(precision, scale) => {
                let buffer = self.provider.finalize_data_buffer(column_index, row_count * 16)?;
                let values = ScalarBuffer::<i128>::from(buffer);
                Ok(Arc::new(Decimal128Array::new(values, None).with_precision_and_scale(*precision, *scale)?))
            }
            DataType::Decimal256(precision, scale) => {
                let buffer = self.provider.finalize_data_buffer(column_index, row_count * 32)?;
                let values = ScalarBuffer::<i256>::from(buffer);
                Ok(Arc::new(Decimal256Array::new(values, None).with_precision_and_scale(*precision, *scale)?))
            }
            DataType::Interval(IntervalUnit::MonthDayNano) => {
                let buffer = self.provider.finalize_data_buffer(column_index, row_count * 16)?;
                let values = ScalarBuffer::<IntervalMonthDayNano>::from(buffer);
                Ok(Arc::new(IntervalMonthDayNanoArray::new(values, None)))
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

    /// Flush a Map column from nested state to Arrow MapArray
    fn flush_map_column(
        &mut self,
        col_idx: usize,
        entries_field: FieldRef,
        keys_sorted: bool,
    ) -> Result<ArrayRef, ArrowError> {
        let col = &mut self.columns[col_idx];

        if let NestedState::Map {
            map_offsets, key_offsets, key_data, value_data, value_type, ..
        } = &mut col.nested {

            // Create key array (StringArray)
            let key_offsets_buffer = Buffer::from_slice_ref(key_offsets);
            let key_data_buffer = Buffer::from_slice_ref(key_data);
            let key_array = StringArray::try_new(
                OffsetBuffer::new(ScalarBuffer::<i32>::from(key_offsets_buffer)),
                key_data_buffer,
                None,
            )?;

            // Create value array based on type
            let value_array = Self::build_array_from_vec(value_type.clone(), value_data)?;

            // Create entries struct array
            let entries_struct = StructArray::new(
                Fields::from(vec![
                    Arc::new(Field::new("key", DataType::Utf8, false)),
                    Arc::new(Field::new("value", value_type.clone(), true)),
                ]),
                vec![Arc::new(key_array), value_array],
                None,
            );

            // Create map offsets
            let map_offsets_buffer = Buffer::from_slice_ref(map_offsets);
            let map_offsets = OffsetBuffer::new(ScalarBuffer::<i32>::from(map_offsets_buffer));

            // Create MapArray
            let map_array = MapArray::new(
                entries_field,
                map_offsets,
                entries_struct,
                None,
                keys_sorted,
            );

            Ok(Arc::new(map_array))
        } else {
            Err(ArrowError::InvalidArgumentError(
                "Map column missing nested state".to_string()
            ))
        }
    }

    /// Flush a Union column from nested state to Arrow UnionArray
    fn flush_union_column(
        &mut self,
        col_idx: usize,
        fields: UnionFields,
        mode: UnionMode,
    ) -> Result<ArrayRef, ArrowError> {
        let col = &mut self.columns[col_idx];

        if let NestedState::Union {
            type_ids, offsets, child_data, child_counts, fields: _nested_fields, ..
        } = &mut col.nested {
            // Build type_ids buffer
            let type_ids_buffer = ScalarBuffer::<i8>::from(Buffer::from_slice_ref(type_ids));

            // Build child arrays - use child_counts for Null types since they have no data
            let mut children: Vec<ArrayRef> = Vec::with_capacity(fields.len());
            for (i, (_, field)) in fields.iter().enumerate() {
                let child_type = field.data_type().clone();
                match child_type {
                    DataType::Null => {
                        // For Null type, create NullArray with the correct count
                        children.push(Arc::new(NullArray::new(child_counts[i])));
                    }
                    _ => {
                        let child_array = Self::build_array_from_vec(child_type, &mut child_data[i])?;
                        children.push(child_array);
                    }
                }
            }

            // Create union array based on mode
            let union_array = match mode {
                UnionMode::Dense => {
                    let offsets_buffer = offsets.as_ref()
                        .map(|o| ScalarBuffer::<i32>::from(Buffer::from_slice_ref(o)));
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
        let col = &mut self.columns[col_idx];

        if let NestedState::List { offsets, child_data, child_type, .. } = &mut col.nested {
            // Build child array
            let child_array = Self::build_array_from_vec(child_type.clone(), child_data)?;

            // Create offsets
            let offsets_buffer = Buffer::from_slice_ref(offsets);
            let list_offsets = OffsetBuffer::new(ScalarBuffer::<i32>::from(offsets_buffer));

            // Create ListArray
            let list_array = ListArray::new(child_field, list_offsets, child_array, None);

            Ok(Arc::new(list_array))
        } else {
            Err(ArrowError::InvalidArgumentError(
                "List column missing nested state".to_string()
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
            other => Err(ArrowError::NotYetImplemented(format!(
                "build_array_from_vec not implemented for {:?}",
                other
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer_provider::{BufferLayout, ColumnRegion, PooledBufferProvider};

    fn create_int64_layout(row_capacity: usize) -> BufferLayout {
        BufferLayout {
            columns: vec![ColumnRegion {
                data_offset: 0,
                data_capacity: row_capacity * 8,
                offsets_offset: None,
                offsets_capacity: None,
                null_bitmap_offset: None,
                null_bitmap_capacity: None,
            }],
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
                ColumnRegion {
                    data_offset: 0,
                    data_capacity: 1024,
                    offsets_offset: None,
                    offsets_capacity: None,
                    null_bitmap_offset: None,
                    null_bitmap_capacity: None,
                },
                ColumnRegion {
                    data_offset: 1024,
                    data_capacity: 1024,
                    offsets_offset: None,
                    offsets_capacity: None,
                    null_bitmap_offset: None,
                    null_bitmap_capacity: None,
                },
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
}

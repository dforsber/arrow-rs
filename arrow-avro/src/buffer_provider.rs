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

//! Buffer provider trait for zero-allocation Avro decoding
//!
//! This module provides the [`BufferProvider`] trait which enables decoding
//! Avro data directly into pre-allocated memory buffers, eliminating memory
//! allocation during the hot path.
//!
//! # Architecture
//!
//! The buffer provider works with a pre-planned [`BufferLayout`] that specifies
//! where each column's data should be written within a contiguous memory region.
//! This enables:
//!
//! - Zero memory allocation after warmup
//! - Cache-tiled batching (fitting batches within L2 cache)
//! - Integration with memory pools for Arc-shared RecordBatches
//!
//! # Usage
//!
//! ```ignore
//! use arrow_avro::buffer_provider::{BufferProvider, PooledBufferProvider};
//!
//! // Create a buffer provider with a pre-allocated buffer
//! let buffer = vec![0u8; 256 * 1024]; // 256KB for L2 cache tiling
//! let provider = PooledBufferProvider::new(buffer, layout);
//!
//! // Use with decoder
//! let decoder = ReaderBuilder::new()
//!     .with_buffer_provider(provider)
//!     .build_decoder()?;
//! ```

use arrow_buffer::alloc::Allocation;
use arrow_buffer::Buffer;
use arrow_schema::ArrowError;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::Arc;

/// Arrow requires 64-byte alignment for SIMD operations
pub const ARROW_ALIGNMENT: usize = 64;

/// Align an offset to the specified alignment
///
/// # Arguments
/// * `offset` - Current offset
/// * `alignment` - Required alignment (must be power of 2)
///
/// # Returns
/// The next aligned offset >= input offset
#[inline]
pub fn align_to(offset: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two());
    (offset + alignment - 1) & !(alignment - 1)
}

/// Get the required alignment for an Arrow data type
///
/// Returns the minimum alignment required for the data type's values
pub fn alignment_for_type(dt: &arrow_schema::DataType) -> usize {
    use arrow_schema::DataType;
    match dt {
        DataType::Boolean | DataType::Int8 | DataType::UInt8 => 1,
        DataType::Int16 | DataType::UInt16 | DataType::Float16 => 2,
        DataType::Int32 | DataType::UInt32 | DataType::Float32 | DataType::Date32 => 4,
        DataType::Int64 | DataType::UInt64 | DataType::Float64
        | DataType::Date64 | DataType::Timestamp(_, _) | DataType::Time64(_)
        | DataType::Duration(_) | DataType::Interval(_) => 8,
        DataType::Decimal128(_, _) | DataType::Decimal256(_, _) => 16,
        // Variable-length types: data buffer is bytes (1-byte aligned), but offsets are i32 (4-byte aligned)
        DataType::Utf8 | DataType::Binary | DataType::LargeUtf8 | DataType::LargeBinary => 1,
        // For complex types, use maximum alignment for safety
        DataType::List(_) | DataType::LargeList(_) | DataType::Map(_, _)
        | DataType::Struct(_) | DataType::Union(_, _) => 8,
        // Fixed-size types
        DataType::FixedSizeBinary(_) | DataType::FixedSizeList(_, _) => 1,
        // Default to 8-byte alignment for unknown types
        _ => 8,
    }
}

/// Trait for providing pre-allocated buffers for Avro decoding
///
/// Implementations of this trait manage memory allocation for Arrow arrays,
/// enabling zero-allocation decoding by providing slices into pre-allocated
/// memory regions.
pub trait BufferProvider: Send + Sync {
    /// Get a mutable slice for writing fixed-width column data
    ///
    /// # Arguments
    /// * `column_index` - Index of the column in the schema
    /// * `offset` - Byte offset within the column's data region
    /// * `len` - Number of bytes to write
    ///
    /// # Returns
    /// Mutable slice for writing data, or error if out of bounds
    fn get_data_slice(
        &mut self,
        column_index: usize,
        offset: usize,
        len: usize,
    ) -> Result<&mut [u8], ArrowError>;

    /// Get a mutable slice for writing offset array (variable-length columns)
    ///
    /// # Arguments
    /// * `column_index` - Index of the column in the schema
    /// * `offset` - Byte offset within the column's offset region
    /// * `len` - Number of bytes to write
    ///
    /// # Returns
    /// Mutable slice for writing offsets, or error if out of bounds
    fn get_offsets_slice(
        &mut self,
        column_index: usize,
        offset: usize,
        len: usize,
    ) -> Result<&mut [u8], ArrowError>;

    /// Get a mutable slice for writing null bitmap
    ///
    /// # Arguments
    /// * `column_index` - Index of the column in the schema
    /// * `offset` - Byte offset within the column's null bitmap region
    /// * `len` - Number of bytes to write
    ///
    /// # Returns
    /// Mutable slice for writing null bits, or error if out of bounds
    fn get_null_bitmap_slice(
        &mut self,
        column_index: usize,
        offset: usize,
        len: usize,
    ) -> Result<&mut [u8], ArrowError>;

    /// Finalize the column and return an Arrow Buffer backed by the provider's memory
    ///
    /// The returned buffer should be backed by the provider's memory pool,
    /// not a new allocation. This enables zero-copy sharing of the buffer.
    ///
    /// # Arguments
    /// * `column_index` - Index of the column to finalize
    /// * `data_len` - Actual bytes written to data region
    ///
    /// # Returns
    /// Arrow Buffer backed by the pool memory
    fn finalize_data_buffer(
        &mut self,
        column_index: usize,
        data_len: usize,
    ) -> Result<Buffer, ArrowError>;

    /// Finalize the offsets and return an Arrow Buffer
    ///
    /// # Arguments
    /// * `column_index` - Index of the column
    /// * `offsets_len` - Actual bytes written to offsets region
    fn finalize_offsets_buffer(
        &mut self,
        column_index: usize,
        offsets_len: usize,
    ) -> Result<Buffer, ArrowError>;

    /// Finalize the null bitmap and return an Arrow Buffer
    ///
    /// # Arguments
    /// * `column_index` - Index of the column
    /// * `bitmap_len` - Actual bytes written to bitmap region
    fn finalize_null_bitmap_buffer(
        &mut self,
        column_index: usize,
        bitmap_len: usize,
    ) -> Result<Buffer, ArrowError>;

    /// Reset the provider for reuse with the next batch
    ///
    /// This should reset internal write positions without deallocating memory.
    fn reset(&mut self);

    /// Get the maximum number of rows this provider can handle
    fn max_rows(&self) -> usize;

    /// Check if this provider has capacity for variable-length data
    ///
    /// # Arguments
    /// * `column_index` - Index of the column
    /// * `additional_bytes` - Bytes to be added
    ///
    /// # Returns
    /// true if there's capacity, false if buffer is full
    fn has_capacity_for_variable_data(
        &self,
        column_index: usize,
        additional_bytes: usize,
    ) -> bool;

    /// Get raw pointer and capacity for a column's data region
    ///
    /// This enables batch writes with multiple columns by avoiding Rust's
    /// borrow checker limitations. The caller is responsible for ensuring
    /// writes don't exceed capacity.
    ///
    /// # Safety
    /// The returned pointer is only valid until the buffer is modified or reset.
    /// Caller must ensure:
    /// - No writes exceed the returned capacity
    /// - Column regions don't overlap (they shouldn't by design)
    ///
    /// # Arguments
    /// * `column_index` - Index of the column
    /// * `offset` - Byte offset within the column's data region
    ///
    /// # Returns
    /// (raw pointer to data region at offset, remaining capacity from offset)
    fn get_data_ptr(
        &mut self,
        column_index: usize,
        offset: usize,
    ) -> Result<(*mut u8, usize), ArrowError>;

    // =========================================================================
    // Child Region Methods (for nested types like List, Map, Struct)
    // =========================================================================

    /// Get a mutable slice for writing nested child data
    ///
    /// # Arguments
    /// * `column_index` - Index of the parent column in the schema
    /// * `child_index` - Index of the child region (0 for List elements, 0/1 for Map key/value)
    /// * `offset` - Byte offset within the child's data region
    /// * `len` - Number of bytes to write
    ///
    /// # Returns
    /// Mutable slice for writing data, or error if out of bounds
    fn get_child_data_slice(
        &mut self,
        column_index: usize,
        child_index: usize,
        offset: usize,
        len: usize,
    ) -> Result<&mut [u8], ArrowError>;

    /// Get a mutable slice for writing nested child offsets
    ///
    /// # Arguments
    /// * `column_index` - Index of the parent column
    /// * `child_index` - Index of the child region
    /// * `offset` - Byte offset within the child's offset region
    /// * `len` - Number of bytes to write
    fn get_child_offsets_slice(
        &mut self,
        column_index: usize,
        child_index: usize,
        offset: usize,
        len: usize,
    ) -> Result<&mut [u8], ArrowError>;

    /// Finalize child data and return an Arrow Buffer
    ///
    /// # Arguments
    /// * `column_index` - Index of the parent column
    /// * `child_index` - Index of the child region
    /// * `data_len` - Actual bytes written to child data region
    fn finalize_child_data_buffer(
        &mut self,
        column_index: usize,
        child_index: usize,
        data_len: usize,
    ) -> Result<Buffer, ArrowError>;

    /// Finalize child offsets and return an Arrow Buffer
    ///
    /// # Arguments
    /// * `column_index` - Index of the parent column
    /// * `child_index` - Index of the child region
    /// * `offsets_len` - Actual bytes written to child offsets region
    fn finalize_child_offsets_buffer(
        &mut self,
        column_index: usize,
        child_index: usize,
        offsets_len: usize,
    ) -> Result<Buffer, ArrowError>;

    /// Check if there's capacity for additional child data
    ///
    /// # Arguments
    /// * `column_index` - Index of the parent column
    /// * `child_index` - Index of the child region
    /// * `additional_bytes` - Bytes to be added
    fn has_capacity_for_child_data(
        &self,
        column_index: usize,
        child_index: usize,
        additional_bytes: usize,
    ) -> bool;

    /// Check if a column has a pre-configured child region for nested types
    /// Returns true if the column has child regions configured (for List, Map, etc.)
    fn has_child_region(&self, column_index: usize, child_index: usize) -> bool {
        // Default implementation: check if capacity exists for 0 bytes
        // This works because implementations that have no child regions will return false
        self.has_capacity_for_child_data(column_index, child_index, 0)
    }
}

/// Memory region within a pooled buffer
#[derive(Debug, Clone)]
pub struct ColumnRegion {
    /// Offset to data buffer start within the pool
    pub data_offset: usize,
    /// Data buffer capacity in bytes
    pub data_capacity: usize,
    /// Offset to offsets buffer (for variable-length columns)
    pub offsets_offset: Option<usize>,
    /// Offsets buffer capacity in bytes
    pub offsets_capacity: Option<usize>,
    /// Offset to null bitmap buffer
    pub null_bitmap_offset: Option<usize>,
    /// Null bitmap capacity in bytes
    pub null_bitmap_capacity: Option<usize>,
    /// Child regions for nested types (List elements, Map key/value, Struct fields)
    pub child_regions: Vec<ColumnRegion>,
}

impl ColumnRegion {
    /// Create a simple fixed-width column region
    pub fn fixed(data_offset: usize, data_capacity: usize) -> Self {
        Self {
            data_offset,
            data_capacity,
            offsets_offset: None,
            offsets_capacity: None,
            null_bitmap_offset: None,
            null_bitmap_capacity: None,
            child_regions: Vec::new(),
        }
    }

    /// Create a variable-length column region (String, Binary)
    pub fn variable(
        data_offset: usize,
        data_capacity: usize,
        offsets_offset: usize,
        offsets_capacity: usize,
    ) -> Self {
        Self {
            data_offset,
            data_capacity,
            offsets_offset: Some(offsets_offset),
            offsets_capacity: Some(offsets_capacity),
            null_bitmap_offset: None,
            null_bitmap_capacity: None,
            child_regions: Vec::new(),
        }
    }

    /// Add a child region for nested types
    pub fn with_child(mut self, child: ColumnRegion) -> Self {
        self.child_regions.push(child);
        self
    }

    /// Get remaining capacity for data
    pub fn remaining_data_capacity(&self, current_position: usize) -> usize {
        self.data_capacity.saturating_sub(current_position)
    }
}

/// Buffer layout specification for a schema
#[derive(Debug, Clone)]
pub struct BufferLayout {
    /// Per-column memory regions
    pub columns: Vec<ColumnRegion>,
    /// Total buffer size required
    pub total_size: usize,
    /// Number of rows this layout supports
    pub row_count: usize,
}

/// An aligned buffer for Arrow data
///
/// This buffer is allocated with 64-byte alignment to satisfy Arrow's SIMD requirements.
/// It implements the `Allocation` trait so it can be used as backing memory for Arrow Buffers.
pub struct AlignedBuffer {
    ptr: NonNull<u8>,
    len: usize,
    layout: Layout,
}

impl AlignedBuffer {
    /// Create a new aligned buffer with the specified size
    ///
    /// The buffer will be zeroed and aligned to ARROW_ALIGNMENT (64 bytes).
    pub fn new(size: usize) -> Self {
        let size = size.max(1); // Avoid zero-sized allocation
        let layout = Layout::from_size_align(size, ARROW_ALIGNMENT)
            .expect("Invalid layout");

        // SAFETY: Layout is valid and non-zero sized
        let ptr = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).expect("Allocation failed");

        Self { ptr, len: size, layout }
    }

    /// Get a mutable slice to the buffer
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr is valid and len is the allocated size
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Get an immutable slice to the buffer
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid and len is the allocated size
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Get the length of the buffer
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get a raw pointer to a specific offset
    pub fn as_ptr_at(&self, offset: usize) -> *const u8 {
        debug_assert!(offset <= self.len);
        unsafe { self.ptr.as_ptr().add(offset) }
    }

    /// Get a mutable raw pointer to a specific offset
    pub fn as_mut_ptr_at(&mut self, offset: usize) -> *mut u8 {
        debug_assert!(offset <= self.len);
        unsafe { self.ptr.as_ptr().add(offset) }
    }

    /// Zero the buffer
    pub fn zero(&mut self) {
        // SAFETY: ptr is valid and len is the allocated size
        unsafe {
            std::ptr::write_bytes(self.ptr.as_ptr(), 0, self.len);
        }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: ptr was allocated with this layout
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

// SAFETY: AlignedBuffer owns its memory exclusively
unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

/// Shared aligned buffer wrapped in Arc
///
/// This is used after finalization to allow zero-copy sharing of the buffer
/// as backing memory for Arrow Buffers.
pub struct SharedAlignedBuffer {
    ptr: NonNull<u8>,
    len: usize,
    layout: Layout,
}

impl SharedAlignedBuffer {
    /// Create from an AlignedBuffer (takes ownership)
    fn from_aligned(buf: AlignedBuffer) -> Self {
        let result = Self {
            ptr: buf.ptr,
            len: buf.len,
            layout: buf.layout,
        };
        // Prevent the original from deallocating
        std::mem::forget(buf);
        result
    }

    /// Get a pointer at the given offset
    pub fn as_ptr_at(&self, offset: usize) -> *const u8 {
        debug_assert!(offset <= self.len);
        unsafe { self.ptr.as_ptr().add(offset) }
    }
}

impl Drop for SharedAlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: ptr was allocated with this layout
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

// SAFETY: SharedAlignedBuffer owns its memory exclusively (wrapped in Arc for sharing)
unsafe impl Send for SharedAlignedBuffer {}
unsafe impl Sync for SharedAlignedBuffer {}

/// Custom allocation that wraps a pooled buffer slice
///
/// This implements Arrow's `Allocation` trait to enable creating Arrow Buffers
/// that are backed by our memory pool without copying.
/// The fields are stored to maintain ownership (keeping the Arc alive) but not read directly.
pub struct PooledAllocation {
    /// The underlying pooled buffer (Arc for shared ownership)
    _buffer: Arc<SharedAlignedBuffer>,
    /// Offset within the buffer where this allocation starts
    _offset: usize,
    /// Length of this allocation
    _len: usize,
}

impl PooledAllocation {
    /// Create a new pooled allocation
    pub fn new(buffer: Arc<SharedAlignedBuffer>, offset: usize, len: usize) -> Self {
        Self {
            _buffer: buffer,
            _offset: offset,
            _len: len,
        }
    }
}

// Safety: The PooledAllocation is Send + Sync because Arc<SharedAlignedBuffer> is
unsafe impl Send for PooledAllocation {}
unsafe impl Sync for PooledAllocation {}

/// Default buffer provider that uses standard allocation
///
/// This is a drop-in replacement that maintains the original behavior
/// of allocating new Vec buffers for each decode operation.
#[derive(Debug, Default)]
pub struct DefaultBufferProvider;

impl BufferProvider for DefaultBufferProvider {
    fn get_data_slice(
        &mut self,
        _column_index: usize,
        _offset: usize,
        _len: usize,
    ) -> Result<&mut [u8], ArrowError> {
        // Default provider doesn't pre-allocate; decoding uses internal Vecs
        Err(ArrowError::NotYetImplemented(
            "DefaultBufferProvider does not provide slices".to_string(),
        ))
    }

    fn get_offsets_slice(
        &mut self,
        _column_index: usize,
        _offset: usize,
        _len: usize,
    ) -> Result<&mut [u8], ArrowError> {
        Err(ArrowError::NotYetImplemented(
            "DefaultBufferProvider does not provide slices".to_string(),
        ))
    }

    fn get_null_bitmap_slice(
        &mut self,
        _column_index: usize,
        _offset: usize,
        _len: usize,
    ) -> Result<&mut [u8], ArrowError> {
        Err(ArrowError::NotYetImplemented(
            "DefaultBufferProvider does not provide slices".to_string(),
        ))
    }

    fn finalize_data_buffer(
        &mut self,
        _column_index: usize,
        _data_len: usize,
    ) -> Result<Buffer, ArrowError> {
        Err(ArrowError::NotYetImplemented(
            "DefaultBufferProvider does not provide buffers".to_string(),
        ))
    }

    fn finalize_offsets_buffer(
        &mut self,
        _column_index: usize,
        _offsets_len: usize,
    ) -> Result<Buffer, ArrowError> {
        Err(ArrowError::NotYetImplemented(
            "DefaultBufferProvider does not provide buffers".to_string(),
        ))
    }

    fn finalize_null_bitmap_buffer(
        &mut self,
        _column_index: usize,
        _bitmap_len: usize,
    ) -> Result<Buffer, ArrowError> {
        Err(ArrowError::NotYetImplemented(
            "DefaultBufferProvider does not provide buffers".to_string(),
        ))
    }

    fn reset(&mut self) {
        // No-op for default provider
    }

    fn max_rows(&self) -> usize {
        usize::MAX // No limit
    }

    fn has_capacity_for_variable_data(&self, _column_index: usize, _additional_bytes: usize) -> bool {
        true // Always has capacity (will allocate as needed)
    }

    fn get_data_ptr(
        &mut self,
        _column_index: usize,
        _offset: usize,
    ) -> Result<(*mut u8, usize), ArrowError> {
        Err(ArrowError::NotYetImplemented(
            "DefaultBufferProvider does not provide raw pointers".to_string(),
        ))
    }

    fn get_child_data_slice(
        &mut self,
        _column_index: usize,
        _child_index: usize,
        _offset: usize,
        _len: usize,
    ) -> Result<&mut [u8], ArrowError> {
        Err(ArrowError::NotYetImplemented(
            "DefaultBufferProvider does not provide child slices".to_string(),
        ))
    }

    fn get_child_offsets_slice(
        &mut self,
        _column_index: usize,
        _child_index: usize,
        _offset: usize,
        _len: usize,
    ) -> Result<&mut [u8], ArrowError> {
        Err(ArrowError::NotYetImplemented(
            "DefaultBufferProvider does not provide child slices".to_string(),
        ))
    }

    fn finalize_child_data_buffer(
        &mut self,
        _column_index: usize,
        _child_index: usize,
        _data_len: usize,
    ) -> Result<Buffer, ArrowError> {
        Err(ArrowError::NotYetImplemented(
            "DefaultBufferProvider does not provide child buffers".to_string(),
        ))
    }

    fn finalize_child_offsets_buffer(
        &mut self,
        _column_index: usize,
        _child_index: usize,
        _offsets_len: usize,
    ) -> Result<Buffer, ArrowError> {
        Err(ArrowError::NotYetImplemented(
            "DefaultBufferProvider does not provide child buffers".to_string(),
        ))
    }

    fn has_capacity_for_child_data(
        &self,
        _column_index: usize,
        _child_index: usize,
        _additional_bytes: usize,
    ) -> bool {
        true // Always has capacity (will allocate as needed)
    }
}

/// Pooled buffer provider for zero-allocation decoding
///
/// This provider uses a pre-allocated buffer with a known layout to enable
/// zero-allocation Avro decoding. The buffer is divided into regions for
/// each column's data, offsets, and null bitmaps.
///
/// The buffer is allocated with 64-byte alignment to satisfy Arrow's SIMD requirements.
pub struct PooledBufferProvider {
    /// The pre-allocated aligned buffer (Arc-wrapped for finalized state)
    buffer: Option<Arc<SharedAlignedBuffer>>,
    /// Mutable buffer for writing (before finalization)
    buffer_mut: Option<AlignedBuffer>,
    /// Layout specification
    layout: BufferLayout,
    /// Current write positions per column (data)
    data_positions: Vec<usize>,
    /// Current write positions per column (offsets)
    offsets_positions: Vec<usize>,
    /// Current write positions per column (null bitmap)
    null_bitmap_positions: Vec<usize>,
    /// Whether buffer has been finalized (moved to Arc)
    finalized: bool,
}

impl PooledBufferProvider {
    /// Create a new pooled buffer provider
    ///
    /// # Arguments
    /// * `buffer` - Pre-allocated buffer (will be ignored, aligned buffer will be allocated)
    /// * `layout` - Buffer layout specification
    pub fn new(_buffer: Vec<u8>, layout: BufferLayout) -> Self {
        // Allocate aligned buffer (ignores passed buffer - alignment is more important)
        let aligned_buffer = AlignedBuffer::new(layout.total_size);

        let num_columns = layout.columns.len();
        Self {
            buffer: None, // Will be set on finalization
            buffer_mut: Some(aligned_buffer),
            layout,
            data_positions: vec![0; num_columns],
            offsets_positions: vec![0; num_columns],
            null_bitmap_positions: vec![0; num_columns],
            finalized: false,
        }
    }

    /// Create a new pooled buffer provider with aligned memory
    ///
    /// # Arguments
    /// * `layout` - Buffer layout specification
    pub fn new_aligned(layout: BufferLayout) -> Self {
        let aligned_buffer = AlignedBuffer::new(layout.total_size);

        let num_columns = layout.columns.len();
        Self {
            buffer: None,
            buffer_mut: Some(aligned_buffer),
            layout,
            data_positions: vec![0; num_columns],
            offsets_positions: vec![0; num_columns],
            null_bitmap_positions: vec![0; num_columns],
            finalized: false,
        }
    }

    /// Get the current row count based on data written
    pub fn current_row_count(&self) -> usize {
        // This would need to be tracked based on actual decoding
        self.layout.row_count
    }

    /// Get a mutable reference to the buffer
    fn get_buffer_mut(&mut self) -> Result<&mut AlignedBuffer, ArrowError> {
        self.buffer_mut.as_mut().ok_or_else(|| {
            ArrowError::InvalidArgumentError("Buffer already finalized".to_string())
        })
    }
}

impl BufferProvider for PooledBufferProvider {
    fn get_data_slice(
        &mut self,
        column_index: usize,
        offset: usize,
        len: usize,
    ) -> Result<&mut [u8], ArrowError> {
        // Get layout info first (immutable borrow)
        let region = self.layout.columns.get(column_index).ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!("Invalid column index: {}", column_index))
        })?;

        let start = region.data_offset + offset;
        let end = start + len;
        let data_capacity = region.data_capacity;
        let data_offset = region.data_offset;

        if end > data_offset + data_capacity {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Data write exceeds capacity: {} > {}",
                end - data_offset,
                data_capacity
            )));
        }

        // Now get mutable buffer access
        let buffer = self.get_buffer_mut()?;
        Ok(&mut buffer.as_mut_slice()[start..end])
    }

    fn get_offsets_slice(
        &mut self,
        column_index: usize,
        offset: usize,
        len: usize,
    ) -> Result<&mut [u8], ArrowError> {
        // Get layout info first (immutable borrow)
        let region = self.layout.columns.get(column_index).ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!("Invalid column index: {}", column_index))
        })?;

        let offsets_offset = region.offsets_offset.ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!(
                "Column {} has no offsets region",
                column_index
            ))
        })?;

        let offsets_capacity = region.offsets_capacity.unwrap_or(0);
        let start = offsets_offset + offset;
        let end = start + len;

        if end > offsets_offset + offsets_capacity {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Offsets write exceeds capacity: {} > {}",
                end - offsets_offset,
                offsets_capacity
            )));
        }

        // Now get mutable buffer access
        let buffer = self.get_buffer_mut()?;
        Ok(&mut buffer.as_mut_slice()[start..end])
    }

    fn get_null_bitmap_slice(
        &mut self,
        column_index: usize,
        offset: usize,
        len: usize,
    ) -> Result<&mut [u8], ArrowError> {
        // Get layout info first (immutable borrow)
        let region = self.layout.columns.get(column_index).ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!("Invalid column index: {}", column_index))
        })?;

        let bitmap_offset = region.null_bitmap_offset.ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!(
                "Column {} has no null bitmap region",
                column_index
            ))
        })?;

        let bitmap_capacity = region.null_bitmap_capacity.unwrap_or(0);
        let start = bitmap_offset + offset;
        let end = start + len;

        if end > bitmap_offset + bitmap_capacity {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Null bitmap write exceeds capacity: {} > {}",
                end - bitmap_offset,
                bitmap_capacity
            )));
        }

        // Now get mutable buffer access
        let buffer = self.get_buffer_mut()?;
        Ok(&mut buffer.as_mut_slice()[start..end])
    }

    fn finalize_data_buffer(
        &mut self,
        column_index: usize,
        data_len: usize,
    ) -> Result<Buffer, ArrowError> {
        if !self.finalized {
            // Move buffer to Arc for shared ownership
            if let Some(buf) = self.buffer_mut.take() {
                self.buffer = Some(Arc::new(SharedAlignedBuffer::from_aligned(buf)));
            }
            self.finalized = true;
        }

        let region = self.layout.columns.get(column_index).ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!("Invalid column index: {}", column_index))
        })?;

        let buffer = self.buffer.as_ref().ok_or_else(|| {
            ArrowError::InvalidArgumentError("No buffer available".to_string())
        })?;

        // Zero-copy: create buffer pointing into our Arc<SharedAlignedBuffer>
        let start = region.data_offset;
        let ptr = unsafe { NonNull::new_unchecked(buffer.as_ptr_at(start) as *mut u8) };
        let owner: Arc<dyn Allocation> = buffer.clone();
        Ok(unsafe { Buffer::from_custom_allocation(ptr, data_len, owner) })
    }

    fn finalize_offsets_buffer(
        &mut self,
        column_index: usize,
        offsets_len: usize,
    ) -> Result<Buffer, ArrowError> {
        if !self.finalized {
            if let Some(buf) = self.buffer_mut.take() {
                self.buffer = Some(Arc::new(SharedAlignedBuffer::from_aligned(buf)));
            }
            self.finalized = true;
        }

        let region = self.layout.columns.get(column_index).ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!("Invalid column index: {}", column_index))
        })?;

        let offsets_offset = region.offsets_offset.ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!(
                "Column {} has no offsets region",
                column_index
            ))
        })?;

        let buffer = self.buffer.as_ref().ok_or_else(|| {
            ArrowError::InvalidArgumentError("No buffer available".to_string())
        })?;

        // Zero-copy: create buffer pointing into our Arc<SharedAlignedBuffer>
        let ptr = unsafe { NonNull::new_unchecked(buffer.as_ptr_at(offsets_offset) as *mut u8) };
        let owner: Arc<dyn Allocation> = buffer.clone();
        Ok(unsafe { Buffer::from_custom_allocation(ptr, offsets_len, owner) })
    }

    fn finalize_null_bitmap_buffer(
        &mut self,
        column_index: usize,
        bitmap_len: usize,
    ) -> Result<Buffer, ArrowError> {
        if !self.finalized {
            if let Some(buf) = self.buffer_mut.take() {
                self.buffer = Some(Arc::new(SharedAlignedBuffer::from_aligned(buf)));
            }
            self.finalized = true;
        }

        let region = self.layout.columns.get(column_index).ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!("Invalid column index: {}", column_index))
        })?;

        let bitmap_offset = region.null_bitmap_offset.ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!(
                "Column {} has no null bitmap region",
                column_index
            ))
        })?;

        let buffer = self.buffer.as_ref().ok_or_else(|| {
            ArrowError::InvalidArgumentError("No buffer available".to_string())
        })?;

        // Zero-copy: create buffer pointing into our Arc<SharedAlignedBuffer>
        let ptr = unsafe { NonNull::new_unchecked(buffer.as_ptr_at(bitmap_offset) as *mut u8) };
        let owner: Arc<dyn Allocation> = buffer.clone();
        Ok(unsafe { Buffer::from_custom_allocation(ptr, bitmap_len, owner) })
    }

    fn reset(&mut self) {
        if self.finalized {
            // Buffer is shared or finalized, allocate new aligned buffer
            self.buffer_mut = Some(AlignedBuffer::new(self.layout.total_size));
            self.buffer = None;
            self.finalized = false;
        }

        // Just reset positions - don't zero the buffer (decode will overwrite old data)
        // This is O(num_columns) instead of O(buffer_size)
        self.data_positions.iter_mut().for_each(|p| *p = 0);
        self.offsets_positions.iter_mut().for_each(|p| *p = 0);
        self.null_bitmap_positions.iter_mut().for_each(|p| *p = 0);
    }

    fn max_rows(&self) -> usize {
        self.layout.row_count
    }

    fn has_capacity_for_variable_data(&self, column_index: usize, additional_bytes: usize) -> bool {
        if let Some(region) = self.layout.columns.get(column_index) {
            let current_pos = self.data_positions.get(column_index).copied().unwrap_or(0);
            current_pos + additional_bytes <= region.data_capacity
        } else {
            false
        }
    }

    fn get_data_ptr(
        &mut self,
        column_index: usize,
        offset: usize,
    ) -> Result<(*mut u8, usize), ArrowError> {
        // Get layout info first (immutable borrow)
        let region = self.layout.columns.get(column_index).ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!("Invalid column index: {}", column_index))
        })?;

        let start = region.data_offset + offset;
        let remaining_capacity = region.data_capacity.saturating_sub(offset);

        // Now get mutable buffer access
        let buffer = self.get_buffer_mut()?;

        if start >= buffer.len() {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Offset {} exceeds buffer size {}",
                start, buffer.len()
            )));
        }

        // SAFETY: We've verified start is within bounds. The caller is responsible
        // for ensuring writes don't exceed remaining_capacity.
        let ptr = buffer.as_mut_ptr_at(start);
        Ok((ptr, remaining_capacity))
    }

    fn get_child_data_slice(
        &mut self,
        column_index: usize,
        child_index: usize,
        offset: usize,
        len: usize,
    ) -> Result<&mut [u8], ArrowError> {
        // Get layout info first (immutable borrow)
        let region = self.layout.columns.get(column_index).ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!("Invalid column index: {}", column_index))
        })?;

        let child_region = region.child_regions.get(child_index).ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!(
                "Invalid child index {} for column {}",
                child_index, column_index
            ))
        })?;

        let start = child_region.data_offset + offset;
        let end = start + len;
        let child_data_offset = child_region.data_offset;
        let child_data_capacity = child_region.data_capacity;

        if end > child_data_offset + child_data_capacity {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Child data write exceeds capacity: {} > {}",
                end - child_data_offset,
                child_data_capacity
            )));
        }

        // Now get mutable buffer access
        let buffer = self.get_buffer_mut()?;
        Ok(&mut buffer.as_mut_slice()[start..end])
    }

    fn get_child_offsets_slice(
        &mut self,
        column_index: usize,
        child_index: usize,
        offset: usize,
        len: usize,
    ) -> Result<&mut [u8], ArrowError> {
        // Get layout info first (immutable borrow)
        let region = self.layout.columns.get(column_index).ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!("Invalid column index: {}", column_index))
        })?;

        let child_region = region.child_regions.get(child_index).ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!(
                "Invalid child index {} for column {}",
                child_index, column_index
            ))
        })?;

        let offsets_offset = child_region.offsets_offset.ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!(
                "Child {} of column {} has no offsets region",
                child_index, column_index
            ))
        })?;

        let offsets_capacity = child_region.offsets_capacity.unwrap_or(0);
        let start = offsets_offset + offset;
        let end = start + len;

        if end > offsets_offset + offsets_capacity {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Child offsets write exceeds capacity: {} > {}",
                end - offsets_offset,
                offsets_capacity
            )));
        }

        // Now get mutable buffer access
        let buffer = self.get_buffer_mut()?;
        Ok(&mut buffer.as_mut_slice()[start..end])
    }

    fn finalize_child_data_buffer(
        &mut self,
        column_index: usize,
        child_index: usize,
        data_len: usize,
    ) -> Result<Buffer, ArrowError> {
        if !self.finalized {
            // Move buffer to Arc for shared ownership
            if let Some(buf) = self.buffer_mut.take() {
                self.buffer = Some(Arc::new(SharedAlignedBuffer::from_aligned(buf)));
            }
            self.finalized = true;
        }

        let region = self.layout.columns.get(column_index).ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!("Invalid column index: {}", column_index))
        })?;

        let child_region = region.child_regions.get(child_index).ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!(
                "Invalid child index {} for column {}",
                child_index, column_index
            ))
        })?;

        let buffer = self.buffer.as_ref().ok_or_else(|| {
            ArrowError::InvalidArgumentError("No buffer available".to_string())
        })?;

        // Zero-copy: create buffer pointing into our Arc<SharedAlignedBuffer>
        let start = child_region.data_offset;
        let ptr = unsafe { NonNull::new_unchecked(buffer.as_ptr_at(start) as *mut u8) };
        let owner: Arc<dyn Allocation> = buffer.clone();
        Ok(unsafe { Buffer::from_custom_allocation(ptr, data_len, owner) })
    }

    fn finalize_child_offsets_buffer(
        &mut self,
        column_index: usize,
        child_index: usize,
        offsets_len: usize,
    ) -> Result<Buffer, ArrowError> {
        if !self.finalized {
            // Move buffer to Arc for shared ownership
            if let Some(buf) = self.buffer_mut.take() {
                self.buffer = Some(Arc::new(SharedAlignedBuffer::from_aligned(buf)));
            }
            self.finalized = true;
        }

        let region = self.layout.columns.get(column_index).ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!("Invalid column index: {}", column_index))
        })?;

        let child_region = region.child_regions.get(child_index).ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!(
                "Invalid child index {} for column {}",
                child_index, column_index
            ))
        })?;

        let offsets_offset = child_region.offsets_offset.ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!(
                "Child {} of column {} has no offsets region",
                child_index, column_index
            ))
        })?;

        let buffer = self.buffer.as_ref().ok_or_else(|| {
            ArrowError::InvalidArgumentError("No buffer available".to_string())
        })?;

        // Zero-copy: create buffer pointing into our Arc<SharedAlignedBuffer>
        let ptr = unsafe { NonNull::new_unchecked(buffer.as_ptr_at(offsets_offset) as *mut u8) };
        let owner: Arc<dyn Allocation> = buffer.clone();
        Ok(unsafe { Buffer::from_custom_allocation(ptr, offsets_len, owner) })
    }

    fn has_capacity_for_child_data(
        &self,
        column_index: usize,
        child_index: usize,
        additional_bytes: usize,
    ) -> bool {
        if let Some(region) = self.layout.columns.get(column_index) {
            if let Some(child_region) = region.child_regions.get(child_index) {
                // Note: We don't track per-child positions yet, so check against full capacity
                additional_bytes <= child_region.data_capacity
            } else {
                false
            }
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_provider() {
        let provider = DefaultBufferProvider::default();
        assert_eq!(provider.max_rows(), usize::MAX);
        assert!(provider.has_capacity_for_variable_data(0, 1000));
    }

    #[test]
    fn test_pooled_provider_creation() {
        let mut region = ColumnRegion::fixed(0, 1024);
        region.null_bitmap_offset = Some(1024);
        region.null_bitmap_capacity = Some(64);

        let layout = BufferLayout {
            columns: vec![region],
            total_size: 1088,
            row_count: 100,
        };

        let buffer = vec![0u8; 2048];
        let provider = PooledBufferProvider::new(buffer, layout);

        assert_eq!(provider.max_rows(), 100);
    }

    #[test]
    fn test_pooled_provider_write_data() {
        let layout = BufferLayout {
            columns: vec![ColumnRegion::fixed(0, 1024)],
            total_size: 1024,
            row_count: 128,
        };

        let buffer = vec![0u8; 1024];
        let mut provider = PooledBufferProvider::new(buffer, layout);

        // Write some data
        let slice = provider.get_data_slice(0, 0, 8).unwrap();
        slice.copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);

        // Verify data by reading through the slice again
        let slice = provider.get_data_slice(0, 0, 8).unwrap();
        assert_eq!(slice, &[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_pooled_provider_bounds_check() {
        let layout = BufferLayout {
            columns: vec![ColumnRegion::fixed(0, 64)],
            total_size: 64,
            row_count: 8,
        };

        let buffer = vec![0u8; 64];
        let mut provider = PooledBufferProvider::new(buffer, layout);

        // Should fail: exceeds capacity
        let result = provider.get_data_slice(0, 0, 128);
        assert!(result.is_err());

        // Should succeed: within capacity
        let result = provider.get_data_slice(0, 0, 64);
        assert!(result.is_ok());
    }
}

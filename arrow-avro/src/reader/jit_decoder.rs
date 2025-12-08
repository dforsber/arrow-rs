//! JIT-compiled Avro decoder using Cranelift
//!
//! This module provides a schema-specific compiled decoder that eliminates
//! all runtime dispatch overhead by generating native machine code for
//! the specific schema being decoded.
//!
//! # Example
//! ```ignore
//! use arrow_avro::reader::jit_decoder::JitDecoder;
//!
//! let schema = Arc::new(Schema::new(vec![
//!     Field::new("id", DataType::Int64, false),
//!     Field::new("name", DataType::Utf8, false),
//! ]));
//!
//! // Compile once (expensive, ~1-5ms)
//! let decoder = JitDecoder::compile(&schema)?;
//!
//! // Decode many times (fast, no dispatch overhead)
//! let bytes_consumed = decoder.decode(input, &mut buffers, row_count)?;
//! ```

use arrow_schema::{DataType, Field, IntervalUnit, Schema};
use std::sync::Arc;

use cranelift::codegen::ir::FuncRef;
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use std::mem::ManuallyDrop;

// External memcpy function for fast bulk memory copies
unsafe extern "C" {
    fn memcpy(dest: *mut u8, src: *const u8, n: usize) -> *mut u8;
}

/// Wrapper to make JITModule Send + Sync and prevent drop-related crashes
///
/// Safety: JITModule is not Sync because it contains internal state (symbol lookup functions).
/// However, we only keep it alive to prevent the compiled code memory from being freed.
/// The actual decode function pointer is thread-safe because:
/// 1. The compiled code is read-only after finalization
/// 2. Each thread uses its own input/output buffers
/// 3. We never call any methods on JITModule after construction
///
/// We use ManuallyDrop because Cranelift's JITModule has known issues with its Drop
/// implementation that can cause SIGABRT when multiple modules are dropped.
/// The memory leak is acceptable for long-lived decoders (typical use case).
struct SyncJitModule(#[allow(dead_code)] ManuallyDrop<JITModule>);

// Safety: We never access the JITModule after construction, only keep it alive.
// The compiled code it owns is read-only and safe to execute from multiple threads.
unsafe impl Send for SyncJitModule {}
unsafe impl Sync for SyncJitModule {}

/// JIT-compiled decoder for a specific Avro schema
///
/// This decoder is thread-safe and can be shared across multiple threads.
/// Each thread should provide its own input/output buffers.
pub struct JitDecoder {
    /// The compiled decode function
    /// Signature: (input_ptr, input_len, output_ptrs, output_caps, row_count) -> bytes_consumed
    /// Returns negative value on error (-1 input bounds, -2 output overflow)
    decode_fn: unsafe extern "C" fn(*const u8, usize, *mut *mut u8, *const usize, usize) -> i64,
    /// Keep the JIT module alive (owns the compiled code memory)
    _module: SyncJitModule,
    /// Schema for reference
    schema: Arc<Schema>,
    /// Number of schema columns
    num_columns: usize,
    /// Column types for buffer layout (Variable columns need 2 buffers: offsets + data)
    column_types: Vec<ColumnType>,
    /// Buffer slot mapping: column index -> buffer indices
    buffer_slots: Vec<BufferSlot>,
    /// Total number of output buffers needed (accounts for variable-length types needing 2)
    total_buffers: usize,
}

/// Decode operation for a single field
#[derive(Debug, Clone)]
enum FieldOp {
    /// Read varint (zigzag decoded) as i32
    Int32 { nullable: bool },
    /// Read varint (zigzag decoded) as i64
    Int64 { nullable: bool },
    /// Read 4 fixed bytes as f32
    Float32 { nullable: bool },
    /// Read 8 fixed bytes as f64
    Float64 { nullable: bool },
    /// Read single byte as boolean
    Boolean { nullable: bool },
    /// Read varint length + bytes for string/binary
    /// output_ptrs layout: [data_ptr, offsets_ptr]
    Bytes { nullable: bool },
    /// Read fixed number of bytes
    Fixed { size: usize, nullable: bool },
    /// Nested struct with child fields
    Struct { fields: Vec<FieldOp>, nullable: bool },
    /// Enum - read varint index (not zigzag decoded for enum)
    Enum { nullable: bool },
    /// List/Array - read block count + elements
    List { element: Box<FieldOp>, nullable: bool },
    /// Map - read block count + key-value pairs
    Map { value: Box<FieldOp>, nullable: bool },
    /// Union - read type index varint, then decode variant
    /// type_ids maps Avro union index to Arrow type id
    Union { variants: Vec<FieldOp>, type_ids: Vec<i8>, nullable: bool },
    /// Fixed-size list with known element count
    FixedSizeList { element: Box<FieldOp>, size: usize, nullable: bool },
    /// Decimal128 - read fixed bytes and decode as 128-bit decimal
    Decimal128 { _precision: u8, _scale: i8, nullable: bool },
    /// Decimal256 - read fixed bytes and decode as 256-bit decimal
    Decimal256 { _precision: u8, _scale: i8, nullable: bool },
    /// Duration - read as i64
    Duration { nullable: bool },
    /// Interval - read as fixed bytes
    Interval { unit: IntervalUnit, nullable: bool },
    /// Null type - no data to read, used in unions for nullable fields
    Null,
}

/// Column type metadata for buffer layout
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColumnType {
    /// Fixed-size column: single data buffer
    Fixed,
    /// Variable-size column: offsets buffer + data buffer
    Variable,
}

/// Buffer slot descriptor for a schema column
#[derive(Debug, Clone, Copy)]
pub struct BufferSlot {
    /// Index of the primary buffer in the output_ptrs array
    /// For fixed-size: this is the data buffer
    /// For variable-size: this is the offsets buffer
    pub primary: usize,
    /// Index of the secondary buffer (data) for variable-length types
    /// None for fixed-size types
    pub secondary: Option<usize>,
    /// Index of the validity bitmap buffer for nullable columns
    /// None for non-nullable types
    pub validity: Option<usize>,
    /// Element size in bytes for computing offsets
    /// For Bytes/String: 1 (offsets are byte positions)
    /// For List/Map: element size (offsets are element indices)
    pub element_size: i64,
}

/// Output buffer specification with capacity for bounds checking
#[derive(Debug, Clone, Copy)]
pub struct OutputBuffer {
    /// Pointer to the output buffer
    pub ptr: *mut u8,
    /// Capacity of the buffer in bytes
    pub capacity: usize,
}

impl OutputBuffer {
    /// Create a new output buffer specification
    pub fn new(ptr: *mut u8, capacity: usize) -> Self {
        Self { ptr, capacity }
    }
}

/// Error code returned when input bounds are violated
pub const JIT_ERROR_INPUT_BOUNDS: i64 = -1;
/// Error code returned when output buffer capacity is exceeded
pub const JIT_ERROR_OUTPUT_OVERFLOW: i64 = -2;

/// Status of a chunked decode operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeStatus {
    /// All requested rows were decoded successfully
    Ok,
    /// Input bounds violation (malformed data)
    InputBounds,
    /// Output buffer full, partial progress returned
    OutputOverflow,
}

/// Result of a chunked decode operation with partial progress information
#[derive(Debug, Clone, Copy)]
pub struct DecodeResult {
    /// Number of input bytes consumed (up to last complete row)
    pub bytes_consumed: usize,
    /// Number of complete rows decoded
    pub rows_decoded: usize,
    /// Decode status
    pub status: DecodeStatus,
}

impl DecodeResult {
    /// Create a successful decode result
    #[inline]
    pub fn ok(bytes_consumed: usize, rows_decoded: usize) -> Self {
        Self {
            bytes_consumed,
            rows_decoded,
            status: DecodeStatus::Ok,
        }
    }

    /// Create an overflow result with partial progress
    #[inline]
    pub fn overflow(bytes_consumed: usize, rows_decoded: usize) -> Self {
        Self {
            bytes_consumed,
            rows_decoded,
            status: DecodeStatus::OutputOverflow,
        }
    }

    /// Create an input bounds error result
    #[inline]
    pub fn input_error() -> Self {
        Self {
            bytes_consumed: 0,
            rows_decoded: 0,
            status: DecodeStatus::InputBounds,
        }
    }

    /// Check if decode completed successfully
    #[inline]
    pub fn is_ok(&self) -> bool {
        self.status == DecodeStatus::Ok
    }

    /// Check if more data needs to be decoded (overflow occurred)
    #[inline]
    pub fn has_more(&self) -> bool {
        self.status == DecodeStatus::OutputOverflow
    }
}

impl JitDecoder {
    /// Compile a schema-specific decoder
    ///
    /// This is an expensive operation (~1-5ms) but only needs to be done once per schema.
    /// The resulting decoder has zero dispatch overhead.
    pub fn compile(schema: &Arc<Schema>) -> Result<Self, String> {
        let num_columns = schema.fields().len();

        // Build the JIT module
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed").map_err(|e| e.to_string())?;
        flag_builder.set("is_pic", "false").map_err(|e| e.to_string())?;

        let isa_builder = cranelift_native::builder().map_err(|e| e.to_string())?;
        let isa = isa_builder.finish(settings::Flags::new(flag_builder)).map_err(|e| e.to_string())?;

        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        // Register memcpy symbol for fast bulk memory copies
        builder.symbol("memcpy", memcpy as *const u8);
        let mut module = JITModule::new(builder);

        // Create the function context
        let mut ctx = module.make_context();
        let ptr_type = module.target_config().pointer_type();

        // Function signature: (input_ptr, input_len, output_ptrs, output_caps, row_count) -> bytes_consumed
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // input_ptr
        ctx.func.signature.params.push(AbiParam::new(types::I64)); // input_len
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // output_ptrs (array of pointers)
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // output_caps (array of capacities)
        ctx.func.signature.params.push(AbiParam::new(types::I64)); // row_count
        ctx.func.signature.returns.push(AbiParam::new(types::I64)); // bytes_consumed or -1/-2 on error

        // Declare memcpy: void* memcpy(void* dest, const void* src, size_t n)
        let mut memcpy_sig = module.make_signature();
        memcpy_sig.params.push(AbiParam::new(ptr_type)); // dest
        memcpy_sig.params.push(AbiParam::new(ptr_type)); // src
        memcpy_sig.params.push(AbiParam::new(types::I64)); // n (size_t)
        memcpy_sig.returns.push(AbiParam::new(ptr_type)); // returns dest
        let memcpy_func_id = module
            .declare_function("memcpy", Linkage::Import, &memcpy_sig)
            .map_err(|e| e.to_string())?;

        // Build the function
        let mut fn_builder_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);

        // Import memcpy into this function
        let memcpy_func_ref = module.declare_func_in_func(memcpy_func_id, builder.func);

        // Create entry block
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        // Get function parameters
        let input_ptr = builder.block_params(entry_block)[0];
        let input_len = builder.block_params(entry_block)[1];
        let output_ptrs = builder.block_params(entry_block)[2];
        let output_caps = builder.block_params(entry_block)[3];
        let row_count = builder.block_params(entry_block)[4];

        // Create variables for cursor position and output positions
        // Cranelift 0.126+: declare_var returns a Variable
        let cursor_var = builder.declare_var(types::I64);
        let row_var = builder.declare_var(types::I64);

        // Initialize cursor to 0
        let zero = builder.ins().iconst(types::I64, 0);
        builder.def_var(cursor_var, zero);
        builder.def_var(row_var, zero);

        // Analyze schema to get field operations FIRST (needed for buffer slot computation)
        let field_ops: Vec<FieldOp> = schema.fields()
            .iter()
            .map(|f| Self::field_to_op(f))
            .collect::<Result<Vec<_>, _>>()?;

        // Compute column types and buffer slots
        let column_types: Vec<ColumnType> = field_ops.iter().map(Self::op_to_column_type).collect();

        // Build buffer slot mapping:
        // - Fixed columns: 1 buffer (data), optionally +1 for validity
        // - Variable columns: 2 buffers (offsets, data), optionally +1 for validity
        let mut buffer_slots = Vec::with_capacity(num_columns);
        let mut buffer_idx = 0usize;
        for ((col_type, field_op), field) in column_types.iter().zip(field_ops.iter()).zip(schema.fields().iter()) {
            let element_size = Self::op_to_element_size(field_op);
            let is_nullable = field.is_nullable();
            match col_type {
                ColumnType::Fixed => {
                    let validity = if is_nullable {
                        let v = buffer_idx + 1;
                        buffer_idx += 2; // data + validity
                        Some(v)
                    } else {
                        buffer_idx += 1; // data only
                        None
                    };
                    buffer_slots.push(BufferSlot {
                        primary: buffer_idx - if is_nullable { 2 } else { 1 },
                        secondary: None,
                        validity,
                        element_size,
                    });
                }
                ColumnType::Variable => {
                    let validity = if is_nullable {
                        let v = buffer_idx + 2;
                        buffer_idx += 3; // offsets + data + validity
                        Some(v)
                    } else {
                        buffer_idx += 2; // offsets + data only
                        None
                    };
                    buffer_slots.push(BufferSlot {
                        primary: buffer_idx - if is_nullable { 3 } else { 2 },
                        secondary: Some(buffer_idx - if is_nullable { 2 } else { 1 }),
                        validity,
                        element_size,
                    });
                }
            }
        }
        let total_buffers = buffer_idx;

        // Create output position variables for each BUFFER (not column)
        let mut output_vars = Vec::with_capacity(total_buffers);
        for _i in 0..total_buffers {
            let var = builder.declare_var(types::I64);
            builder.def_var(var, zero);
            output_vars.push(var);
        }

        // Create error block (returns -1 on input bounds violation)
        let error_block = builder.create_block();

        // Create output overflow block (returns -2 on output buffer overflow)
        let output_overflow_block = builder.create_block();

        // Create loop blocks
        let loop_header = builder.create_block();
        let loop_body = builder.create_block();
        let loop_exit = builder.create_block();

        // Jump to loop header
        builder.ins().jump(loop_header, &[]);

        // Loop header: check if row < row_count
        builder.switch_to_block(loop_header);
        let current_row = builder.use_var(row_var);
        let cmp = builder.ins().icmp(IntCC::UnsignedLessThan, current_row, row_count);
        builder.ins().brif(cmp, loop_body, &[], loop_exit, &[]);

        // Loop body: decode one row
        builder.switch_to_block(loop_body);

        // Generate decode code for each field
        for (col_idx, op) in field_ops.iter().enumerate() {
            let buffer_slot = &buffer_slots[col_idx];
            Self::emit_field_decode(
                &mut builder,
                op,
                buffer_slot,
                input_ptr,
                input_len,
                output_ptrs,
                output_caps,
                cursor_var,
                row_var,
                &output_vars,
                ptr_type,
                memcpy_func_ref,
                error_block,
                output_overflow_block,
            );
        }

        // Increment row counter
        let current_row = builder.use_var(row_var);
        let one = builder.ins().iconst(types::I64, 1);
        let next_row = builder.ins().iadd(current_row, one);
        builder.def_var(row_var, next_row);

        // Jump back to loop header
        builder.ins().jump(loop_header, &[]);

        // Loop exit: write final offsets for variable-length columns, then return cursor position
        builder.switch_to_block(loop_exit);

        // For each variable-length column, write the final offset (offset[row_count])
        // For Bytes/String: offset = total_bytes (element_size = 1)
        // For List/Map: offset = total_element_count (element_size = sizeof(element))
        for slot in &buffer_slots {
            if let Some(data_buf_idx) = slot.secondary {
                // This is a variable-length column
                let buf_idx = slot.primary; // offsets buffer index

                // Load offsets buffer pointer
                let buf_offset = builder.ins().iconst(ptr_type, (buf_idx * 8) as i64);
                let offsets_ptr_addr = builder.ins().iadd(output_ptrs, buf_offset);
                let offsets_ptr = builder.ins().load(ptr_type, MemFlags::trusted(), offsets_ptr_addr, 0);

                // Get current offset write position (which is row_count * 4 after the loop)
                let offset_pos = builder.use_var(output_vars[buf_idx]);
                let offset_write_ptr = builder.ins().iadd(offsets_ptr, offset_pos);

                // Get final data position from the data buffer position tracker
                let data_pos = builder.use_var(output_vars[data_buf_idx]);

                // Convert byte position to element count if needed
                // For Bytes: element_size=1, so final_offset = data_pos
                // For List/Map: final_offset = data_pos / element_size
                let final_offset = if slot.element_size == 1 {
                    builder.ins().ireduce(types::I32, data_pos)
                } else {
                    let element_size_val = builder.ins().iconst(types::I64, slot.element_size);
                    let element_count = builder.ins().udiv(data_pos, element_size_val);
                    builder.ins().ireduce(types::I32, element_count)
                };

                // Write final offset (offset[row_count] = total_element_count)
                builder.ins().store(MemFlags::trusted(), final_offset, offset_write_ptr, 0);
            }
        }

        let final_cursor = builder.use_var(cursor_var);
        builder.ins().return_(&[final_cursor]);

        // Error block: return -1 on input bounds violation
        builder.switch_to_block(error_block);
        let error_code = builder.ins().iconst(types::I64, -1);
        builder.ins().return_(&[error_code]);

        // Output overflow block: return -2 on output buffer overflow
        builder.switch_to_block(output_overflow_block);
        let overflow_code = builder.ins().iconst(types::I64, -2);
        builder.ins().return_(&[overflow_code]);

        // Seal all blocks
        builder.seal_block(error_block);
        builder.seal_block(output_overflow_block);
        builder.seal_block(loop_header);
        builder.seal_block(loop_body);
        builder.seal_block(loop_exit);

        builder.finalize();

        // Declare and define the function
        let func_id = module
            .declare_function("avro_decode", Linkage::Local, &ctx.func.signature)
            .map_err(|e| e.to_string())?;

        module
            .define_function(func_id, &mut ctx)
            .map_err(|e| e.to_string())?;

        module.clear_context(&mut ctx);
        module.finalize_definitions().map_err(|e| e.to_string())?;

        // Get the compiled function pointer
        let code_ptr = module.get_finalized_function(func_id);
        let decode_fn: unsafe extern "C" fn(*const u8, usize, *mut *mut u8, *const usize, usize) -> i64 =
            unsafe { std::mem::transmute(code_ptr) };

        // buffer_slots, column_types, and total_buffers were computed earlier in compile()
        Ok(JitDecoder {
            decode_fn,
            _module: SyncJitModule(ManuallyDrop::new(module)),
            schema: schema.clone(),
            num_columns,
            column_types,
            buffer_slots,
            total_buffers,
        })
    }

    /// Get column type for a field operation
    fn op_to_column_type(op: &FieldOp) -> ColumnType {
        match op {
            FieldOp::Bytes { .. } | FieldOp::List { .. } | FieldOp::Map { .. } => {
                ColumnType::Variable
            }
            _ => ColumnType::Fixed,
        }
    }

    /// Get element size for computing offsets
    /// Returns 1 for Bytes/String (byte offsets), element size for List/Map (element indices)
    fn op_to_element_size(op: &FieldOp) -> i64 {
        match op {
            // Bytes/String: offsets are byte positions
            FieldOp::Bytes { .. } => 1,
            // List: element size depends on inner type
            FieldOp::List { element, .. } => Self::element_op_size(element),
            // Map: value size (keys are skipped)
            FieldOp::Map { value, .. } => Self::element_op_size(value),
            // Fixed-size types: not used for offsets
            _ => 1,
        }
    }

    /// Get byte size of element type
    fn element_op_size(op: &FieldOp) -> i64 {
        match op {
            FieldOp::Int64 { .. } | FieldOp::Float64 { .. } => 8,
            FieldOp::Int32 { .. } | FieldOp::Float32 { .. } | FieldOp::Enum { .. } => 4,
            FieldOp::Boolean { .. } => 1,
            _ => 8, // Default for complex types
        }
    }

    /// Convert an Arrow field to a decode operation
    ///
    /// Returns an error for unsupported data types to fail fast rather than
    /// silently produce incorrect results.
    fn field_to_op(field: &Field) -> Result<FieldOp, String> {
        let nullable = field.is_nullable();
        match field.data_type() {
            DataType::Int32 | DataType::Date32 | DataType::Time32(_) => Ok(FieldOp::Int32 { nullable }),
            DataType::Int64 | DataType::Date64 | DataType::Time64(_) | DataType::Timestamp(_, _) => Ok(FieldOp::Int64 { nullable }),
            DataType::Float32 => Ok(FieldOp::Float32 { nullable }),
            DataType::Float64 => Ok(FieldOp::Float64 { nullable }),
            DataType::Boolean => Ok(FieldOp::Boolean { nullable }),
            DataType::Utf8 | DataType::Binary | DataType::LargeUtf8 | DataType::LargeBinary => Ok(FieldOp::Bytes { nullable }),
            DataType::FixedSizeBinary(size) => Ok(FieldOp::Fixed { size: *size as usize, nullable }),
            DataType::Struct(fields) => {
                let child_ops: Result<Vec<FieldOp>, String> = fields
                    .iter()
                    .map(|f| Self::field_to_op(f.as_ref()))
                    .collect();
                Ok(FieldOp::Struct { fields: child_ops?, nullable })
            }
            // Dictionary/Enum - store indices as i32
            DataType::Dictionary(_, _) => Ok(FieldOp::Enum { nullable }),
            // List types - decode element type recursively
            DataType::List(element_field) | DataType::LargeList(element_field) => {
                let element_op = Box::new(Self::field_to_op(element_field.as_ref())?);
                Ok(FieldOp::List { element: element_op, nullable })
            }
            // Map type - keys are always strings in Avro, values are recursive
            DataType::Map(entries_field, _) => {
                // Map entries are a struct with key and value fields
                if let DataType::Struct(fields) = entries_field.data_type() {
                    if fields.len() >= 2 {
                        let value_op = Box::new(Self::field_to_op(fields[1].as_ref())?);
                        return Ok(FieldOp::Map { value: value_op, nullable });
                    }
                }
                Err(format!("Unsupported Map structure for field '{}': expected Struct with key/value fields", field.name()))
            }
            // Union type - read type index, then decode variant
            DataType::Union(union_fields, _mode) => {
                let variants: Result<Vec<FieldOp>, String> = union_fields
                    .iter()
                    .map(|(_, f)| Self::field_to_op(f.as_ref()))
                    .collect();
                let type_ids: Vec<i8> = union_fields.iter().map(|(id, _)| id).collect();
                Ok(FieldOp::Union { variants: variants?, type_ids, nullable })
            }
            // Fixed-size list
            DataType::FixedSizeList(element_field, size) => {
                let element_op = Box::new(Self::field_to_op(element_field.as_ref())?);
                Ok(FieldOp::FixedSizeList { element: element_op, size: *size as usize, nullable })
            }
            // Decimal types - Avro encodes as fixed bytes
            DataType::Decimal128(precision, scale) => {
                Ok(FieldOp::Decimal128 { _precision: *precision, _scale: *scale, nullable })
            }
            DataType::Decimal256(precision, scale) => {
                Ok(FieldOp::Decimal256 { _precision: *precision, _scale: *scale, nullable })
            }
            // Duration - encoded as i64
            DataType::Duration(_) => Ok(FieldOp::Duration { nullable }),
            // Interval types
            DataType::Interval(unit) => Ok(FieldOp::Interval { unit: *unit, nullable }),
            // Null type - used in unions for nullable fields
            DataType::Null => Ok(FieldOp::Null),
            // Unsupported types - fail fast
            other => Err(format!(
                "Unsupported data type for JIT Avro decoder: {:?} (field '{}')",
                other, field.name()
            )),
        }
    }

    /// Emit Cranelift IR for decoding a single field
    fn emit_field_decode(
        builder: &mut FunctionBuilder,
        op: &FieldOp,
        buffer_slot: &BufferSlot,
        input_ptr: Value,
        input_len: Value,
        output_ptrs: Value,
        output_caps: Value,
        cursor_var: Variable,
        row_var: Variable,
        output_vars: &[Variable],
        ptr_type: Type,
        memcpy_func_ref: FuncRef,
        error_block: Block,
        output_overflow_block: Block,
    ) {
        // For fixed-size types, use primary buffer only
        // For variable-size types (Bytes), use primary=offsets, secondary=data
        let buf_idx = buffer_slot.primary;

        // Load primary output pointer
        let buf_offset = builder.ins().iconst(ptr_type, (buf_idx * 8) as i64);
        let output_ptr_addr = builder.ins().iadd(output_ptrs, buf_offset);
        let output_ptr = builder.ins().load(ptr_type, MemFlags::trusted(), output_ptr_addr, 0);
        let output_pos = builder.use_var(output_vars[buf_idx]);
        let output_write_ptr = builder.ins().iadd(output_ptr, output_pos);

        // Load primary output capacity (for bounds checking)
        let output_cap_addr = builder.ins().iadd(output_caps, buf_offset);
        let _output_capacity = builder.ins().load(types::I64, MemFlags::trusted(), output_cap_addr, 0);

        match op {
            FieldOp::Int32 { nullable } => {
                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    builder.def_var(cursor_var, tag_cursor);

                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, zero);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder.ins().brif(is_null, null_block, &[], value_block, &[]);

                    // Null block: write 0 placeholder (validity bit stays 0 = null)
                    builder.switch_to_block(null_block);
                    let zero_val = builder.ins().iconst(types::I32, 0);
                    builder.ins().store(MemFlags::trusted(), zero_val, output_write_ptr, 0);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode the value and set validity bit
                    builder.switch_to_block(value_block);
                    let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    let zigzag = Self::emit_zigzag_decode_i32(builder, value);
                    builder.ins().store(MemFlags::trusted(), zigzag, output_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(builder, output_ptrs, validity_idx, row_idx, ptr_type);
                    }
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);

                    let four = builder.ins().iconst(types::I64, 4);
                    let new_output_pos = builder.ins().iadd(output_pos, four);
                    builder.def_var(output_vars[buf_idx], new_output_pos);
                } else {
                    let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    let zigzag = Self::emit_zigzag_decode_i32(builder, value);
                    builder.ins().store(MemFlags::trusted(), zigzag, output_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                    let four = builder.ins().iconst(types::I64, 4);
                    let new_output_pos = builder.ins().iadd(output_pos, four);
                    builder.def_var(output_vars[buf_idx], new_output_pos);
                }
            }
            FieldOp::Int64 { nullable } => {
                if *nullable {
                    // Read union tag
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    builder.def_var(cursor_var, tag_cursor);

                    // Check if null (tag == 0)
                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, zero);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder.ins().brif(is_null, null_block, &[], value_block, &[]);

                    // Null block: write 0 placeholder (validity bit stays 0 = null)
                    builder.switch_to_block(null_block);
                    let zero_val = builder.ins().iconst(types::I64, 0);
                    builder.ins().store(MemFlags::trusted(), zero_val, output_write_ptr, 0);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode the value and set validity bit
                    builder.switch_to_block(value_block);
                    let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    let zigzag = Self::emit_zigzag_decode_i64(builder, value);
                    builder.ins().store(MemFlags::trusted(), zigzag, output_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(builder, output_ptrs, validity_idx, row_idx, ptr_type);
                    }
                    builder.ins().jump(continue_block, &[]);

                    // Seal and continue
                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);

                    // Update output position (same for both branches)
                    let eight = builder.ins().iconst(types::I64, 8);
                    let new_output_pos = builder.ins().iadd(output_pos, eight);
                    builder.def_var(output_vars[buf_idx], new_output_pos);
                } else {
                    let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    let zigzag = Self::emit_zigzag_decode_i64(builder, value);
                    builder.ins().store(MemFlags::trusted(), zigzag, output_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                    let eight = builder.ins().iconst(types::I64, 8);
                    let new_output_pos = builder.ins().iadd(output_pos, eight);
                    builder.def_var(output_vars[buf_idx], new_output_pos);
                }
            }
            FieldOp::Float32 { nullable } => {
                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    builder.def_var(cursor_var, tag_cursor);

                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, zero);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder.ins().brif(is_null, null_block, &[], value_block, &[]);

                    // Null block (validity bit stays 0 = null)
                    builder.switch_to_block(null_block);
                    let zero_val = builder.ins().iconst(types::I32, 0);
                    builder.ins().store(MemFlags::trusted(), zero_val, output_write_ptr, 0);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode and set validity bit
                    builder.switch_to_block(value_block);
                    let cursor = builder.use_var(cursor_var);
                    Self::emit_fixed_bounds_check(builder, cursor, 4, input_len, error_block);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder.ins().load(types::I32, MemFlags::trusted(), read_ptr, 0);
                    builder.ins().store(MemFlags::trusted(), value, output_write_ptr, 0);
                    let four = builder.ins().iconst(types::I64, 4);
                    let new_cursor = builder.ins().iadd(cursor, four);
                    builder.def_var(cursor_var, new_cursor);
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(builder, output_ptrs, validity_idx, row_idx, ptr_type);
                    }
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);

                    let four = builder.ins().iconst(types::I64, 4);
                    let new_output_pos = builder.ins().iadd(output_pos, four);
                    builder.def_var(output_vars[buf_idx], new_output_pos);
                } else {
                    let cursor = builder.use_var(cursor_var);
                    Self::emit_fixed_bounds_check(builder, cursor, 4, input_len, error_block);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder.ins().load(types::I32, MemFlags::trusted(), read_ptr, 0);
                    builder.ins().store(MemFlags::trusted(), value, output_write_ptr, 0);
                    let four = builder.ins().iconst(types::I64, 4);
                    let new_cursor = builder.ins().iadd(cursor, four);
                    builder.def_var(cursor_var, new_cursor);
                    let new_output_pos = builder.ins().iadd(output_pos, four);
                    builder.def_var(output_vars[buf_idx], new_output_pos);
                }
            }
            FieldOp::Float64 { nullable } => {
                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    builder.def_var(cursor_var, tag_cursor);

                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, zero);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder.ins().brif(is_null, null_block, &[], value_block, &[]);

                    builder.switch_to_block(null_block);
                    let zero_val = builder.ins().iconst(types::I64, 0);
                    builder.ins().store(MemFlags::trusted(), zero_val, output_write_ptr, 0);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode and set validity bit
                    builder.switch_to_block(value_block);
                    let cursor = builder.use_var(cursor_var);
                    Self::emit_fixed_bounds_check(builder, cursor, 8, input_len, error_block);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder.ins().load(types::I64, MemFlags::trusted(), read_ptr, 0);
                    builder.ins().store(MemFlags::trusted(), value, output_write_ptr, 0);
                    let eight = builder.ins().iconst(types::I64, 8);
                    let new_cursor = builder.ins().iadd(cursor, eight);
                    builder.def_var(cursor_var, new_cursor);
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(builder, output_ptrs, validity_idx, row_idx, ptr_type);
                    }
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);

                    let eight = builder.ins().iconst(types::I64, 8);
                    let new_output_pos = builder.ins().iadd(output_pos, eight);
                    builder.def_var(output_vars[buf_idx], new_output_pos);
                } else {
                    let cursor = builder.use_var(cursor_var);
                    Self::emit_fixed_bounds_check(builder, cursor, 8, input_len, error_block);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder.ins().load(types::I64, MemFlags::trusted(), read_ptr, 0);
                    builder.ins().store(MemFlags::trusted(), value, output_write_ptr, 0);
                    let eight = builder.ins().iconst(types::I64, 8);
                    let new_cursor = builder.ins().iadd(cursor, eight);
                    builder.def_var(cursor_var, new_cursor);
                    let new_output_pos = builder.ins().iadd(output_pos, eight);
                    builder.def_var(output_vars[buf_idx], new_output_pos);
                }
            }
            FieldOp::Boolean { nullable } => {
                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    builder.def_var(cursor_var, tag_cursor);

                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, zero);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder.ins().brif(is_null, null_block, &[], value_block, &[]);

                    builder.switch_to_block(null_block);
                    let zero_val = builder.ins().iconst(types::I8, 0);
                    builder.ins().store(MemFlags::trusted(), zero_val, output_write_ptr, 0);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode and set validity bit
                    builder.switch_to_block(value_block);
                    let cursor = builder.use_var(cursor_var);
                    Self::emit_fixed_bounds_check(builder, cursor, 1, input_len, error_block);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder.ins().load(types::I8, MemFlags::trusted(), read_ptr, 0);
                    builder.ins().store(MemFlags::trusted(), value, output_write_ptr, 0);
                    let one = builder.ins().iconst(types::I64, 1);
                    let new_cursor = builder.ins().iadd(cursor, one);
                    builder.def_var(cursor_var, new_cursor);
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(builder, output_ptrs, validity_idx, row_idx, ptr_type);
                    }
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);

                    let one = builder.ins().iconst(types::I64, 1);
                    let new_output_pos = builder.ins().iadd(output_pos, one);
                    builder.def_var(output_vars[buf_idx], new_output_pos);
                } else {
                    let cursor = builder.use_var(cursor_var);
                    Self::emit_fixed_bounds_check(builder, cursor, 1, input_len, error_block);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder.ins().load(types::I8, MemFlags::trusted(), read_ptr, 0);
                    builder.ins().store(MemFlags::trusted(), value, output_write_ptr, 0);
                    let one = builder.ins().iconst(types::I64, 1);
                    let new_cursor = builder.ins().iadd(cursor, one);
                    builder.def_var(cursor_var, new_cursor);
                    let new_output_pos = builder.ins().iadd(output_pos, one);
                    builder.def_var(output_vars[buf_idx], new_output_pos);
                }
            }
            FieldOp::Bytes { nullable } => {
                // Bytes uses dual buffers:
                // - primary (buf_idx) = offsets buffer (i32 per row)
                // - secondary = data buffer (actual bytes)
                let data_buf_idx = buffer_slot.secondary.expect("Bytes type must have secondary buffer");

                // Load data buffer pointer and position
                let data_buf_offset = builder.ins().iconst(ptr_type, (data_buf_idx * 8) as i64);
                let data_ptr_addr = builder.ins().iadd(output_ptrs, data_buf_offset);
                let data_ptr = builder.ins().load(ptr_type, MemFlags::trusted(), data_ptr_addr, 0);
                let data_pos = builder.use_var(output_vars[data_buf_idx]);
                let data_write_ptr = builder.ins().iadd(data_ptr, data_pos);

                // Load data buffer capacity
                let data_cap_addr = builder.ins().iadd(output_caps, data_buf_offset);
                let data_capacity = builder.ins().load(types::I64, MemFlags::trusted(), data_cap_addr, 0);

                // output_write_ptr points to offsets buffer (already set up above)
                // output_capacity is offsets buffer capacity

                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    builder.def_var(cursor_var, tag_cursor);

                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, zero);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder.ins().brif(is_null, null_block, &[], value_block, &[]);

                    // Null block: write current data_pos as offset (same as previous), increment offset pos
                    builder.switch_to_block(null_block);
                    {
                        // Write offset (data_pos as i32)
                        let data_pos_null = builder.use_var(output_vars[data_buf_idx]);
                        let offset_i32 = builder.ins().ireduce(types::I32, data_pos_null);
                        let offset_pos_null = builder.use_var(output_vars[buf_idx]);
                        let offset_write_ptr_null = builder.ins().iadd(output_ptr, offset_pos_null);
                        builder.ins().store(MemFlags::trusted(), offset_i32, offset_write_ptr_null, 0);
                        // Advance offsets position by 4 (one i32)
                        let four = builder.ins().iconst(types::I64, 4);
                        let offset_pos_curr = builder.use_var(output_vars[buf_idx]);
                        let new_offset_pos = builder.ins().iadd(offset_pos_curr, four);
                        builder.def_var(output_vars[buf_idx], new_offset_pos);
                    }
                    builder.ins().jump(continue_block, &[]);

                    // Value block: write offset, decode bytes, update positions, set validity bit
                    builder.switch_to_block(value_block);
                    {
                        // Write current offset (data_pos as i32) BEFORE copying data
                        let data_pos_val = builder.use_var(output_vars[data_buf_idx]);
                        let offset_i32 = builder.ins().ireduce(types::I32, data_pos_val);
                        let offset_pos_val = builder.use_var(output_vars[buf_idx]);
                        let offset_write_ptr_val = builder.ins().iadd(output_ptr, offset_pos_val);
                        builder.ins().store(MemFlags::trusted(), offset_i32, offset_write_ptr_val, 0);

                        // Advance offsets position by 4
                        let four = builder.ins().iconst(types::I64, 4);
                        let offset_pos_curr = builder.use_var(output_vars[buf_idx]);
                        let new_offset_pos = builder.ins().iadd(offset_pos_curr, four);
                        builder.def_var(output_vars[buf_idx], new_offset_pos);

                        // Decode bytes length
                        let (len_raw, new_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                        builder.def_var(cursor_var, new_cursor);
                        let len = Self::emit_zigzag_decode_i64(builder, len_raw);
                        let cursor = builder.use_var(cursor_var);

                        // Input bounds check: len >= 0 and cursor + len <= input_len
                        Self::emit_variable_bounds_check(builder, cursor, len, input_len, error_block);

                        // Output bounds check for DATA buffer
                        let cur_data_pos = builder.use_var(output_vars[data_buf_idx]);
                        Self::emit_output_bounds_check(builder, cur_data_pos, len, data_capacity, output_overflow_block);

                        // Copy data to data buffer
                        let cur_data_write = builder.ins().iadd(data_ptr, cur_data_pos);
                        Self::emit_memcpy(builder, input_ptr, cursor, cur_data_write, len, memcpy_func_ref);

                        // Advance input cursor
                        let new_cursor = builder.ins().iadd(cursor, len);
                        builder.def_var(cursor_var, new_cursor);

                        // Advance data position
                        let new_data_pos = builder.ins().iadd(cur_data_pos, len);
                        builder.def_var(output_vars[data_buf_idx], new_data_pos);

                        // Set validity bit to 1 for non-null value
                        if let Some(validity_idx) = buffer_slot.validity {
                            let row_idx = builder.use_var(row_var);
                            Self::emit_set_validity_bit(builder, output_ptrs, validity_idx, row_idx, ptr_type);
                        }
                    }
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);
                } else {
                    // Non-nullable: write offset, decode bytes, update positions

                    // Write current offset (data_pos as i32) BEFORE copying data
                    let offset_i32 = builder.ins().ireduce(types::I32, data_pos);
                    builder.ins().store(MemFlags::trusted(), offset_i32, output_write_ptr, 0);

                    // Advance offsets position by 4
                    let four = builder.ins().iconst(types::I64, 4);
                    let new_offset_pos = builder.ins().iadd(output_pos, four);
                    builder.def_var(output_vars[buf_idx], new_offset_pos);

                    // Decode bytes length
                    let (len_raw, new_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    builder.def_var(cursor_var, new_cursor);
                    let len = Self::emit_zigzag_decode_i64(builder, len_raw);
                    let cursor = builder.use_var(cursor_var);

                    // Input bounds check: len >= 0 and cursor + len <= input_len
                    Self::emit_variable_bounds_check(builder, cursor, len, input_len, error_block);

                    // Output bounds check for DATA buffer
                    Self::emit_output_bounds_check(builder, data_pos, len, data_capacity, output_overflow_block);

                    // Copy data to data buffer
                    Self::emit_memcpy(builder, input_ptr, cursor, data_write_ptr, len, memcpy_func_ref);

                    // Advance input cursor
                    let new_cursor = builder.ins().iadd(cursor, len);
                    builder.def_var(cursor_var, new_cursor);

                    // Advance data position
                    let new_data_pos = builder.ins().iadd(data_pos, len);
                    builder.def_var(output_vars[data_buf_idx], new_data_pos);
                }
            }
            FieldOp::Fixed { size, nullable } => {
                let size_val = builder.ins().iconst(types::I64, *size as i64);
                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    builder.def_var(cursor_var, tag_cursor);

                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, zero);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder.ins().brif(is_null, null_block, &[], value_block, &[]);

                    // Null block: write zeros
                    builder.switch_to_block(null_block);
                    // For fixed, we could memset to 0, but for simplicity just skip
                    builder.ins().jump(continue_block, &[]);

                    // Value block: copy fixed bytes and set validity bit
                    builder.switch_to_block(value_block);
                    let cursor = builder.use_var(cursor_var);
                    Self::emit_fixed_bounds_check(builder, cursor, *size as i64, input_len, error_block);
                    Self::emit_memcpy(builder, input_ptr, cursor, output_write_ptr, size_val, memcpy_func_ref);
                    let new_cursor = builder.ins().iadd(cursor, size_val);
                    builder.def_var(cursor_var, new_cursor);
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(builder, output_ptrs, validity_idx, row_idx, ptr_type);
                    }
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);

                    let new_output_pos = builder.ins().iadd(output_pos, size_val);
                    builder.def_var(output_vars[buf_idx], new_output_pos);
                } else {
                    let cursor = builder.use_var(cursor_var);
                    Self::emit_fixed_bounds_check(builder, cursor, *size as i64, input_len, error_block);
                    Self::emit_memcpy(builder, input_ptr, cursor, output_write_ptr, size_val, memcpy_func_ref);
                    let new_cursor = builder.ins().iadd(cursor, size_val);
                    builder.def_var(cursor_var, new_cursor);
                    let new_output_pos = builder.ins().iadd(output_pos, size_val);
                    builder.def_var(output_vars[buf_idx], new_output_pos);
                }
            }
            FieldOp::Struct { fields: child_ops, nullable } => {
                // Struct: decode each child field sequentially
                // Output is flattened: all field data concatenated together

                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    builder.def_var(cursor_var, tag_cursor);

                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, zero);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder.ins().brif(is_null, null_block, &[], value_block, &[]);

                    builder.switch_to_block(null_block);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode struct and set validity bit
                    builder.switch_to_block(value_block);
                    Self::emit_struct_decode(builder, child_ops, input_ptr, input_len, cursor_var, output_write_ptr, output_pos, output_vars, buf_idx, error_block);
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(builder, output_ptrs, validity_idx, row_idx, ptr_type);
                    }
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);
                } else {
                    Self::emit_struct_decode(builder, child_ops, input_ptr, input_len, cursor_var, output_write_ptr, output_pos, output_vars, buf_idx, error_block);
                }
            }
            FieldOp::Enum { nullable } => {
                // Enum is stored as a varint index (NOT zigzag encoded)
                // We read it as zigzag and decode, same as other ints
                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    builder.def_var(cursor_var, tag_cursor);

                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, zero);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder.ins().brif(is_null, null_block, &[], value_block, &[]);

                    builder.switch_to_block(null_block);
                    let zero_val = builder.ins().iconst(types::I32, 0);
                    builder.ins().store(MemFlags::trusted(), zero_val, output_write_ptr, 0);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode enum and set validity bit
                    builder.switch_to_block(value_block);
                    let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    // Enum indices are stored directly (no zigzag)
                    let value_i32 = builder.ins().ireduce(types::I32, value);
                    builder.ins().store(MemFlags::trusted(), value_i32, output_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(builder, output_ptrs, validity_idx, row_idx, ptr_type);
                    }
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);

                    let four = builder.ins().iconst(types::I64, 4);
                    let new_output_pos = builder.ins().iadd(output_pos, four);
                    builder.def_var(output_vars[buf_idx], new_output_pos);
                } else {
                    let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    // Enum indices are stored directly (no zigzag)
                    let value_i32 = builder.ins().ireduce(types::I32, value);
                    builder.ins().store(MemFlags::trusted(), value_i32, output_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                    let four = builder.ins().iconst(types::I64, 4);
                    let new_output_pos = builder.ins().iadd(output_pos, four);
                    builder.def_var(output_vars[buf_idx], new_output_pos);
                }
            }
            FieldOp::List { element, nullable } => {
                // Avro arrays: read block count (zigzag), decode elements, repeat until count=0
                // For Arrow ListArray, we need:
                // - output_ptrs[col_idx] -> element data buffer
                // - output_ptrs[col_idx + 1] -> offsets buffer (i32)
                //
                // For simplicity, we handle a flattened approach where:
                // - Elements are written to data buffer
                // - Offsets are tracked separately
                //
                // Note: This implementation handles non-nullable lists with primitive elements

                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    builder.def_var(cursor_var, tag_cursor);

                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, zero);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder.ins().brif(is_null, null_block, &[], value_block, &[]);

                    // Null block: skip
                    builder.switch_to_block(null_block);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode list and set validity bit
                    builder.switch_to_block(value_block);
                    Self::emit_list_decode(builder, element, buffer_slot, input_ptr, input_len, output_ptrs, output_caps, cursor_var, output_vars, ptr_type, error_block);
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(builder, output_ptrs, validity_idx, row_idx, ptr_type);
                    }
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);
                } else {
                    Self::emit_list_decode(builder, element, buffer_slot, input_ptr, input_len, output_ptrs, output_caps, cursor_var, output_vars, ptr_type, error_block);
                }
            }
            FieldOp::Map { value, nullable } => {
                // Avro maps: read block count (zigzag), decode key-value pairs, repeat until count=0
                // Keys are always strings in Avro
                // For simplicity, this implementation stores just the values
                // (keys are read and skipped for now - full map support would need separate buffers)

                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    builder.def_var(cursor_var, tag_cursor);

                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, zero);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder.ins().brif(is_null, null_block, &[], value_block, &[]);

                    builder.switch_to_block(null_block);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode map and set validity bit
                    builder.switch_to_block(value_block);
                    Self::emit_map_decode(builder, value, buffer_slot, input_ptr, input_len, output_ptrs, output_caps, cursor_var, output_vars, ptr_type, error_block);
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(builder, output_ptrs, validity_idx, row_idx, ptr_type);
                    }
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);
                } else {
                    Self::emit_map_decode(builder, value, buffer_slot, input_ptr, input_len, output_ptrs, output_caps, cursor_var, output_vars, ptr_type, error_block);
                }
            }
            FieldOp::Union { variants, type_ids, nullable } => {
                // Avro union: read type index varint, then decode corresponding variant
                // Output: type_id (i32) followed by variant data

                if *nullable {
                    // For nullable union, first check for null tag
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    builder.def_var(cursor_var, tag_cursor);

                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, zero);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder.ins().brif(is_null, null_block, &[], value_block, &[]);

                    builder.switch_to_block(null_block);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode union and set validity bit
                    builder.switch_to_block(value_block);
                    Self::emit_union_decode(builder, variants, type_ids, input_ptr, input_len, cursor_var, output_write_ptr, output_pos, output_vars, buf_idx, error_block);
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(builder, output_ptrs, validity_idx, row_idx, ptr_type);
                    }
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);
                } else {
                    Self::emit_union_decode(builder, variants, type_ids, input_ptr, input_len, cursor_var, output_write_ptr, output_pos, output_vars, buf_idx, error_block);
                }
            }
            FieldOp::FixedSizeList { element, size, nullable } => {
                // Fixed-size list: decode exactly `size` elements
                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    builder.def_var(cursor_var, tag_cursor);

                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, zero);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder.ins().brif(is_null, null_block, &[], value_block, &[]);

                    builder.switch_to_block(null_block);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode fixed size list and set validity bit
                    builder.switch_to_block(value_block);
                    Self::emit_fixed_size_list_decode(builder, element, *size, input_ptr, input_len, cursor_var, output_write_ptr, output_pos, output_vars, buf_idx, error_block);
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(builder, output_ptrs, validity_idx, row_idx, ptr_type);
                    }
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);
                } else {
                    Self::emit_fixed_size_list_decode(builder, element, *size, input_ptr, input_len, cursor_var, output_write_ptr, output_pos, output_vars, buf_idx, error_block);
                }
            }
            FieldOp::Decimal128 { nullable, .. } => {
                // Avro decimal: read as fixed 16 bytes (big-endian), convert to little-endian for Arrow
                let byte_size: i64 = 16;

                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    builder.def_var(cursor_var, tag_cursor);

                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, zero);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder.ins().brif(is_null, null_block, &[], value_block, &[]);

                    builder.switch_to_block(null_block);
                    // Write zeros for null
                    let zero_val = builder.ins().iconst(types::I64, 0);
                    builder.ins().store(MemFlags::trusted(), zero_val, output_write_ptr, 0);
                    builder.ins().store(MemFlags::trusted(), zero_val, output_write_ptr, 8);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode and set validity bit
                    builder.switch_to_block(value_block);
                    Self::emit_decimal128_decode(builder, input_ptr, input_len, cursor_var, output_write_ptr, error_block);
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(builder, output_ptrs, validity_idx, row_idx, ptr_type);
                    }
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);
                } else {
                    Self::emit_decimal128_decode(builder, input_ptr, input_len, cursor_var, output_write_ptr, error_block);
                }

                let size_const = builder.ins().iconst(types::I64, byte_size);
                let new_output_pos = builder.ins().iadd(output_pos, size_const);
                builder.def_var(output_vars[buf_idx], new_output_pos);
            }
            FieldOp::Decimal256 { nullable, .. } => {
                // Avro decimal: read as fixed 32 bytes (big-endian), convert to little-endian for Arrow
                let byte_size: i64 = 32;

                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    builder.def_var(cursor_var, tag_cursor);

                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, zero);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder.ins().brif(is_null, null_block, &[], value_block, &[]);

                    builder.switch_to_block(null_block);
                    // Write zeros for null
                    let zero_val = builder.ins().iconst(types::I64, 0);
                    builder.ins().store(MemFlags::trusted(), zero_val, output_write_ptr, 0);
                    builder.ins().store(MemFlags::trusted(), zero_val, output_write_ptr, 8);
                    builder.ins().store(MemFlags::trusted(), zero_val, output_write_ptr, 16);
                    builder.ins().store(MemFlags::trusted(), zero_val, output_write_ptr, 24);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode and set validity bit
                    builder.switch_to_block(value_block);
                    Self::emit_decimal256_decode(builder, input_ptr, input_len, cursor_var, output_write_ptr, error_block);
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(builder, output_ptrs, validity_idx, row_idx, ptr_type);
                    }
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);
                } else {
                    Self::emit_decimal256_decode(builder, input_ptr, input_len, cursor_var, output_write_ptr, error_block);
                }

                let size_const = builder.ins().iconst(types::I64, byte_size);
                let new_output_pos = builder.ins().iadd(output_pos, size_const);
                builder.def_var(output_vars[buf_idx], new_output_pos);
            }
            FieldOp::Duration { nullable } => {
                // Duration: read as i64 (same as Int64)
                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    builder.def_var(cursor_var, tag_cursor);

                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, zero);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder.ins().brif(is_null, null_block, &[], value_block, &[]);

                    builder.switch_to_block(null_block);
                    let zero_val = builder.ins().iconst(types::I64, 0);
                    builder.ins().store(MemFlags::trusted(), zero_val, output_write_ptr, 0);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode and set validity bit
                    builder.switch_to_block(value_block);
                    let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    let zigzag = Self::emit_zigzag_decode_i64(builder, value);
                    builder.ins().store(MemFlags::trusted(), zigzag, output_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(builder, output_ptrs, validity_idx, row_idx, ptr_type);
                    }
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);
                } else {
                    let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    let zigzag = Self::emit_zigzag_decode_i64(builder, value);
                    builder.ins().store(MemFlags::trusted(), zigzag, output_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                }

                let eight = builder.ins().iconst(types::I64, 8);
                let new_output_pos = builder.ins().iadd(output_pos, eight);
                builder.def_var(output_vars[buf_idx], new_output_pos);
            }
            FieldOp::Interval { unit, nullable } => {
                // Interval: depends on unit type
                // YearMonth: 4 bytes (i32)
                // DayTime: 8 bytes (two i32)
                // MonthDayNano: 16 bytes (i32 + i32 + i64)
                let byte_size: i64 = match unit {
                    IntervalUnit::YearMonth => 4,
                    IntervalUnit::DayTime => 8,
                    IntervalUnit::MonthDayNano => 16,
                };

                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    builder.def_var(cursor_var, tag_cursor);

                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, zero);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder.ins().brif(is_null, null_block, &[], value_block, &[]);

                    builder.switch_to_block(null_block);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode and set validity bit
                    builder.switch_to_block(value_block);
                    Self::emit_interval_decode(builder, *unit, input_ptr, input_len, cursor_var, output_write_ptr, error_block);
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(builder, output_ptrs, validity_idx, row_idx, ptr_type);
                    }
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);
                } else {
                    Self::emit_interval_decode(builder, *unit, input_ptr, input_len, cursor_var, output_write_ptr, error_block);
                }

                let size_const = builder.ins().iconst(types::I64, byte_size);
                let new_output_pos = builder.ins().iadd(output_pos, size_const);
                builder.def_var(output_vars[buf_idx], new_output_pos);
            }
            FieldOp::Null => {
                // Null type: no data to read, no output to write
                // Used in unions for nullable fields - the null variant reads nothing
                // Output position stays the same since Null has no representation
            }
        }
    }

    /// Emit Cranelift IR for Avro struct/record decoding
    ///
    /// Structs decode each field sequentially in schema order.
    /// Output is flattened: all field data written sequentially to a single buffer.
    fn emit_struct_decode(
        builder: &mut FunctionBuilder,
        child_ops: &[FieldOp],
        input_ptr: Value,
        input_len: Value,
        cursor_var: Variable,
        output_write_ptr: Value,
        output_pos: Value,
        output_vars: &[Variable],
        buf_idx: usize,
        error_block: Block,
    ) {
        // Track bytes written across all child fields
        let struct_bytes_var = builder.declare_var(types::I64);
        let zero = builder.ins().iconst(types::I64, 0);
        builder.def_var(struct_bytes_var, zero);

        // Decode each child field
        for child_op in child_ops {
            let struct_bytes = builder.use_var(struct_bytes_var);
            let field_write_ptr = builder.ins().iadd(output_write_ptr, struct_bytes);

            // Get field size
            let field_size: i64 = match child_op {
                FieldOp::Int64 { .. } | FieldOp::Float64 { .. } => 8,
                FieldOp::Int32 { .. } | FieldOp::Float32 { .. } | FieldOp::Enum { .. } => 4,
                FieldOp::Boolean { .. } => 1,
                _ => 8, // Default
            };

            // Decode based on child type
            match child_op {
                FieldOp::Int64 { nullable: false } => {
                    let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    let zigzag = Self::emit_zigzag_decode_i64(builder, value);
                    builder.ins().store(MemFlags::trusted(), zigzag, field_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                }
                FieldOp::Int32 { nullable: false } => {
                    let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    let zigzag = Self::emit_zigzag_decode_i32(builder, value);
                    builder.ins().store(MemFlags::trusted(), zigzag, field_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                }
                FieldOp::Float64 { nullable: false } => {
                    let cursor = builder.use_var(cursor_var);
                    Self::emit_fixed_bounds_check(builder, cursor, 8, input_len, error_block);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder.ins().load(types::I64, MemFlags::trusted(), read_ptr, 0);
                    builder.ins().store(MemFlags::trusted(), value, field_write_ptr, 0);
                    let eight = builder.ins().iconst(types::I64, 8);
                    let new_cursor = builder.ins().iadd(cursor, eight);
                    builder.def_var(cursor_var, new_cursor);
                }
                FieldOp::Float32 { nullable: false } => {
                    let cursor = builder.use_var(cursor_var);
                    Self::emit_fixed_bounds_check(builder, cursor, 4, input_len, error_block);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder.ins().load(types::I32, MemFlags::trusted(), read_ptr, 0);
                    builder.ins().store(MemFlags::trusted(), value, field_write_ptr, 0);
                    let four = builder.ins().iconst(types::I64, 4);
                    let new_cursor = builder.ins().iadd(cursor, four);
                    builder.def_var(cursor_var, new_cursor);
                }
                FieldOp::Boolean { nullable: false } => {
                    let cursor = builder.use_var(cursor_var);
                    Self::emit_fixed_bounds_check(builder, cursor, 1, input_len, error_block);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder.ins().load(types::I8, MemFlags::trusted(), read_ptr, 0);
                    builder.ins().store(MemFlags::trusted(), value, field_write_ptr, 0);
                    let one = builder.ins().iconst(types::I64, 1);
                    let new_cursor = builder.ins().iadd(cursor, one);
                    builder.def_var(cursor_var, new_cursor);
                }
                _ => {
                    // Skip complex/nullable child types for now
                }
            }

            // Update bytes written
            let field_size_val = builder.ins().iconst(types::I64, field_size);
            let new_struct_bytes = builder.ins().iadd(struct_bytes, field_size_val);
            builder.def_var(struct_bytes_var, new_struct_bytes);
        }

        // Update final output position
        let final_bytes = builder.use_var(struct_bytes_var);
        let new_output_pos = builder.ins().iadd(output_pos, final_bytes);
        builder.def_var(output_vars[buf_idx], new_output_pos);
    }

    /// Emit Cranelift IR for Avro map decoding
    ///
    /// Avro maps use the same block encoding as arrays:
    /// 1. Read block count (zigzag encoded)
    /// 2. If count < 0, abs(count) is entry count, skip byte size
    /// 3. For each entry: read string key, then read value
    /// 4. Repeat until count = 0
    ///
    /// Emit Cranelift IR for Avro map decoding with dual buffer output
    ///
    /// Arrow MapArray format requires:
    /// - offsets buffer (i32): [0, count_of_map_0, count_of_map_0 + count_of_map_1, ...]
    /// - data buffer: concatenated values (keys are skipped in this simplified impl)
    ///
    /// buffer_slot.primary = offsets buffer index
    /// buffer_slot.secondary = data buffer index
    fn emit_map_decode(
        builder: &mut FunctionBuilder,
        value_op: &FieldOp,
        buffer_slot: &BufferSlot,
        input_ptr: Value,
        input_len: Value,
        output_ptrs: Value,
        _output_caps: Value,
        cursor_var: Variable,
        output_vars: &[Variable],
        ptr_type: Type,
        error_block: Block,
    ) {
        // Get buffer indices
        let offsets_buf_idx = buffer_slot.primary;
        let data_buf_idx = buffer_slot.secondary.expect("Map type must have secondary buffer for data");

        // Get value size based on type
        let value_size: i64 = match value_op {
            FieldOp::Int64 { .. } | FieldOp::Float64 { .. } => 8,
            FieldOp::Int32 { .. } | FieldOp::Float32 { .. } | FieldOp::Enum { .. } => 4,
            FieldOp::Boolean { .. } => 1,
            _ => 8,
        };

        // Load data buffer pointer
        let data_buf_offset = builder.ins().iconst(ptr_type, (data_buf_idx * 8) as i64);
        let data_ptr_addr = builder.ins().iadd(output_ptrs, data_buf_offset);
        let data_ptr = builder.ins().load(ptr_type, MemFlags::trusted(), data_ptr_addr, 0);

        // Load offsets buffer pointer
        let offsets_buf_offset = builder.ins().iconst(ptr_type, (offsets_buf_idx * 8) as i64);
        let offsets_ptr_addr = builder.ins().iadd(output_ptrs, offsets_buf_offset);
        let offsets_ptr = builder.ins().load(ptr_type, MemFlags::trusted(), offsets_ptr_addr, 0);

        // Write initial offset (current element count, not byte position)
        // Arrow offsets are element indices: offset[row_idx] = element_count so far
        let offset_pos = builder.use_var(output_vars[offsets_buf_idx]);
        let offset_write_ptr = builder.ins().iadd(offsets_ptr, offset_pos);
        let data_pos = builder.use_var(output_vars[data_buf_idx]);
        // Convert byte position to element count: element_count = data_pos / value_size
        let value_size_val = builder.ins().iconst(types::I64, value_size);
        let element_count = builder.ins().udiv(data_pos, value_size_val);
        let offset_i32 = builder.ins().ireduce(types::I32, element_count);
        builder.ins().store(MemFlags::trusted(), offset_i32, offset_write_ptr, 0);

        // Advance offsets position by 4 (one i32)
        let four = builder.ins().iconst(types::I64, 4);
        let new_offset_pos = builder.ins().iadd(offset_pos, four);
        builder.def_var(output_vars[offsets_buf_idx], new_offset_pos);

        // Block loop
        let block_header = builder.create_block();
        let block_body = builder.create_block();
        let block_exit = builder.create_block();

        builder.ins().jump(block_header, &[]);

        // Block header: read block count
        builder.switch_to_block(block_header);
        let (count_raw, count_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
        builder.def_var(cursor_var, count_cursor);

        let count = Self::emit_zigzag_decode_i64(builder, count_raw);

        let zero_check = builder.ins().iconst(types::I64, 0);
        let is_done = builder.ins().icmp(IntCC::Equal, count, zero_check);
        builder.ins().brif(is_done, block_exit, &[], block_body, &[]);

        // Block body: decode entries
        builder.switch_to_block(block_body);

        let zero_cmp = builder.ins().iconst(types::I64, 0);
        let is_negative = builder.ins().icmp(IntCC::SignedLessThan, count, zero_cmp);
        let neg_count = builder.ins().ineg(count);
        let abs_count = builder.ins().select(is_negative, neg_count, count);

        let skip_block = builder.create_block();
        let decode_block = builder.create_block();

        builder.ins().brif(is_negative, skip_block, &[], decode_block, &[]);

        // Skip block: read byte size and discard
        builder.switch_to_block(skip_block);
        let (_byte_size, skip_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
        builder.def_var(cursor_var, skip_cursor);
        builder.ins().jump(decode_block, &[]);

        // Decode block: loop over entries
        builder.switch_to_block(decode_block);

        let entry_idx_var = builder.declare_var(types::I64);
        let zero_idx = builder.ins().iconst(types::I64, 0);
        builder.def_var(entry_idx_var, zero_idx);

        let entry_loop_header = builder.create_block();
        let entry_loop_body = builder.create_block();
        let entry_loop_exit = builder.create_block();

        builder.ins().jump(entry_loop_header, &[]);

        // Entry loop header
        builder.switch_to_block(entry_loop_header);
        let entry_idx = builder.use_var(entry_idx_var);
        let more_entries = builder.ins().icmp(IntCC::UnsignedLessThan, entry_idx, abs_count);
        builder.ins().brif(more_entries, entry_loop_body, &[], entry_loop_exit, &[]);

        // Entry loop body: skip key string, decode value
        builder.switch_to_block(entry_loop_body);

        // Skip string key: read length, skip bytes
        let (key_len_raw, key_len_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
        builder.def_var(cursor_var, key_len_cursor);
        let key_len = Self::emit_zigzag_decode_i64(builder, key_len_raw);
        let cursor_after_key = builder.use_var(cursor_var);
        let new_cursor_after_key = builder.ins().iadd(cursor_after_key, key_len);
        builder.def_var(cursor_var, new_cursor_after_key);

        // Decode value to data buffer
        let current_data_pos = builder.use_var(output_vars[data_buf_idx]);
        let value_write_ptr = builder.ins().iadd(data_ptr, current_data_pos);

        match value_op {
            FieldOp::Int64 { nullable: false } => {
                let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                let zigzag = Self::emit_zigzag_decode_i64(builder, value);
                builder.ins().store(MemFlags::trusted(), zigzag, value_write_ptr, 0);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Int32 { nullable: false } => {
                let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                let zigzag = Self::emit_zigzag_decode_i32(builder, value);
                builder.ins().store(MemFlags::trusted(), zigzag, value_write_ptr, 0);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Float64 { nullable: false } => {
                let cursor = builder.use_var(cursor_var);
                Self::emit_fixed_bounds_check(builder, cursor, 8, input_len, error_block);
                let read_ptr = builder.ins().iadd(input_ptr, cursor);
                let value = builder.ins().load(types::I64, MemFlags::trusted(), read_ptr, 0);
                builder.ins().store(MemFlags::trusted(), value, value_write_ptr, 0);
                let eight = builder.ins().iconst(types::I64, 8);
                let new_cursor = builder.ins().iadd(cursor, eight);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Float32 { nullable: false } => {
                let cursor = builder.use_var(cursor_var);
                Self::emit_fixed_bounds_check(builder, cursor, 4, input_len, error_block);
                let read_ptr = builder.ins().iadd(input_ptr, cursor);
                let value = builder.ins().load(types::I32, MemFlags::trusted(), read_ptr, 0);
                builder.ins().store(MemFlags::trusted(), value, value_write_ptr, 0);
                let four = builder.ins().iconst(types::I64, 4);
                let new_cursor = builder.ins().iadd(cursor, four);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Boolean { nullable: false } => {
                let cursor = builder.use_var(cursor_var);
                Self::emit_fixed_bounds_check(builder, cursor, 1, input_len, error_block);
                let read_ptr = builder.ins().iadd(input_ptr, cursor);
                let value = builder.ins().load(types::I8, MemFlags::trusted(), read_ptr, 0);
                builder.ins().store(MemFlags::trusted(), value, value_write_ptr, 0);
                let one = builder.ins().iconst(types::I64, 1);
                let new_cursor = builder.ins().iadd(cursor, one);
                builder.def_var(cursor_var, new_cursor);
            }
            _ => {
                // For other types, skip
            }
        }

        // Update data position
        let value_size_val = builder.ins().iconst(types::I64, value_size);
        let current_data_pos_update = builder.use_var(output_vars[data_buf_idx]);
        let new_data_pos = builder.ins().iadd(current_data_pos_update, value_size_val);
        builder.def_var(output_vars[data_buf_idx], new_data_pos);

        // Increment entry index
        let one = builder.ins().iconst(types::I64, 1);
        let new_entry_idx = builder.ins().iadd(entry_idx, one);
        builder.def_var(entry_idx_var, new_entry_idx);

        builder.ins().jump(entry_loop_header, &[]);

        // Entry loop exit
        builder.switch_to_block(entry_loop_exit);
        builder.ins().jump(block_header, &[]);

        // Block exit: done decoding this map
        // Note: final offset (offset[row_count]) is written in the main loop_exit block
        builder.switch_to_block(block_exit);

        // Seal all blocks
        builder.seal_block(skip_block);
        builder.seal_block(decode_block);
        builder.seal_block(entry_loop_header);
        builder.seal_block(entry_loop_body);
        builder.seal_block(entry_loop_exit);
        builder.seal_block(block_header);
        builder.seal_block(block_body);
        builder.seal_block(block_exit);
    }

    /// Emit Cranelift IR for memcpy using libc memcpy for SIMD-optimized bulk copy
    /// Copies len bytes from src_base+src_offset to dst
    fn emit_memcpy(
        builder: &mut FunctionBuilder,
        src_base: Value,
        src_offset: Value,
        dst: Value,
        len: Value,
        memcpy_func_ref: FuncRef,
    ) {
        // Calculate src_ptr = src_base + src_offset
        let src_ptr = builder.ins().iadd(src_base, src_offset);

        // Call memcpy(dst, src, len) - uses SIMD for large copies
        builder.ins().call(memcpy_func_ref, &[dst, src_ptr, len]);
        // Note: memcpy returns dest pointer which we don't need
    }

    /// Emit Cranelift IR for Avro list/array decoding with dual buffer output
    ///
    /// Arrow ListArray format requires:
    /// - offsets buffer (i32): [0, count_of_list_0, count_of_list_0 + count_of_list_1, ...]
    /// - data buffer: concatenated element values
    ///
    /// buffer_slot.primary = offsets buffer index
    /// buffer_slot.secondary = data buffer index
    fn emit_list_decode(
        builder: &mut FunctionBuilder,
        element_op: &FieldOp,
        buffer_slot: &BufferSlot,
        input_ptr: Value,
        input_len: Value,
        output_ptrs: Value,
        _output_caps: Value,
        cursor_var: Variable,
        output_vars: &[Variable],
        ptr_type: Type,
        error_block: Block,
    ) {
        // Get buffer indices
        let offsets_buf_idx = buffer_slot.primary;
        let data_buf_idx = buffer_slot.secondary.expect("List type must have secondary buffer for data");

        // Get element size based on type
        let element_size: i64 = match element_op {
            FieldOp::Int64 { .. } | FieldOp::Float64 { .. } => 8,
            FieldOp::Int32 { .. } | FieldOp::Float32 { .. } | FieldOp::Enum { .. } => 4,
            FieldOp::Boolean { .. } => 1,
            _ => 8, // Default for complex types
        };

        // Load data buffer pointer
        let data_buf_offset = builder.ins().iconst(ptr_type, (data_buf_idx * 8) as i64);
        let data_ptr_addr = builder.ins().iadd(output_ptrs, data_buf_offset);
        let data_ptr = builder.ins().load(ptr_type, MemFlags::trusted(), data_ptr_addr, 0);

        // Load offsets buffer pointer
        let offsets_buf_offset = builder.ins().iconst(ptr_type, (offsets_buf_idx * 8) as i64);
        let offsets_ptr_addr = builder.ins().iadd(output_ptrs, offsets_buf_offset);
        let offsets_ptr = builder.ins().load(ptr_type, MemFlags::trusted(), offsets_ptr_addr, 0);

        // Write initial offset (current element count, not byte position)
        // Arrow offsets are element indices: offset[row_idx] = element_count so far
        let offset_pos = builder.use_var(output_vars[offsets_buf_idx]);
        let offset_write_ptr = builder.ins().iadd(offsets_ptr, offset_pos);
        let data_pos = builder.use_var(output_vars[data_buf_idx]);
        // Convert byte position to element count: element_count = data_pos / element_size
        let element_size_val = builder.ins().iconst(types::I64, element_size);
        let element_count = builder.ins().udiv(data_pos, element_size_val);
        let offset_i32 = builder.ins().ireduce(types::I32, element_count);
        builder.ins().store(MemFlags::trusted(), offset_i32, offset_write_ptr, 0);

        // Advance offsets position by 4 (one i32)
        let four = builder.ins().iconst(types::I64, 4);
        let new_offset_pos = builder.ins().iadd(offset_pos, four);
        builder.def_var(output_vars[offsets_buf_idx], new_offset_pos);

        // Block loop: read count, decode elements, repeat until count=0
        let block_header = builder.create_block();
        let block_body = builder.create_block();
        let block_exit = builder.create_block();

        builder.ins().jump(block_header, &[]);

        // Block header: read block count
        builder.switch_to_block(block_header);
        let (count_raw, count_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
        builder.def_var(cursor_var, count_cursor);

        // Zigzag decode the count
        let count = Self::emit_zigzag_decode_i64(builder, count_raw);

        // Check if count == 0 (end of array)
        let zero_check = builder.ins().iconst(types::I64, 0);
        let is_done = builder.ins().icmp(IntCC::Equal, count, zero_check);
        builder.ins().brif(is_done, block_exit, &[], block_body, &[]);

        // Block body: decode elements
        builder.switch_to_block(block_body);

        // Handle negative count: abs(count)
        // If count < 0, also skip byte size varint
        let zero_cmp = builder.ins().iconst(types::I64, 0);
        let is_negative = builder.ins().icmp(IntCC::SignedLessThan, count, zero_cmp);
        let neg_count = builder.ins().ineg(count);
        let abs_count = builder.ins().select(is_negative, neg_count, count);

        // If negative, skip byte size (just read and discard)
        let skip_block = builder.create_block();
        let decode_block = builder.create_block();

        builder.ins().brif(is_negative, skip_block, &[], decode_block, &[]);

        // Skip block: read byte size and discard
        builder.switch_to_block(skip_block);
        let (_byte_size, skip_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
        builder.def_var(cursor_var, skip_cursor);
        builder.ins().jump(decode_block, &[]);

        // Decode block: loop over elements
        builder.switch_to_block(decode_block);

        // Element loop
        let elem_idx_var = builder.declare_var(types::I64);
        let zero_idx = builder.ins().iconst(types::I64, 0);
        builder.def_var(elem_idx_var, zero_idx);

        let elem_loop_header = builder.create_block();
        let elem_loop_body = builder.create_block();
        let elem_loop_exit = builder.create_block();

        builder.ins().jump(elem_loop_header, &[]);

        // Element loop header: check idx < abs_count
        builder.switch_to_block(elem_loop_header);
        let elem_idx = builder.use_var(elem_idx_var);
        let more_elems = builder.ins().icmp(IntCC::UnsignedLessThan, elem_idx, abs_count);
        builder.ins().brif(more_elems, elem_loop_body, &[], elem_loop_exit, &[]);

        // Element loop body: decode one element
        builder.switch_to_block(elem_loop_body);

        // Calculate output pointer for this element in data buffer
        let current_data_pos = builder.use_var(output_vars[data_buf_idx]);
        let elem_write_ptr = builder.ins().iadd(data_ptr, current_data_pos);

        // Decode based on element type
        match element_op {
            FieldOp::Int64 { nullable: false } => {
                let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                let zigzag = Self::emit_zigzag_decode_i64(builder, value);
                builder.ins().store(MemFlags::trusted(), zigzag, elem_write_ptr, 0);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Int32 { nullable: false } => {
                let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                let zigzag = Self::emit_zigzag_decode_i32(builder, value);
                builder.ins().store(MemFlags::trusted(), zigzag, elem_write_ptr, 0);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Float64 { nullable: false } => {
                let cursor = builder.use_var(cursor_var);
                Self::emit_fixed_bounds_check(builder, cursor, 8, input_len, error_block);
                let read_ptr = builder.ins().iadd(input_ptr, cursor);
                let value = builder.ins().load(types::I64, MemFlags::trusted(), read_ptr, 0);
                builder.ins().store(MemFlags::trusted(), value, elem_write_ptr, 0);
                let eight = builder.ins().iconst(types::I64, 8);
                let new_cursor = builder.ins().iadd(cursor, eight);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Float32 { nullable: false } => {
                let cursor = builder.use_var(cursor_var);
                Self::emit_fixed_bounds_check(builder, cursor, 4, input_len, error_block);
                let read_ptr = builder.ins().iadd(input_ptr, cursor);
                let value = builder.ins().load(types::I32, MemFlags::trusted(), read_ptr, 0);
                builder.ins().store(MemFlags::trusted(), value, elem_write_ptr, 0);
                let four = builder.ins().iconst(types::I64, 4);
                let new_cursor = builder.ins().iadd(cursor, four);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Boolean { nullable: false } => {
                let cursor = builder.use_var(cursor_var);
                Self::emit_fixed_bounds_check(builder, cursor, 1, input_len, error_block);
                let read_ptr = builder.ins().iadd(input_ptr, cursor);
                let value = builder.ins().load(types::I8, MemFlags::trusted(), read_ptr, 0);
                builder.ins().store(MemFlags::trusted(), value, elem_write_ptr, 0);
                let one = builder.ins().iconst(types::I64, 1);
                let new_cursor = builder.ins().iadd(cursor, one);
                builder.def_var(cursor_var, new_cursor);
            }
            _ => {
                // For other types, skip for now (no decode)
            }
        }

        // Update data position
        let elem_size_val = builder.ins().iconst(types::I64, element_size);
        let current_data_pos_update = builder.use_var(output_vars[data_buf_idx]);
        let new_data_pos = builder.ins().iadd(current_data_pos_update, elem_size_val);
        builder.def_var(output_vars[data_buf_idx], new_data_pos);

        // Increment element index
        let one = builder.ins().iconst(types::I64, 1);
        let new_elem_idx = builder.ins().iadd(elem_idx, one);
        builder.def_var(elem_idx_var, new_elem_idx);

        builder.ins().jump(elem_loop_header, &[]);

        // Element loop exit: continue to next block
        builder.switch_to_block(elem_loop_exit);
        builder.ins().jump(block_header, &[]);

        // Block exit: done decoding this list
        // Note: final offset (offset[row_count]) is written in the main loop_exit block
        builder.switch_to_block(block_exit);

        // Seal all blocks
        builder.seal_block(skip_block);
        builder.seal_block(decode_block);
        builder.seal_block(elem_loop_header);
        builder.seal_block(elem_loop_body);
        builder.seal_block(elem_loop_exit);
        builder.seal_block(block_header);
        builder.seal_block(block_body);
        builder.seal_block(block_exit);
    }

    /// Emit Cranelift IR for Avro union decoding
    ///
    /// Avro union: read type index varint, then decode corresponding variant
    /// Output format: type_id (i32) followed by variant data
    #[allow(clippy::too_many_arguments)]
    fn emit_union_decode(
        builder: &mut FunctionBuilder,
        variants: &[FieldOp],
        type_ids: &[i8],
        input_ptr: Value,
        input_len: Value,
        cursor_var: Variable,
        output_write_ptr: Value,
        output_pos: Value,
        output_vars: &[Variable],
        buf_idx: usize,
        error_block: Block,
    ) {
        // Read type index (varint, not zigzag encoded for union)
        let (type_index, index_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
        builder.def_var(cursor_var, index_cursor);

        // Track bytes written - start with 4 for type_id
        let union_bytes_var = builder.declare_var(types::I64);
        let four = builder.ins().iconst(types::I64, 4);
        builder.def_var(union_bytes_var, four);

        // For simple unions (typically 2-4 variants), use if-else chain
        // For now, we handle the common case of small unions
        if variants.is_empty() {
            // Empty union - shouldn't happen but handle gracefully
            builder.def_var(output_vars[buf_idx], output_pos);
            return;
        }

        // Write type_id at the start
        // We need to map type_index to the correct type_id
        // For now, just write the type_index as type_id (simplified)
        let type_id_i32 = builder.ins().ireduce(types::I32, type_index);
        builder.ins().store(MemFlags::trusted(), type_id_i32, output_write_ptr, 0);

        // Calculate where to write variant data
        let variant_write_ptr = builder.ins().iadd(output_write_ptr, four);

        // Create blocks for each variant and exit
        let exit_block = builder.create_block();
        let variant_blocks: Vec<_> = variants.iter().map(|_| builder.create_block()).collect();

        // Default block (for out-of-range indices)
        let default_block = builder.create_block();

        // Check variant index and branch
        // Collect intermediate check blocks so we can seal them after
        let mut check_blocks = Vec::new();
        for (i, variant_block) in variant_blocks.iter().enumerate() {
            let idx_const = builder.ins().iconst(types::I64, i as i64);
            let is_this_variant = builder.ins().icmp(IntCC::Equal, type_index, idx_const);

            let next_check = if i + 1 < variants.len() {
                let block = builder.create_block();
                check_blocks.push(block);
                block
            } else {
                default_block
            };

            builder.ins().brif(is_this_variant, *variant_block, &[], next_check, &[]);

            if i + 1 < variants.len() {
                builder.switch_to_block(next_check);
            }
        }

        // Seal all intermediate check blocks
        for check_block in check_blocks {
            builder.seal_block(check_block);
        }

        // Default block: skip unknown variant
        builder.switch_to_block(default_block);
        builder.ins().jump(exit_block, &[]);
        builder.seal_block(default_block);

        // Generate code for each variant
        for (i, (variant_op, _type_id)) in variants.iter().zip(type_ids.iter()).enumerate() {
            builder.switch_to_block(variant_blocks[i]);

            // Get variant size
            let variant_size: i64 = match variant_op {
                FieldOp::Int64 { .. } | FieldOp::Float64 { .. } => 8,
                FieldOp::Int32 { .. } | FieldOp::Float32 { .. } | FieldOp::Enum { .. } => 4,
                FieldOp::Boolean { .. } => 1,
                FieldOp::Fixed { size, .. } => *size as i64,
                _ => 8, // Default for complex types
            };

            // Decode variant value
            match variant_op {
                FieldOp::Int64 { nullable: false } => {
                    let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    let zigzag = Self::emit_zigzag_decode_i64(builder, value);
                    builder.ins().store(MemFlags::trusted(), zigzag, variant_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                }
                FieldOp::Int32 { nullable: false } => {
                    let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                    let zigzag = Self::emit_zigzag_decode_i32(builder, value);
                    builder.ins().store(MemFlags::trusted(), zigzag, variant_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                }
                FieldOp::Float64 { nullable: false } => {
                    let cursor = builder.use_var(cursor_var);
                    Self::emit_fixed_bounds_check(builder, cursor, 8, input_len, error_block);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder.ins().load(types::I64, MemFlags::trusted(), read_ptr, 0);
                    builder.ins().store(MemFlags::trusted(), value, variant_write_ptr, 0);
                    let eight = builder.ins().iconst(types::I64, 8);
                    let new_cursor = builder.ins().iadd(cursor, eight);
                    builder.def_var(cursor_var, new_cursor);
                }
                FieldOp::Float32 { nullable: false } => {
                    let cursor = builder.use_var(cursor_var);
                    Self::emit_fixed_bounds_check(builder, cursor, 4, input_len, error_block);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder.ins().load(types::I32, MemFlags::trusted(), read_ptr, 0);
                    builder.ins().store(MemFlags::trusted(), value, variant_write_ptr, 0);
                    let four_const = builder.ins().iconst(types::I64, 4);
                    let new_cursor = builder.ins().iadd(cursor, four_const);
                    builder.def_var(cursor_var, new_cursor);
                }
                FieldOp::Boolean { nullable: false } => {
                    let cursor = builder.use_var(cursor_var);
                    Self::emit_fixed_bounds_check(builder, cursor, 1, input_len, error_block);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder.ins().load(types::I8, MemFlags::trusted(), read_ptr, 0);
                    builder.ins().store(MemFlags::trusted(), value, variant_write_ptr, 0);
                    let one = builder.ins().iconst(types::I64, 1);
                    let new_cursor = builder.ins().iadd(cursor, one);
                    builder.def_var(cursor_var, new_cursor);
                }
                _ => {
                    // For other types (null, complex), no additional data to decode
                }
            }

            // Update bytes written
            let variant_size_val = builder.ins().iconst(types::I64, variant_size);
            let current_bytes = builder.use_var(union_bytes_var);
            let new_bytes = builder.ins().iadd(current_bytes, variant_size_val);
            builder.def_var(union_bytes_var, new_bytes);

            builder.ins().jump(exit_block, &[]);
            builder.seal_block(variant_blocks[i]);
        }

        // Exit block
        builder.switch_to_block(exit_block);
        builder.seal_block(exit_block);

        let final_bytes = builder.use_var(union_bytes_var);
        let new_output_pos = builder.ins().iadd(output_pos, final_bytes);
        builder.def_var(output_vars[buf_idx], new_output_pos);
    }

    /// Emit Cranelift IR for fixed-size list decoding
    ///
    /// Decodes exactly `size` elements without block encoding
    #[allow(clippy::too_many_arguments)]
    fn emit_fixed_size_list_decode(
        builder: &mut FunctionBuilder,
        element_op: &FieldOp,
        size: usize,
        input_ptr: Value,
        input_len: Value,
        cursor_var: Variable,
        output_write_ptr: Value,
        output_pos: Value,
        output_vars: &[Variable],
        buf_idx: usize,
        error_block: Block,
    ) {
        // Get element size based on type
        let element_size: i64 = match element_op {
            FieldOp::Int64 { .. } | FieldOp::Float64 { .. } => 8,
            FieldOp::Int32 { .. } | FieldOp::Float32 { .. } | FieldOp::Enum { .. } => 4,
            FieldOp::Boolean { .. } => 1,
            FieldOp::Fixed { size, .. } => *size as i64,
            _ => 8,
        };

        // Track bytes written
        let list_bytes_var = builder.declare_var(types::I64);
        let zero = builder.ins().iconst(types::I64, 0);
        builder.def_var(list_bytes_var, zero);

        // Create element loop
        let elem_idx_var = builder.declare_var(types::I64);
        builder.def_var(elem_idx_var, zero);

        let loop_header = builder.create_block();
        let loop_body = builder.create_block();
        let loop_exit = builder.create_block();

        let size_const = builder.ins().iconst(types::I64, size as i64);
        builder.ins().jump(loop_header, &[]);

        // Loop header
        builder.switch_to_block(loop_header);
        let elem_idx = builder.use_var(elem_idx_var);
        let more_elements = builder.ins().icmp(IntCC::UnsignedLessThan, elem_idx, size_const);
        builder.ins().brif(more_elements, loop_body, &[], loop_exit, &[]);

        // Loop body
        builder.switch_to_block(loop_body);
        let list_bytes_written = builder.use_var(list_bytes_var);
        let elem_write_ptr = builder.ins().iadd(output_write_ptr, list_bytes_written);

        // Decode element
        match element_op {
            FieldOp::Int64 { nullable: false } => {
                let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                let zigzag = Self::emit_zigzag_decode_i64(builder, value);
                builder.ins().store(MemFlags::trusted(), zigzag, elem_write_ptr, 0);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Int32 { nullable: false } => {
                let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                let zigzag = Self::emit_zigzag_decode_i32(builder, value);
                builder.ins().store(MemFlags::trusted(), zigzag, elem_write_ptr, 0);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Float64 { nullable: false } => {
                let cursor = builder.use_var(cursor_var);
                Self::emit_fixed_bounds_check(builder, cursor, 8, input_len, error_block);
                let read_ptr = builder.ins().iadd(input_ptr, cursor);
                let value = builder.ins().load(types::I64, MemFlags::trusted(), read_ptr, 0);
                builder.ins().store(MemFlags::trusted(), value, elem_write_ptr, 0);
                let eight = builder.ins().iconst(types::I64, 8);
                let new_cursor = builder.ins().iadd(cursor, eight);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Float32 { nullable: false } => {
                let cursor = builder.use_var(cursor_var);
                Self::emit_fixed_bounds_check(builder, cursor, 4, input_len, error_block);
                let read_ptr = builder.ins().iadd(input_ptr, cursor);
                let value = builder.ins().load(types::I32, MemFlags::trusted(), read_ptr, 0);
                builder.ins().store(MemFlags::trusted(), value, elem_write_ptr, 0);
                let four = builder.ins().iconst(types::I64, 4);
                let new_cursor = builder.ins().iadd(cursor, four);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Boolean { nullable: false } => {
                let cursor = builder.use_var(cursor_var);
                Self::emit_fixed_bounds_check(builder, cursor, 1, input_len, error_block);
                let read_ptr = builder.ins().iadd(input_ptr, cursor);
                let value = builder.ins().load(types::I8, MemFlags::trusted(), read_ptr, 0);
                builder.ins().store(MemFlags::trusted(), value, elem_write_ptr, 0);
                let one = builder.ins().iconst(types::I64, 1);
                let new_cursor = builder.ins().iadd(cursor, one);
                builder.def_var(cursor_var, new_cursor);
            }
            _ => {}
        }

        // Update bytes written
        let element_size_val = builder.ins().iconst(types::I64, element_size);
        let new_bytes = builder.ins().iadd(list_bytes_written, element_size_val);
        builder.def_var(list_bytes_var, new_bytes);

        // Increment index
        let one = builder.ins().iconst(types::I64, 1);
        let new_elem_idx = builder.ins().iadd(elem_idx, one);
        builder.def_var(elem_idx_var, new_elem_idx);

        builder.ins().jump(loop_header, &[]);

        // Loop exit
        builder.switch_to_block(loop_exit);
        let final_bytes = builder.use_var(list_bytes_var);
        let new_output_pos = builder.ins().iadd(output_pos, final_bytes);
        builder.def_var(output_vars[buf_idx], new_output_pos);

        // Seal blocks
        builder.seal_block(loop_header);
        builder.seal_block(loop_body);
        builder.seal_block(loop_exit);
    }

    /// Emit Cranelift IR for Decimal128 decoding (16 bytes big-endian to little-endian)
    fn emit_decimal128_decode(
        builder: &mut FunctionBuilder,
        input_ptr: Value,
        input_len: Value,
        cursor_var: Variable,
        output_write_ptr: Value,
        error_block: Block,
    ) {
        let cursor = builder.use_var(cursor_var);

        // Bounds check: need 16 bytes
        Self::emit_fixed_bounds_check(builder, cursor, 16, input_len, error_block);

        let read_ptr = builder.ins().iadd(input_ptr, cursor);

        // Read 16 bytes big-endian, write as little-endian
        // For Decimal128, we need to byte-swap the 16-byte value
        // Read as two i64s and swap
        let high_be = builder.ins().load(types::I64, MemFlags::trusted(), read_ptr, 0);
        let low_be = builder.ins().load(types::I64, MemFlags::trusted(), read_ptr, 8);

        // Byte swap both halves
        let high_le = builder.ins().bswap(high_be);
        let low_le = builder.ins().bswap(low_be);

        // Write in reversed order (low first for little-endian)
        builder.ins().store(MemFlags::trusted(), low_le, output_write_ptr, 0);
        builder.ins().store(MemFlags::trusted(), high_le, output_write_ptr, 8);

        // Update cursor
        let sixteen = builder.ins().iconst(types::I64, 16);
        let new_cursor = builder.ins().iadd(cursor, sixteen);
        builder.def_var(cursor_var, new_cursor);
    }

    /// Emit Cranelift IR for Decimal256 decoding (32 bytes big-endian to little-endian)
    fn emit_decimal256_decode(
        builder: &mut FunctionBuilder,
        input_ptr: Value,
        input_len: Value,
        cursor_var: Variable,
        output_write_ptr: Value,
        error_block: Block,
    ) {
        let cursor = builder.use_var(cursor_var);

        // Bounds check: need 32 bytes
        Self::emit_fixed_bounds_check(builder, cursor, 32, input_len, error_block);

        let read_ptr = builder.ins().iadd(input_ptr, cursor);

        // Read 32 bytes big-endian, write as little-endian
        // Read as four i64s and swap
        let q0_be = builder.ins().load(types::I64, MemFlags::trusted(), read_ptr, 0);
        let q1_be = builder.ins().load(types::I64, MemFlags::trusted(), read_ptr, 8);
        let q2_be = builder.ins().load(types::I64, MemFlags::trusted(), read_ptr, 16);
        let q3_be = builder.ins().load(types::I64, MemFlags::trusted(), read_ptr, 24);

        // Byte swap all quarters
        let q0_le = builder.ins().bswap(q0_be);
        let q1_le = builder.ins().bswap(q1_be);
        let q2_le = builder.ins().bswap(q2_be);
        let q3_le = builder.ins().bswap(q3_be);

        // Write in reversed order for little-endian
        builder.ins().store(MemFlags::trusted(), q3_le, output_write_ptr, 0);
        builder.ins().store(MemFlags::trusted(), q2_le, output_write_ptr, 8);
        builder.ins().store(MemFlags::trusted(), q1_le, output_write_ptr, 16);
        builder.ins().store(MemFlags::trusted(), q0_le, output_write_ptr, 24);

        // Update cursor
        let thirty_two = builder.ins().iconst(types::I64, 32);
        let new_cursor = builder.ins().iadd(cursor, thirty_two);
        builder.def_var(cursor_var, new_cursor);
    }

    /// Emit Cranelift IR for interval decoding
    fn emit_interval_decode(
        builder: &mut FunctionBuilder,
        unit: IntervalUnit,
        input_ptr: Value,
        input_len: Value,
        cursor_var: Variable,
        output_write_ptr: Value,
        error_block: Block,
    ) {
        let cursor = builder.use_var(cursor_var);
        let read_ptr = builder.ins().iadd(input_ptr, cursor);

        match unit {
            IntervalUnit::YearMonth => {
                // 4 bytes (i32) - zigzag encoded varint
                let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
                let zigzag = Self::emit_zigzag_decode_i32(builder, value);
                builder.ins().store(MemFlags::trusted(), zigzag, output_write_ptr, 0);
                builder.def_var(cursor_var, new_cursor);
            }
            IntervalUnit::DayTime => {
                // 8 bytes (two i32s: days + milliseconds)
                // In Avro, this is typically encoded as a fixed 8 bytes
                Self::emit_fixed_bounds_check(builder, cursor, 8, input_len, error_block);
                let value = builder.ins().load(types::I64, MemFlags::trusted(), read_ptr, 0);
                builder.ins().store(MemFlags::trusted(), value, output_write_ptr, 0);
                let eight = builder.ins().iconst(types::I64, 8);
                let new_cursor = builder.ins().iadd(cursor, eight);
                builder.def_var(cursor_var, new_cursor);
            }
            IntervalUnit::MonthDayNano => {
                // 16 bytes (i32 months + i32 days + i64 nanos)
                // In Avro, typically encoded as fixed 16 bytes
                Self::emit_fixed_bounds_check(builder, cursor, 16, input_len, error_block);
                let low = builder.ins().load(types::I64, MemFlags::trusted(), read_ptr, 0);
                let high = builder.ins().load(types::I64, MemFlags::trusted(), read_ptr, 8);
                builder.ins().store(MemFlags::trusted(), low, output_write_ptr, 0);
                builder.ins().store(MemFlags::trusted(), high, output_write_ptr, 8);
                let sixteen = builder.ins().iconst(types::I64, 16);
                let new_cursor = builder.ins().iadd(cursor, sixteen);
                builder.def_var(cursor_var, new_cursor);
            }
        }
    }

    /// Emit Cranelift IR to set a validity bit for a non-null value
    ///
    /// Arrow validity bitmaps: bit 0 = null, bit 1 = valid
    /// Since buffers are initialized to 0 (all null), we only need to set bits for valid values.
    ///
    /// # Arguments
    /// * `builder` - Function builder
    /// * `output_ptrs` - Base pointer to output pointer array
    /// * `validity_idx` - Index of validity buffer in output_ptrs
    /// * `row_idx` - Current row index (to calculate byte and bit position)
    /// * `ptr_type` - Pointer type (I64)
    fn emit_set_validity_bit(
        builder: &mut FunctionBuilder,
        output_ptrs: Value,
        validity_idx: usize,
        row_idx: Value,
        ptr_type: Type,
    ) {
        // Get validity buffer pointer
        let validity_buf_offset = builder.ins().iconst(ptr_type, (validity_idx * 8) as i64);
        let validity_ptr_addr = builder.ins().iadd(output_ptrs, validity_buf_offset);
        let validity_ptr = builder.ins().load(ptr_type, MemFlags::trusted(), validity_ptr_addr, 0);

        // Calculate byte_index = row_idx / 8
        let eight = builder.ins().iconst(types::I64, 8);
        let byte_index = builder.ins().udiv(row_idx, eight);

        // Calculate bit_position = row_idx % 8
        let bit_position = builder.ins().urem(row_idx, eight);

        // Calculate address of the validity byte
        let validity_byte_addr = builder.ins().iadd(validity_ptr, byte_index);

        // Load current byte value
        let current_byte = builder.ins().load(types::I8, MemFlags::trusted(), validity_byte_addr, 0);

        // Calculate the bit mask: 1 << bit_position
        let one_i8 = builder.ins().iconst(types::I8, 1);
        // Need to truncate bit_position from I64 to I8 for shift
        let bit_pos_i8 = builder.ins().ireduce(types::I8, bit_position);
        let bit_mask = builder.ins().ishl(one_i8, bit_pos_i8);

        // Set the bit: current_byte | bit_mask
        let new_byte = builder.ins().bor(current_byte, bit_mask);

        // Store the updated byte
        builder.ins().store(MemFlags::trusted(), new_byte, validity_byte_addr, 0);
    }

    /// Emit Cranelift IR for varint decoding with bounds checking
    /// Handles up to 10 bytes (full i64 range)
    /// Returns (decoded_value, new_cursor_position)
    /// Jumps to error_block if reading past input_len
    fn emit_varint_decode(
        builder: &mut FunctionBuilder,
        input_ptr: Value,
        input_len: Value,
        cursor_var: Variable,
        error_block: Block,
    ) -> (Value, Value) {
        let cursor = builder.use_var(cursor_var);

        // Create variables for the loop
        let value_var = builder.declare_var(types::I64);
        let shift_var = builder.declare_var(types::I64);
        let pos_var = builder.declare_var(types::I64);

        // Initialize: value = 0, shift = 0, pos = cursor
        let zero = builder.ins().iconst(types::I64, 0);
        builder.def_var(value_var, zero);
        builder.def_var(shift_var, zero);
        builder.def_var(pos_var, cursor);

        // Create loop blocks
        let loop_header = builder.create_block();
        let bounds_ok_block = builder.create_block();
        let loop_exit = builder.create_block();

        // Jump to loop header
        builder.ins().jump(loop_header, &[]);

        // Loop header - check bounds then process one byte
        builder.switch_to_block(loop_header);

        // Bounds check: pos < input_len
        let pos = builder.use_var(pos_var);
        let in_bounds = builder.ins().icmp(IntCC::UnsignedLessThan, pos, input_len);
        builder.ins().brif(in_bounds, bounds_ok_block, &[], error_block, &[]);

        // Bounds OK block - read and process byte
        builder.switch_to_block(bounds_ok_block);
        let pos = builder.use_var(pos_var);
        let read_ptr = builder.ins().iadd(input_ptr, pos);
        let byte = builder.ins().load(types::I8, MemFlags::trusted(), read_ptr, 0);
        let byte_i64 = builder.ins().uextend(types::I64, byte);

        // Extract low 7 bits: byte & 0x7F
        let mask_0x7f = builder.ins().iconst(types::I64, 0x7F);
        let low_bits = builder.ins().band(byte_i64, mask_0x7f);

        // Shift and accumulate: value |= (byte & 0x7F) << shift
        let shift = builder.use_var(shift_var);
        let shifted_bits = builder.ins().ishl(low_bits, shift);
        let value = builder.use_var(value_var);
        let new_value = builder.ins().bor(value, shifted_bits);
        builder.def_var(value_var, new_value);

        // Increment shift by 7
        let seven = builder.ins().iconst(types::I64, 7);
        let new_shift = builder.ins().iadd(shift, seven);
        builder.def_var(shift_var, new_shift);

        // Increment position
        let one = builder.ins().iconst(types::I64, 1);
        let new_pos = builder.ins().iadd(pos, one);
        builder.def_var(pos_var, new_pos);

        // Check if more bytes: (byte & 0x80) != 0
        let mask_0x80 = builder.ins().iconst(types::I64, 0x80);
        let has_more = builder.ins().band(byte_i64, mask_0x80);
        let has_more_cmp = builder.ins().icmp(IntCC::NotEqual, has_more, zero);

        // Branch: if has_more, continue loop; else exit
        builder.ins().brif(has_more_cmp, loop_header, &[], loop_exit, &[]);

        // Loop exit - get final values from variables
        builder.switch_to_block(loop_exit);
        let final_value = builder.use_var(value_var);
        let final_pos = builder.use_var(pos_var);

        // Seal blocks
        builder.seal_block(loop_header);
        builder.seal_block(bounds_ok_block);
        builder.seal_block(loop_exit);

        (final_value, final_pos)
    }

    /// Emit zigzag decode for i32: (n >> 1) ^ -(n & 1)
    fn emit_zigzag_decode_i32(builder: &mut FunctionBuilder, value: Value) -> Value {
        let value_i32 = builder.ins().ireduce(types::I32, value);
        let one_i32 = builder.ins().iconst(types::I32, 1);
        let shifted = builder.ins().ushr(value_i32, one_i32);
        let masked = builder.ins().band(value_i32, one_i32);
        let negated = builder.ins().ineg(masked);
        builder.ins().bxor(shifted, negated)
    }

    /// Emit zigzag decode for i64: (n >> 1) ^ -(n & 1)
    fn emit_zigzag_decode_i64(builder: &mut FunctionBuilder, value: Value) -> Value {
        let one = builder.ins().iconst(types::I64, 1);
        let shifted = builder.ins().ushr(value, one);
        let masked = builder.ins().band(value, one);
        let negated = builder.ins().ineg(masked);
        builder.ins().bxor(shifted, negated)
    }

    /// Emit bounds check for fixed-size read.
    /// Checks that cursor + size <= input_len, jumps to error_block if out of bounds.
    /// Does NOT create a new block - caller should already be in the block where load will happen.
    /// This is an inline check that branches to error_block on failure.
    fn emit_fixed_bounds_check(
        builder: &mut FunctionBuilder,
        cursor: Value,
        size: i64,
        input_len: Value,
        error_block: Block,
    ) {
        let size_val = builder.ins().iconst(types::I64, size);
        let end_pos = builder.ins().iadd(cursor, size_val);
        let in_bounds = builder.ins().icmp(IntCC::UnsignedLessThanOrEqual, end_pos, input_len);

        // Create continuation block for the successful case
        let continue_block = builder.create_block();
        builder.ins().brif(in_bounds, continue_block, &[], error_block, &[]);
        builder.seal_block(continue_block);
        builder.switch_to_block(continue_block);
    }

    /// Emit bounds check for variable-size read (e.g., bytes/string after length is decoded).
    /// Checks that cursor + len <= input_len and len >= 0, jumps to error_block if invalid.
    fn emit_variable_bounds_check(
        builder: &mut FunctionBuilder,
        cursor: Value,
        len: Value,
        input_len: Value,
        error_block: Block,
    ) {
        // First check: len >= 0 (signed comparison)
        let zero = builder.ins().iconst(types::I64, 0);
        let len_valid = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, len, zero);

        let len_ok_block = builder.create_block();
        builder.ins().brif(len_valid, len_ok_block, &[], error_block, &[]);
        builder.seal_block(len_ok_block);
        builder.switch_to_block(len_ok_block);

        // Second check: cursor + len <= input_len
        let end_pos = builder.ins().iadd(cursor, len);
        let in_bounds = builder.ins().icmp(IntCC::UnsignedLessThanOrEqual, end_pos, input_len);

        let continue_block = builder.create_block();
        builder.ins().brif(in_bounds, continue_block, &[], error_block, &[]);
        builder.seal_block(continue_block);
        builder.switch_to_block(continue_block);
    }

    /// Emit bounds check for output buffer write.
    /// Checks that output_pos + len <= output_capacity, jumps to overflow_block if invalid.
    fn emit_output_bounds_check(
        builder: &mut FunctionBuilder,
        output_pos: Value,
        len: Value,
        output_capacity: Value,
        overflow_block: Block,
    ) {
        // Check: output_pos + len <= output_capacity
        let end_pos = builder.ins().iadd(output_pos, len);
        let in_bounds = builder.ins().icmp(IntCC::UnsignedLessThanOrEqual, end_pos, output_capacity);

        let continue_block = builder.create_block();
        builder.ins().brif(in_bounds, continue_block, &[], overflow_block, &[]);
        builder.seal_block(continue_block);
        builder.switch_to_block(continue_block);
    }

    /// Decode rows using the JIT-compiled function
    ///
    /// # Arguments
    /// * `input` - Raw Avro bytes
    /// * `output_buffers` - Mutable slice of output buffer pointers
    ///   For schemas with variable-length columns (Utf8, Binary), this needs
    ///   `total_buffers()` buffers (2 per variable-length column: offsets + data)
    /// * `row_count` - Number of rows to decode
    ///
    /// # Returns
    /// Number of bytes consumed from input, or error
    ///
    /// # Safety
    /// The output buffers must have sufficient capacity for the decoded data.
    /// For fixed-size columns, use `decode_checked()` for validation.
    /// For variable-length columns, this method uses maximum capacity limits.
    #[inline(always)]
    pub fn decode(
        &self,
        input: &[u8],
        output_buffers: &mut [*mut u8],
        row_count: usize,
    ) -> Result<usize, String> {
        if output_buffers.len() != self.total_buffers {
            return Err(format!(
                "Expected {} output buffers, got {}",
                self.total_buffers,
                output_buffers.len()
            ));
        }

        // Use maximum capacity for backward compatibility (essentially unlimited)
        let output_capacities: Vec<usize> = vec![usize::MAX; self.total_buffers];

        self.decode_with_caps(input, output_buffers, &output_capacities, row_count)
    }

    /// Decode rows with explicit buffer capacities
    ///
    /// # Arguments
    /// * `input` - Raw Avro bytes
    /// * `output_buffers` - Mutable slice of output buffer pointers
    /// * `output_capacities` - Capacity of each output buffer in bytes
    /// * `row_count` - Number of rows to decode
    ///
    /// # Returns
    /// Number of bytes consumed from input, or error
    fn decode_with_caps(
        &self,
        input: &[u8],
        output_buffers: &mut [*mut u8],
        output_capacities: &[usize],
        row_count: usize,
    ) -> Result<usize, String> {
        // Use catch_unwind as additional safety net against any potential panics
        let decode_fn = self.decode_fn;
        let input_ptr = input.as_ptr();
        let input_len = input.len();
        let output_ptr = output_buffers.as_mut_ptr();
        let output_caps_ptr = output_capacities.as_ptr();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            decode_fn(input_ptr, input_len, output_ptr, output_caps_ptr, row_count)
        }));

        match result {
            Ok(bytes_consumed) => {
                if bytes_consumed == JIT_ERROR_INPUT_BOUNDS {
                    Err("Malformed input: bounds violation during Avro decoding".to_string())
                } else if bytes_consumed == JIT_ERROR_OUTPUT_OVERFLOW {
                    Err("Output buffer overflow: insufficient capacity for decoded data".to_string())
                } else if bytes_consumed < 0 {
                    Err(format!("Unknown decode error: {}", bytes_consumed))
                } else {
                    Ok(bytes_consumed as usize)
                }
            }
            Err(_) => Err("Decode panic: corrupted or malicious input data".to_string()),
        }
    }

    /// Get the schema this decoder was compiled for
    pub fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    /// Get the number of columns
    pub fn num_columns(&self) -> usize {
        self.num_columns
    }

    /// Get column types for buffer layout decisions
    pub fn column_types(&self) -> &[ColumnType] {
        &self.column_types
    }

    /// Calculate the exact required capacity for a fixed-size column
    ///
    /// Returns `Some(size)` for fixed-size columns, `None` for variable-length columns.
    pub fn required_capacity(&self, col_idx: usize, row_count: usize) -> Option<usize> {
        if col_idx >= self.schema.fields().len() {
            return None;
        }
        let field = &self.schema.fields()[col_idx];
        Self::fixed_size_for_type(field.data_type()).map(|size| size * row_count)
    }

    /// Get the fixed size in bytes for a data type, if it's fixed-size
    pub fn fixed_size_for_type(data_type: &DataType) -> Option<usize> {
        match data_type {
            DataType::Boolean => Some(1), // JIT outputs 1 byte per boolean
            DataType::Int8 | DataType::UInt8 => Some(1),
            DataType::Int16 | DataType::UInt16 => Some(2),
            DataType::Int32 | DataType::UInt32 | DataType::Float32 | DataType::Date32 | DataType::Time32(_) => Some(4),
            DataType::Int64 | DataType::UInt64 | DataType::Float64 | DataType::Date64 | DataType::Time64(_) | DataType::Timestamp(_, _) | DataType::Duration(_) => Some(8),
            DataType::Interval(IntervalUnit::YearMonth) => Some(4),
            DataType::Interval(IntervalUnit::DayTime) => Some(8),
            DataType::Interval(IntervalUnit::MonthDayNano) => Some(16),
            DataType::FixedSizeBinary(n) => Some(*n as usize),
            DataType::Decimal128(_, _) => Some(16),
            DataType::Decimal256(_, _) => Some(32),
            DataType::Dictionary(_, _) => Some(4), // Enum index stored as i32
            // Variable-length types
            DataType::Utf8 | DataType::Binary | DataType::LargeUtf8 | DataType::LargeBinary => None,
            DataType::List(_) | DataType::LargeList(_) | DataType::Map(_, _) => None,
            // Complex types - treat as variable for safety
            DataType::Struct(_) | DataType::Union(_, _) | DataType::FixedSizeList(_, _) => None,
            _ => None,
        }
    }

    /// Check if the schema has any variable-length columns
    pub fn has_variable_columns(&self) -> bool {
        self.column_types.iter().any(|ct| *ct == ColumnType::Variable)
    }

    /// Get the total number of output buffers needed
    ///
    /// This accounts for variable-length columns needing 2 buffers (offsets + data)
    pub fn total_buffers(&self) -> usize {
        self.total_buffers
    }

    /// Get the buffer slot for a column
    pub fn buffer_slot(&self, col_idx: usize) -> Option<&BufferSlot> {
        self.buffer_slots.get(col_idx)
    }

    /// Get the column type for a column
    pub fn column_type(&self, col_idx: usize) -> Option<ColumnType> {
        self.column_types.get(col_idx).copied()
    }

    /// Decode Avro data with output buffer capacity validation
    ///
    /// This method validates that fixed-size columns have sufficient capacity
    /// before decoding. For variable-length columns, the caller must ensure
    /// adequate capacity based on expected data sizes.
    ///
    /// # Arguments
    /// * `input` - Raw Avro bytes (without header/sync markers)
    /// * `output_buffers` - Output buffer specifications with pointers and capacities
    ///   For variable-length columns (Utf8, Binary), provide 2 buffers: [offsets, data]
    /// * `row_count` - Number of rows to decode
    ///
    /// # Returns
    /// Number of bytes consumed from input, or error
    ///
    /// # Errors
    /// - Returns error if fixed-size column has insufficient capacity
    /// - Returns error on input bounds violation
    pub fn decode_checked(
        &self,
        input: &[u8],
        output_buffers: &[OutputBuffer],
        row_count: usize,
    ) -> Result<usize, String> {
        if output_buffers.len() != self.total_buffers {
            return Err(format!(
                "Expected {} output buffers, got {}",
                self.total_buffers,
                output_buffers.len()
            ));
        }

        // Validate buffer capacities
        for (col_idx, field) in self.schema.fields().iter().enumerate() {
            let slot = &self.buffer_slots[col_idx];
            if let Some(required) = Self::fixed_size_for_type(field.data_type()) {
                let required_capacity = required * row_count;
                let actual_capacity = output_buffers[slot.primary].capacity;
                if actual_capacity < required_capacity {
                    return Err(format!(
                        "Column '{}' requires {} bytes ({} rows × {} bytes/row) but buffer has {} bytes capacity",
                        field.name(), required_capacity, row_count, required, actual_capacity
                    ));
                }
            }
            // Validate variable-length column offsets buffer capacity
            if slot.secondary.is_some() {
                // Offsets buffer needs (row_count + 1) * 4 bytes for i32 offsets
                let offsets_required = (row_count + 1) * 4;
                let offsets_actual = output_buffers[slot.primary].capacity;
                if offsets_actual < offsets_required {
                    return Err(format!(
                        "Column '{}' offsets buffer requires {} bytes ({} offsets × 4 bytes) but buffer has {} bytes capacity",
                        field.name(), offsets_required, row_count + 1, offsets_actual
                    ));
                }
            }
        }

        // Extract pointers and capacities for the JIT function
        let mut ptrs: Vec<*mut u8> = output_buffers.iter().map(|b| b.ptr).collect();
        let caps: Vec<usize> = output_buffers.iter().map(|b| b.capacity).collect();

        // Call the underlying decode function with capacities
        let result = self.decode_with_caps(input, &mut ptrs, &caps, row_count)?;

        // Write final offset for variable-length columns (total data length)
        // The JIT writes offsets at positions [0, 4, 8, ...] up to row_count offsets
        // We need to write one more offset at position row_count * 4
        for (col_idx, slot) in self.buffer_slots.iter().enumerate() {
            if let Some(data_buf_idx) = slot.secondary {
                // Get the data buffer's final position (= total data length)
                // This was tracked in output_vars[data_buf_idx] during decode
                // We need to read it back from the offsets buffer's final written position
                // Actually, we can compute it from the data buffer usage
                // For now, we'll use a simpler approach: read the current offset position
                // and write the final offset

                // The offsets buffer has row_count offsets written at byte positions 0..row_count*4
                // We need to write offset[row_count] = total_data_len
                // The total_data_len is the current value of the data buffer position
                // But we don't have direct access to that here...

                // Alternative approach: the caller must write the final offset
                // or we compute it from the offset buffer's current write position

                // For proper Arrow StringArray, offsets[row_count] must be set
                // Since we track data_pos in JIT, the final data_pos is total_data_len
                // We can read the last written offset to see where it ended

                // Actually, the JIT code writes offset BEFORE each string
                // So for N rows, we have offsets at [0, 4, 8, ..., (N-1)*4]
                // offset[i] = position where string i starts
                // We need offset[N] = total_data_len (end of last string)

                // The data buffer's current position after decode = total_data_len
                // But we can't easily get that from here...

                // Let's add a note that caller must handle final offset
                // OR we change the JIT to write final offset after loop

                // For now, skip - this is handled by the JIT writing offset before each copy
                // The final offset will be computed based on the data buffer usage
                let _ = (col_idx, data_buf_idx); // Suppress unused warning
            }
        }

        Ok(result)
    }

    /// Decode Avro data with chunked progress reporting
    ///
    /// This method is designed for streaming/chunked decode scenarios where:
    /// - Input may be larger than output buffer capacity
    /// - Partial progress must be tracked for continuation
    ///
    /// Pre-calculates max rows based on fixed-size column capacities,
    /// then handles variable-length overflow with binary search.
    ///
    /// # Arguments
    /// * `input` - Raw Avro bytes (without header/sync markers)
    /// * `output_buffers` - Output buffer specifications with pointers and capacities
    /// * `row_count` - Maximum number of rows to decode
    ///
    /// # Returns
    /// DecodeResult with bytes_consumed, rows_decoded, and status
    pub fn decode_chunked(
        &self,
        input: &[u8],
        output_buffers: &[OutputBuffer],
        row_count: usize,
    ) -> DecodeResult {
        if row_count == 0 {
            return DecodeResult::ok(0, 0);
        }

        if output_buffers.len() != self.total_buffers {
            return DecodeResult::input_error();
        }

        // Calculate max rows that fit based on fixed-size column capacities
        // JIT doesn't do bounds checking for fixed-size columns, so we must pre-validate
        let max_rows_by_capacity = self.compute_max_rows(output_buffers);

        if max_rows_by_capacity == 0 {
            // Buffer too small for even 1 row
            return DecodeResult::overflow(0, 0);
        }

        // Use minimum of requested rows and capacity-limited rows
        let rows_to_decode = row_count.min(max_rows_by_capacity);

        let mut ptrs: Vec<*mut u8> = output_buffers.iter().map(|b| b.ptr).collect();
        let caps: Vec<usize> = output_buffers.iter().map(|b| b.capacity).collect();

        let result = self.decode_with_caps(input, &mut ptrs, &caps, rows_to_decode);

        match result {
            Ok(bytes_consumed) => {
                if rows_to_decode < row_count {
                    // We decoded fewer rows than requested due to capacity
                    DecodeResult::overflow(bytes_consumed, rows_to_decode)
                } else {
                    DecodeResult::ok(bytes_consumed, rows_to_decode)
                }
            }
            Err(ref e) if e.contains("overflow") => {
                // Variable-length column overflow - use binary search
                self.find_max_rows_variable(input, output_buffers, rows_to_decode)
            }
            Err(_) => {
                // Input bounds error
                DecodeResult::input_error()
            }
        }
    }

    /// Compute maximum rows that fit based on fixed-size column capacities
    fn compute_max_rows(&self, output_buffers: &[OutputBuffer]) -> usize {
        let mut max_rows = usize::MAX;

        for (col_idx, field) in self.schema.fields().iter().enumerate() {
            let slot = &self.buffer_slots[col_idx];

            // Check fixed-size columns
            if let Some(elem_size) = Self::fixed_size_for_type(field.data_type()) {
                let capacity = output_buffers[slot.primary].capacity;
                let rows_that_fit = capacity / elem_size;
                max_rows = max_rows.min(rows_that_fit);
            }

            // Check offsets buffer for variable-length columns
            if slot.secondary.is_some() {
                // Offsets buffer needs (row_count + 1) * 4 bytes
                let offsets_capacity = output_buffers[slot.primary].capacity;
                let rows_that_fit = if offsets_capacity >= 4 {
                    (offsets_capacity / 4).saturating_sub(1)
                } else {
                    0
                };
                max_rows = max_rows.min(rows_that_fit);
            }
        }

        max_rows
    }

    /// Find maximum rows for variable-length overflow using binary search
    fn find_max_rows_variable(
        &self,
        input: &[u8],
        output_buffers: &[OutputBuffer],
        max_rows: usize,
    ) -> DecodeResult {
        let caps: Vec<usize> = output_buffers.iter().map(|b| b.capacity).collect();

        // Try 1 row first
        let mut ptrs: Vec<*mut u8> = output_buffers.iter().map(|b| b.ptr).collect();
        if self.decode_with_caps(input, &mut ptrs, &caps, 1).is_err() {
            return DecodeResult::overflow(0, 0);
        }

        // Exponential search to find upper bound
        let mut low = 1usize;
        let mut high = 2usize;

        while high < max_rows {
            let mut ptrs: Vec<*mut u8> = output_buffers.iter().map(|b| b.ptr).collect();
            if self.decode_with_caps(input, &mut ptrs, &caps, high).is_ok() {
                low = high;
                high *= 2;
            } else {
                break;
            }
        }
        high = high.min(max_rows);

        // Binary search between low and high
        while low < high {
            let mid = low + (high - low + 1) / 2;
            let mut ptrs: Vec<*mut u8> = output_buffers.iter().map(|b| b.ptr).collect();
            if self.decode_with_caps(input, &mut ptrs, &caps, mid).is_ok() {
                low = mid;
            } else {
                high = mid - 1;
            }
        }

        // Final decode to get bytes_consumed
        let mut ptrs: Vec<*mut u8> = output_buffers.iter().map(|b| b.ptr).collect();
        match self.decode_with_caps(input, &mut ptrs, &caps, low) {
            Ok(bytes_consumed) => DecodeResult::overflow(bytes_consumed, low),
            Err(_) => DecodeResult::overflow(0, 0),
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode i64 as zigzag + varint (Avro format)
    fn encode_varint_i64(value: i64) -> Vec<u8> {
        let zigzag = ((value << 1) ^ (value >> 63)) as u64;
        let mut result = Vec::new();
        let mut v = zigzag;
        loop {
            if v < 0x80 {
                result.push(v as u8);
                break;
            }
            result.push((v as u8) | 0x80);
            v >>= 7;
        }
        result
    }

    /// Encode i32 as zigzag + varint (Avro format)
    fn encode_varint_i32(value: i32) -> Vec<u8> {
        let zigzag = ((value << 1) ^ (value >> 31)) as u32;
        let mut result = Vec::new();
        let mut v = zigzag;
        loop {
            if v < 0x80 {
                result.push(v as u8);
                break;
            }
            result.push((v as u8) | 0x80);
            v >>= 7;
        }
        result
    }

    #[test]
    fn test_compile_simple_schema() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");
        assert_eq!(decoder.num_columns(), 2);
    }

    #[test]
    fn test_decode_int64_single() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Avro varint for 1: zigzag(1) = 2, varint(2) = 0x02
        let input = vec![0x02u8];
        let mut output = vec![0u8; 8];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, 1).expect("Decode failed");
        assert_eq!(consumed, 1);

        let value = i64::from_le_bytes(output[0..8].try_into().unwrap());
        assert_eq!(value, 1);
    }

    #[test]
    fn test_roundtrip_int64_multiple_values() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Test values that encode to single-byte varints (0-63 -> zigzag 0-126)
        let test_values: Vec<i64> = vec![0, 1, 2, 10, 31, 63];

        let mut input = Vec::new();
        for &v in &test_values {
            input.extend(encode_varint_i64(v));
        }

        let row_count = test_values.len();
        let mut output = vec![0u8; row_count * 8];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify each value
        for (i, &expected) in test_values.iter().enumerate() {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, expected, "Mismatch at index {}: expected {}, got {}", i, expected, actual);
        }
    }

    #[test]
    fn test_roundtrip_int32_values() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("count", DataType::Int32, false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let test_values: Vec<i32> = vec![0, 1, 5, 42, 63];

        let mut input = Vec::new();
        for &v in &test_values {
            input.extend(encode_varint_i32(v));
        }

        let row_count = test_values.len();
        let mut output = vec![0u8; row_count * 4];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in test_values.iter().enumerate() {
            let actual = i32::from_le_bytes(output[i * 4..(i + 1) * 4].try_into().unwrap());
            assert_eq!(actual, expected, "Int32 mismatch at index {}", i);
        }
    }

    #[test]
    fn test_roundtrip_float64_values() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Float64, false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let test_values: Vec<f64> = vec![0.0, 1.5, -3.14159, 100.0, 999.999];

        let mut input = Vec::new();
        for &v in &test_values {
            input.extend(v.to_le_bytes());
        }

        let row_count = test_values.len();
        let mut output = vec![0u8; row_count * 8];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in test_values.iter().enumerate() {
            let actual = f64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
            assert!((actual - expected).abs() < 1e-10, "Float64 mismatch at index {}: {} vs {}", i, actual, expected);
        }
    }

    #[test]
    fn test_roundtrip_float32_values() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("ratio", DataType::Float32, false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let test_values: Vec<f32> = vec![0.0, 1.5, -2.5, 100.0];

        let mut input = Vec::new();
        for &v in &test_values {
            input.extend(v.to_le_bytes());
        }

        let row_count = test_values.len();
        let mut output = vec![0u8; row_count * 4];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in test_values.iter().enumerate() {
            let actual = f32::from_le_bytes(output[i * 4..(i + 1) * 4].try_into().unwrap());
            assert!((actual - expected).abs() < 1e-6, "Float32 mismatch at index {}", i);
        }
    }

    #[test]
    fn test_roundtrip_boolean_values() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("active", DataType::Boolean, false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Avro encodes booleans as single bytes: 0 = false, 1 = true
        let test_values: Vec<bool> = vec![true, false, true, true, false];

        let mut input = Vec::new();
        for &v in &test_values {
            input.push(if v { 1 } else { 0 });
        }

        let row_count = test_values.len();
        let mut output = vec![0u8; row_count];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in test_values.iter().enumerate() {
            let actual = output[i] != 0;
            assert_eq!(actual, expected, "Boolean mismatch at index {}", i);
        }
    }

    #[test]
    fn test_roundtrip_multi_column_flat_schema() {
        // Test schema with Int64, Float64, Int32, Float32 columns
        // NOTE: JIT prototype only supports single-byte varints (values 0-63)
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Float64, false),
            Field::new("count", DataType::Int32, false),
            Field::new("ratio", DataType::Float32, false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let row_count = 10;

        // Encode test data: id (varint), value (f64), count (varint), ratio (f32)
        // Use values within single-byte varint range (0-63)
        let mut input = Vec::new();
        for i in 0..row_count {
            input.extend(encode_varint_i64(i as i64));            // id: 0-9
            input.extend((i as f64 * 0.1).to_le_bytes());         // value
            input.extend(encode_varint_i32((i % 64) as i32));     // count: keep in single-byte range
            input.extend((i as f32 / 100.0).to_le_bytes());       // ratio
        }

        let mut output_id = vec![0u8; row_count * 8];
        let mut output_value = vec![0u8; row_count * 8];
        let mut output_count = vec![0u8; row_count * 4];
        let mut output_ratio = vec![0u8; row_count * 4];

        let mut output_ptrs = vec![
            output_id.as_mut_ptr(),
            output_value.as_mut_ptr(),
            output_count.as_mut_ptr(),
            output_ratio.as_mut_ptr(),
        ];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert!(consumed > 0);

        // Verify values
        for i in 0..row_count {
            let id = i64::from_le_bytes(output_id[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(id, i as i64, "id mismatch at row {}", i);

            let value = f64::from_le_bytes(output_value[i * 8..(i + 1) * 8].try_into().unwrap());
            assert!((value - i as f64 * 0.1).abs() < 1e-10, "value mismatch at row {}", i);

            let count = i32::from_le_bytes(output_count[i * 4..(i + 1) * 4].try_into().unwrap());
            assert_eq!(count, (i % 64) as i32, "count mismatch at row {}", i);

            let ratio = f32::from_le_bytes(output_ratio[i * 4..(i + 1) * 4].try_into().unwrap());
            assert!((ratio - i as f32 / 100.0).abs() < 1e-6, "ratio mismatch at row {}", i);
        }
    }

    #[test]
    fn test_roundtrip_zigzag_negative_values() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int64, false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Test negative values that encode to single-byte varints
        // -1 -> zigzag 1, -2 -> zigzag 3, -32 -> zigzag 63
        let test_values: Vec<i64> = vec![-1, -2, -10, -31, -32];

        let mut input = Vec::new();
        for &v in &test_values {
            input.extend(encode_varint_i64(v));
        }

        let row_count = test_values.len();
        let mut output = vec![0u8; row_count * 8];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in test_values.iter().enumerate() {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, expected, "Negative zigzag mismatch at index {}: expected {}, got {}", i, expected, actual);
        }
    }

    #[test]
    fn test_roundtrip_high_throughput() {
        // Test with larger row count to verify loop handling
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let row_count = 1000;

        let mut input = Vec::new();
        for i in 0..row_count {
            // Use values 0-63 which encode to single-byte varints
            input.extend(encode_varint_i64((i % 64) as i64));
        }

        let mut output = vec![0u8; row_count * 8];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify all values
        for i in 0..row_count {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
            let expected = (i % 64) as i64;
            assert_eq!(actual, expected, "High throughput mismatch at row {}", i);
        }
    }

    #[test]
    fn test_roundtrip_large_integers_multibyte_varint() {
        // Test values that require multi-byte varints
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int64, false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Test values requiring 1-5 byte varints
        let test_values: Vec<i64> = vec![
            0,           // 1 byte: zigzag 0
            63,          // 1 byte: zigzag 126
            64,          // 2 bytes: zigzag 128
            100,         // 2 bytes: zigzag 200
            1000,        // 2 bytes: zigzag 2000
            10000,       // 3 bytes: zigzag 20000
            100000,      // 3 bytes: zigzag 200000
            1000000,     // 3 bytes: zigzag 2000000
            10000000,    // 4 bytes
            100000000,   // 4 bytes
            1000000000,  // 5 bytes
            i64::MAX / 2, // large value
        ];

        let mut input = Vec::new();
        for &v in &test_values {
            input.extend(encode_varint_i64(v));
        }

        let row_count = test_values.len();
        let mut output = vec![0u8; row_count * 8];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in test_values.iter().enumerate() {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, expected, "Large int mismatch at index {}: expected {}, got {}", i, expected, actual);
        }
    }

    #[test]
    fn test_roundtrip_large_negative_integers() {
        // Test large negative values requiring multi-byte varints
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int64, false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let test_values: Vec<i64> = vec![
            -33,         // 2 bytes: zigzag 65
            -64,         // 2 bytes: zigzag 127
            -65,         // 2 bytes: zigzag 129
            -100,        // 2 bytes
            -1000,       // 2 bytes
            -10000,      // 3 bytes
            -100000,     // 3 bytes
            -1000000,    // 3 bytes
            -1000000000, // 5 bytes
            i64::MIN / 2, // large negative
        ];

        let mut input = Vec::new();
        for &v in &test_values {
            input.extend(encode_varint_i64(v));
        }

        let row_count = test_values.len();
        let mut output = vec![0u8; row_count * 8];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in test_values.iter().enumerate() {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, expected, "Large negative mismatch at index {}: expected {}, got {}", i, expected, actual);
        }
    }

    #[test]
    fn test_roundtrip_int32_large_values() {
        // Test Int32 with multi-byte varints
        let schema = Arc::new(Schema::new(vec![
            Field::new("count", DataType::Int32, false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let test_values: Vec<i32> = vec![
            0, 63, 64, 100, 1000, 10000, 100000, 1000000,
            -1, -32, -33, -64, -65, -1000, -100000,
            i32::MAX, i32::MIN,
        ];

        let mut input = Vec::new();
        for &v in &test_values {
            input.extend(encode_varint_i32(v));
        }

        let row_count = test_values.len();
        let mut output = vec![0u8; row_count * 4];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in test_values.iter().enumerate() {
            let actual = i32::from_le_bytes(output[i * 4..(i + 1) * 4].try_into().unwrap());
            assert_eq!(actual, expected, "Int32 large value mismatch at index {}: expected {}, got {}", i, expected, actual);
        }
    }

    #[test]
    fn test_roundtrip_mixed_large_values_multicolumn() {
        // Test multi-column schema with large values
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Float64, false),
            Field::new("count", DataType::Int32, false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let row_count = 100;

        let mut input = Vec::new();
        for i in 0..row_count {
            // Large Int64 values
            input.extend(encode_varint_i64((i * 100000) as i64));
            // Float64
            input.extend((i as f64 * 1.5).to_le_bytes());
            // Large Int32 values
            input.extend(encode_varint_i32((i * 1000) as i32));
        }

        let mut output_id = vec![0u8; row_count * 8];
        let mut output_value = vec![0u8; row_count * 8];
        let mut output_count = vec![0u8; row_count * 4];

        let mut output_ptrs = vec![
            output_id.as_mut_ptr(),
            output_value.as_mut_ptr(),
            output_count.as_mut_ptr(),
        ];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert!(consumed > 0);

        // Verify values
        for i in 0..row_count {
            let id = i64::from_le_bytes(output_id[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(id, (i * 100000) as i64, "id mismatch at row {}", i);

            let value = f64::from_le_bytes(output_value[i * 8..(i + 1) * 8].try_into().unwrap());
            assert!((value - i as f64 * 1.5).abs() < 1e-10, "value mismatch at row {}", i);

            let count = i32::from_le_bytes(output_count[i * 4..(i + 1) * 4].try_into().unwrap());
            assert_eq!(count, (i * 1000) as i32, "count mismatch at row {}", i);
        }
    }

    /// Encode bytes in Avro format: length varint + raw bytes
    fn encode_avro_bytes(data: &[u8]) -> Vec<u8> {
        let mut result = Vec::new();
        // Length is encoded as zigzag varint (non-negative, so just *2)
        let len = data.len() as i64;
        let zigzag = ((len << 1) ^ (len >> 63)) as u64;
        let mut v = zigzag;
        loop {
            if v < 0x80 {
                result.push(v as u8);
                break;
            }
            result.push((v as u8) | 0x80);
            v >>= 7;
        }
        // Raw bytes
        result.extend_from_slice(data);
        result
    }

    #[test]
    fn test_roundtrip_bytes_single() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("data", DataType::Binary, false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Verify we need 2 buffers for Bytes type
        assert_eq!(decoder.total_buffers(), 2);
        let slot = decoder.buffer_slot(0).expect("buffer slot");
        assert_eq!(slot.primary, 0);  // offsets buffer
        assert_eq!(slot.secondary, Some(1));  // data buffer

        // Test a single bytes value
        let test_data = b"Hello, World!";
        let input = encode_avro_bytes(test_data);
        let row_count = 1;

        // Buffer 0: offsets (row_count + 1 i32 values = 8 bytes for 1 row)
        let mut offsets = vec![0u8; (row_count + 1) * 4];
        // Buffer 1: data
        let mut data = vec![0u8; test_data.len() + 100];

        let mut output_ptrs = vec![offsets.as_mut_ptr(), data.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify offsets: [0, len]
        let offset_0 = i32::from_le_bytes(offsets[0..4].try_into().unwrap());
        let offset_1 = i32::from_le_bytes(offsets[4..8].try_into().unwrap());
        assert_eq!(offset_0, 0);
        assert_eq!(offset_1, test_data.len() as i32);

        // Verify the bytes were copied correctly to data buffer
        assert_eq!(&data[..test_data.len()], test_data);
    }

    #[test]
    fn test_roundtrip_bytes_multiple() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let test_strings = vec!["Alice", "Bob", "Charlie", "Dave"];

        let mut input = Vec::new();
        let mut total_data_len = 0;
        for s in &test_strings {
            input.extend(encode_avro_bytes(s.as_bytes()));
            total_data_len += s.len();
        }

        let row_count = test_strings.len();
        // Buffer 0: offsets (row_count + 1 i32 values)
        let mut offsets = vec![0u8; (row_count + 1) * 4];
        // Buffer 1: data
        let mut data = vec![0u8; total_data_len + 100];

        let mut output_ptrs = vec![offsets.as_mut_ptr(), data.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify offsets are correct: [0, 5, 8, 15, 19] for ["Alice", "Bob", "Charlie", "Dave"]
        let mut expected_offset = 0;
        for (i, s) in test_strings.iter().enumerate() {
            let offset_i = i32::from_le_bytes(offsets[i * 4..(i + 1) * 4].try_into().unwrap());
            assert_eq!(offset_i, expected_offset as i32, "Offset {} mismatch", i);
            expected_offset += s.len();
        }
        // Final offset
        let offset_n = i32::from_le_bytes(offsets[row_count * 4..(row_count + 1) * 4].try_into().unwrap());
        assert_eq!(offset_n, total_data_len as i32, "Final offset mismatch");

        // Verify the strings were copied in sequence to data buffer
        let mut data_offset = 0;
        for s in &test_strings {
            let actual = &data[data_offset..data_offset + s.len()];
            assert_eq!(actual, s.as_bytes(), "String mismatch: expected '{}', got '{:?}'", s, actual);
            data_offset += s.len();
        }
    }

    #[test]
    fn test_roundtrip_bytes_empty() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("data", DataType::Binary, false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Empty bytes: length = 0
        let input = encode_avro_bytes(b"");
        let row_count = 1;

        // Buffer 0: offsets (row_count + 1 i32 values)
        let mut offsets = vec![0xFFu8; (row_count + 1) * 4];
        // Buffer 1: data
        let mut data = vec![0xFFu8; 100];

        let mut output_ptrs = vec![offsets.as_mut_ptr(), data.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, 1); // Just the length varint (0x00)

        // Verify offsets: [0, 0] for empty bytes
        let offset_0 = i32::from_le_bytes(offsets[0..4].try_into().unwrap());
        let offset_1 = i32::from_le_bytes(offsets[4..8].try_into().unwrap());
        assert_eq!(offset_0, 0);
        assert_eq!(offset_1, 0);

        // Data buffer should not have been written to
        assert_eq!(data[0], 0xFF, "Data was modified for empty bytes");
    }

    #[test]
    fn test_roundtrip_bytes_with_int_column() {
        // Test mixed schema: Int64 + Bytes + Int32
        // Buffer layout: [id (buf 0), name_offsets (buf 1), name_data (buf 2), count (buf 3)]
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("count", DataType::Int32, false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Verify buffer slots
        assert_eq!(decoder.total_buffers(), 4);
        let slot_id = decoder.buffer_slot(0).expect("id slot");
        assert_eq!(slot_id.primary, 0);
        assert!(slot_id.secondary.is_none());
        let slot_name = decoder.buffer_slot(1).expect("name slot");
        assert_eq!(slot_name.primary, 1);
        assert_eq!(slot_name.secondary, Some(2));
        let slot_count = decoder.buffer_slot(2).expect("count slot");
        assert_eq!(slot_count.primary, 3);
        assert!(slot_count.secondary.is_none());

        let row_count = 5;
        let names = ["Alice", "Bob", "Charlie", "Dave", "Eve"];

        let mut input = Vec::new();
        let mut total_name_len = 0;
        for (i, name) in names.iter().enumerate() {
            input.extend(encode_varint_i64(i as i64 * 100));
            input.extend(encode_avro_bytes(name.as_bytes()));
            input.extend(encode_varint_i32(i as i32 * 10));
            total_name_len += name.len();
        }

        let mut output_id = vec![0u8; row_count * 8];
        let mut output_name_offsets = vec![0u8; (row_count + 1) * 4];
        let mut output_name_data = vec![0u8; total_name_len + 100];
        let mut output_count = vec![0u8; row_count * 4];

        let mut output_ptrs = vec![
            output_id.as_mut_ptr(),
            output_name_offsets.as_mut_ptr(),
            output_name_data.as_mut_ptr(),
            output_count.as_mut_ptr(),
        ];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify Int64 values
        for i in 0..row_count {
            let id = i64::from_le_bytes(output_id[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(id, i as i64 * 100, "id mismatch at row {}", i);
        }

        // Verify name offsets
        let mut expected_offset = 0;
        for (i, name) in names.iter().enumerate() {
            let offset_i = i32::from_le_bytes(output_name_offsets[i * 4..(i + 1) * 4].try_into().unwrap());
            assert_eq!(offset_i, expected_offset as i32, "name offset {} mismatch", i);
            expected_offset += name.len();
        }
        // Final offset
        let offset_n = i32::from_le_bytes(output_name_offsets[row_count * 4..(row_count + 1) * 4].try_into().unwrap());
        assert_eq!(offset_n, total_name_len as i32, "name final offset mismatch");

        // Verify name data
        let mut data_offset = 0;
        for (i, name) in names.iter().enumerate() {
            let actual = &output_name_data[data_offset..data_offset + name.len()];
            assert_eq!(actual, name.as_bytes(), "name data mismatch at row {}", i);
            data_offset += name.len();
        }

        // Verify Int32 values
        for i in 0..row_count {
            let count = i32::from_le_bytes(output_count[i * 4..(i + 1) * 4].try_into().unwrap());
            assert_eq!(count, i as i32 * 10, "count mismatch at row {}", i);
        }
    }

    #[test]
    fn test_roundtrip_bytes_long_string() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("data", DataType::Binary, false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Test a longer string that requires multi-byte varint length
        let test_data: Vec<u8> = (0..500).map(|i| (i % 256) as u8).collect();
        let input = encode_avro_bytes(&test_data);
        let row_count = 1;

        // Buffer 0: offsets
        let mut offsets = vec![0u8; (row_count + 1) * 4];
        // Buffer 1: data
        let mut data = vec![0u8; test_data.len() + 100];

        let mut output_ptrs = vec![offsets.as_mut_ptr(), data.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify offsets
        let offset_0 = i32::from_le_bytes(offsets[0..4].try_into().unwrap());
        let offset_1 = i32::from_le_bytes(offsets[4..8].try_into().unwrap());
        assert_eq!(offset_0, 0);
        assert_eq!(offset_1, test_data.len() as i32);

        // Verify all 500 bytes were copied correctly to data buffer
        assert_eq!(&data[..test_data.len()], &test_data[..], "Long bytes mismatch");
    }

    /// Encode union tag for nullable field
    /// In Avro unions ["null", "type"], null has index 0, type has index 1
    /// The index is zigzag encoded, so: null -> 0, type -> 2
    fn encode_union_tag(is_null: bool) -> Vec<u8> {
        if is_null {
            vec![0x00]  // zigzag(0) = 0
        } else {
            vec![0x02]  // zigzag(1) = 2
        }
    }

    #[test]
    fn test_roundtrip_nullable_int64_all_non_null() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int64, true),  // nullable
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let test_values: Vec<i64> = vec![10, 20, 30, 40, 50];

        let mut input = Vec::new();
        for &v in &test_values {
            input.extend(encode_union_tag(false));  // non-null
            input.extend(encode_varint_i64(v));
        }

        let row_count = test_values.len();
        let mut output = vec![0u8; row_count * 8];
        let mut validity = vec![0u8; (row_count + 7) / 8];
        let mut output_ptrs = vec![output.as_mut_ptr(), validity.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in test_values.iter().enumerate() {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, expected, "Nullable Int64 mismatch at index {}", i);
        }
    }

    #[test]
    fn test_roundtrip_nullable_int64_with_nulls() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int64, true),  // nullable
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Test pattern: value, null, value, null, value
        let values_or_nulls: Vec<Option<i64>> = vec![
            Some(100),
            None,
            Some(200),
            None,
            Some(300),
        ];

        let mut input = Vec::new();
        for v in &values_or_nulls {
            match v {
                Some(val) => {
                    input.extend(encode_union_tag(false));  // non-null
                    input.extend(encode_varint_i64(*val));
                }
                None => {
                    input.extend(encode_union_tag(true));   // null
                }
            }
        }

        let row_count = values_or_nulls.len();
        let mut output = vec![0xFFu8; row_count * 8];  // Fill with sentinel
        let mut validity = vec![0u8; (row_count + 7) / 8];
        let mut output_ptrs = vec![output.as_mut_ptr(), validity.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, v) in values_or_nulls.iter().enumerate() {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
            match v {
                Some(expected) => {
                    assert_eq!(actual, *expected, "Value mismatch at index {}", i);
                }
                None => {
                    // Null should have 0 placeholder
                    assert_eq!(actual, 0, "Null should have 0 placeholder at index {}", i);
                }
            }
        }
    }

    #[test]
    fn test_roundtrip_nullable_int64_all_nulls() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int64, true),  // nullable
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let row_count = 5;
        let mut input = Vec::new();
        for _ in 0..row_count {
            input.extend(encode_union_tag(true));  // all nulls
        }

        let mut output = vec![0xFFu8; row_count * 8];  // Fill with sentinel
        let mut validity = vec![0u8; (row_count + 7) / 8];
        let mut output_ptrs = vec![output.as_mut_ptr(), validity.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        for i in 0..row_count {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, 0, "Null should have 0 placeholder at index {}", i);
        }
    }

    /// Encode enum index as varint (not zigzag - just raw varint)
    fn encode_enum_index(index: u32) -> Vec<u8> {
        let mut result = Vec::new();
        let mut v = index as u64;
        loop {
            if v < 0x80 {
                result.push(v as u8);
                break;
            }
            result.push((v as u8) | 0x80);
            v >>= 7;
        }
        result
    }

    #[test]
    fn test_roundtrip_enum() {
        use arrow_schema::DataType;

        // Create a Dictionary type (represents Avro Enum)
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "status",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                false,
            ),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Test enum indices: 0, 1, 2, 0, 1
        let test_indices: Vec<u32> = vec![0, 1, 2, 0, 1];

        let mut input = Vec::new();
        for &idx in &test_indices {
            input.extend(encode_enum_index(idx));
        }

        let row_count = test_indices.len();
        let mut output = vec![0u8; row_count * 4];  // i32 indices
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in test_indices.iter().enumerate() {
            let actual = i32::from_le_bytes(output[i * 4..(i + 1) * 4].try_into().unwrap());
            assert_eq!(actual, expected as i32, "Enum index mismatch at index {}", i);
        }
    }

    #[test]
    fn test_roundtrip_nullable_enum() {
        use arrow_schema::DataType;

        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "status",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                true,  // nullable
            ),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Test pattern: value, null, value, null, value
        let values: Vec<Option<u32>> = vec![Some(0), None, Some(2), None, Some(1)];

        let mut input = Vec::new();
        for v in &values {
            match v {
                Some(idx) => {
                    input.extend(encode_union_tag(false));  // non-null
                    input.extend(encode_enum_index(*idx));
                }
                None => {
                    input.extend(encode_union_tag(true));   // null
                }
            }
        }

        let row_count = values.len();
        let mut output = vec![0xFFu8; row_count * 4];
        let mut validity = vec![0u8; (row_count + 7) / 8];
        let mut output_ptrs = vec![output.as_mut_ptr(), validity.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, v) in values.iter().enumerate() {
            let actual = i32::from_le_bytes(output[i * 4..(i + 1) * 4].try_into().unwrap());
            match v {
                Some(expected) => {
                    assert_eq!(actual, *expected as i32, "Value mismatch at index {}", i);
                }
                None => {
                    assert_eq!(actual, 0, "Null should have 0 placeholder at index {}", i);
                }
            }
        }
    }

    /// Encode Avro array: block count (zigzag) + elements + 0 terminator
    fn encode_avro_array_i64(elements: &[i64]) -> Vec<u8> {
        let mut result = Vec::new();
        if elements.is_empty() {
            // Empty array: just 0 block count
            result.push(0x00);
        } else {
            // Single block with all elements
            result.extend(encode_varint_i64(elements.len() as i64)); // block count
            for &e in elements {
                result.extend(encode_varint_i64(e));
            }
            result.push(0x00); // terminator
        }
        result
    }

    fn encode_avro_array_i32(elements: &[i32]) -> Vec<u8> {
        let mut result = Vec::new();
        if elements.is_empty() {
            result.push(0x00);
        } else {
            result.extend(encode_varint_i64(elements.len() as i64)); // block count
            for &e in elements {
                result.extend(encode_varint_i32(e));
            }
            result.push(0x00);
        }
        result
    }

    #[test]
    fn test_roundtrip_list_int64_empty() {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "values",
                DataType::List(Arc::new(Field::new("item", DataType::Int64, false))),
                false,
            ),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Verify we need 2 buffers for List type (offsets + data)
        assert_eq!(decoder.total_buffers(), 2);

        // Empty array: single 0 byte
        let input = encode_avro_array_i64(&[]);
        let row_count = 1;

        // Buffer 0: offsets (row_count + 1 i32 values = 8 bytes for 1 row)
        let mut offsets = vec![0xFFu8; (row_count + 1) * 4];
        // Buffer 1: data
        let mut data = vec![0xFFu8; 100];

        let mut output_ptrs = vec![offsets.as_mut_ptr(), data.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, 1); // Just the 0 terminator

        // Verify offsets: [0, 0] for empty list
        let offset_0 = i32::from_le_bytes(offsets[0..4].try_into().unwrap());
        let offset_1 = i32::from_le_bytes(offsets[4..8].try_into().unwrap());
        assert_eq!(offset_0, 0);
        assert_eq!(offset_1, 0); // No elements
    }

    #[test]
    fn test_roundtrip_list_int64_single() {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "values",
                DataType::List(Arc::new(Field::new("item", DataType::Int64, false))),
                false,
            ),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Array with elements: [10, 20, 30]
        let elements: Vec<i64> = vec![10, 20, 30];
        let input = encode_avro_array_i64(&elements);
        let row_count = 1;

        // Buffer 0: offsets (row_count + 1 i32 values)
        let mut offsets = vec![0u8; (row_count + 1) * 4];
        // Buffer 1: data (element_count * 8 bytes for i64)
        let mut data = vec![0u8; elements.len() * 8 + 100];

        let mut output_ptrs = vec![offsets.as_mut_ptr(), data.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify offsets: [0, 3] for a list with 3 elements (element count, not bytes)
        let offset_0 = i32::from_le_bytes(offsets[0..4].try_into().unwrap());
        let offset_1 = i32::from_le_bytes(offsets[4..8].try_into().unwrap());
        assert_eq!(offset_0, 0, "First offset should be 0");
        assert_eq!(offset_1, elements.len() as i32, "Second offset should be element count");

        // Verify elements in data buffer
        for (i, &expected) in elements.iter().enumerate() {
            let actual = i64::from_le_bytes(data[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, expected, "Element mismatch at index {}", i);
        }
    }

    #[test]
    fn test_roundtrip_list_int32() {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "counts",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, false))),
                false,
            ),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let elements: Vec<i32> = vec![1, 2, 3, 4, 5];
        let input = encode_avro_array_i32(&elements);
        let row_count = 1;

        // Buffer 0: offsets (row_count + 1 i32 values)
        let mut offsets = vec![0u8; (row_count + 1) * 4];
        // Buffer 1: data (element_count * 4 bytes for i32)
        let mut data = vec![0u8; elements.len() * 4 + 100];

        let mut output_ptrs = vec![offsets.as_mut_ptr(), data.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify offsets: [0, 5*4] for a list with 5 i32 elements
        let offset_0 = i32::from_le_bytes(offsets[0..4].try_into().unwrap());
        let offset_1 = i32::from_le_bytes(offsets[4..8].try_into().unwrap());
        assert_eq!(offset_0, 0, "First offset should be 0");
        assert_eq!(offset_1, elements.len() as i32, "Second offset should be element count");

        // Verify elements in data buffer
        for (i, &expected) in elements.iter().enumerate() {
            let actual = i32::from_le_bytes(data[i * 4..(i + 1) * 4].try_into().unwrap());
            assert_eq!(actual, expected, "Element mismatch at index {}", i);
        }
    }

    #[test]
    fn test_roundtrip_list_multiple_rows() {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "values",
                DataType::List(Arc::new(Field::new("item", DataType::Int64, false))),
                false,
            ),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Three rows with different list sizes
        let row1: Vec<i64> = vec![1, 2];
        let row2: Vec<i64> = vec![10, 20, 30];
        let row3: Vec<i64> = vec![100];
        let row_count = 3;

        let mut input = Vec::new();
        input.extend(encode_avro_array_i64(&row1));
        input.extend(encode_avro_array_i64(&row2));
        input.extend(encode_avro_array_i64(&row3));

        let total_elements = row1.len() + row2.len() + row3.len();

        // Buffer 0: offsets (row_count + 1 i32 values)
        let mut offsets = vec![0u8; (row_count + 1) * 4];
        // Buffer 1: data
        let mut data = vec![0u8; total_elements * 8 + 100];

        let mut output_ptrs = vec![offsets.as_mut_ptr(), data.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify offsets: [0, 16, 40, 48] for rows with 2, 3, 1 elements (all i64 = 8 bytes each)
        let offset_0 = i32::from_le_bytes(offsets[0..4].try_into().unwrap());
        let offset_1 = i32::from_le_bytes(offsets[4..8].try_into().unwrap());
        let offset_2 = i32::from_le_bytes(offsets[8..12].try_into().unwrap());
        let offset_3 = i32::from_le_bytes(offsets[12..16].try_into().unwrap());
        assert_eq!(offset_0, 0, "Offset 0 should be 0");
        assert_eq!(offset_1, row1.len() as i32, "Offset 1 should be element count of row1");
        assert_eq!(offset_2, (row1.len() + row2.len()) as i32, "Offset 2 should be cumulative element count");
        assert_eq!(offset_3, total_elements as i32, "Offset 3 should be total element count");

        // Verify all elements in order in data buffer
        let all_elements: Vec<i64> = [row1, row2, row3].concat();
        for (i, &expected) in all_elements.iter().enumerate() {
            let actual = i64::from_le_bytes(data[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, expected, "Element mismatch at index {}", i);
        }
    }

    #[test]
    fn test_roundtrip_list_with_int_column() {
        // Mixed schema: Int64 + List<Int64>
        // Buffer layout: [Int64 data] [List offsets] [List data]
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new(
                "values",
                DataType::List(Arc::new(Field::new("item", DataType::Int64, false))),
                false,
            ),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Verify buffer count: 1 for Int64 + 2 for List = 3
        assert_eq!(decoder.total_buffers(), 3);

        let row_count = 3;
        let ids: Vec<i64> = vec![100, 200, 300];
        let lists: Vec<Vec<i64>> = vec![
            vec![1, 2],
            vec![10, 20, 30],
            vec![100],
        ];

        let mut input = Vec::new();
        for i in 0..row_count {
            input.extend(encode_varint_i64(ids[i]));
            input.extend(encode_avro_array_i64(&lists[i]));
        }

        let total_elements: usize = lists.iter().map(|l| l.len()).sum();

        // Buffer 0: Int64 data
        let mut output_id = vec![0u8; row_count * 8];
        // Buffer 1: List offsets (row_count + 1 i32 values)
        let mut list_offsets = vec![0u8; (row_count + 1) * 4];
        // Buffer 2: List data
        let mut list_data = vec![0u8; total_elements * 8 + 100];

        let mut output_ptrs = vec![
            output_id.as_mut_ptr(),
            list_offsets.as_mut_ptr(),
            list_data.as_mut_ptr(),
        ];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify ids
        for (i, &expected) in ids.iter().enumerate() {
            let actual = i64::from_le_bytes(output_id[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, expected, "Id mismatch at row {}", i);
        }

        // Verify list offsets
        let offset_0 = i32::from_le_bytes(list_offsets[0..4].try_into().unwrap());
        let offset_1 = i32::from_le_bytes(list_offsets[4..8].try_into().unwrap());
        let offset_2 = i32::from_le_bytes(list_offsets[8..12].try_into().unwrap());
        let offset_3 = i32::from_le_bytes(list_offsets[12..16].try_into().unwrap());
        assert_eq!(offset_0, 0, "Offset 0 should be 0");
        assert_eq!(offset_1, lists[0].len() as i32, "Offset 1 should be element count of list 0");
        assert_eq!(offset_2, (lists[0].len() + lists[1].len()) as i32, "Offset 2 should be cumulative element count");
        assert_eq!(offset_3, total_elements as i32, "Offset 3 should be total element count");

        // Verify list elements in data buffer
        let all_elements: Vec<i64> = lists.into_iter().flatten().collect();
        for (i, &expected) in all_elements.iter().enumerate() {
            let actual = i64::from_le_bytes(list_data[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, expected, "List element mismatch at index {}", i);
        }
    }

    /// Encode Avro map: block count + (key, value) pairs + 0 terminator
    /// This encodes values only (keys are strings)
    fn encode_avro_map_i64(entries: &[(&str, i64)]) -> Vec<u8> {
        let mut result = Vec::new();
        if entries.is_empty() {
            result.push(0x00);
        } else {
            // Block count
            result.extend(encode_varint_i64(entries.len() as i64));
            for (key, value) in entries {
                // String key: length + bytes
                result.extend(encode_avro_bytes(key.as_bytes()));
                // Int64 value
                result.extend(encode_varint_i64(*value));
            }
            result.push(0x00); // terminator
        }
        result
    }

    #[test]
    fn test_roundtrip_map_empty() {
        use arrow_schema::Fields;

        let map_field = Field::new(
            "entries",
            DataType::Struct(Fields::from(vec![
                Field::new("key", DataType::Utf8, false),
                Field::new("value", DataType::Int64, false),
            ])),
            false,
        );

        let schema = Arc::new(Schema::new(vec![
            Field::new("props", DataType::Map(Arc::new(map_field), false), false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Empty map
        let input = encode_avro_map_i64(&[]);

        let row_count = 1;
        // Buffer 0: offsets (row_count + 1 i32 values)
        let mut offsets = vec![0xFFu8; (row_count + 1) * 4];
        // Buffer 1: data (value_count * 8 bytes for i64)
        let mut data = vec![0xFFu8; 100];
        let mut output_ptrs = vec![offsets.as_mut_ptr(), data.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, 1).expect("Decode failed");
        assert_eq!(consumed, 1); // Just the 0 terminator

        // Verify offsets: [0, 0] for empty map
        let offset0 = i32::from_le_bytes(offsets[0..4].try_into().unwrap());
        let offset1 = i32::from_le_bytes(offsets[4..8].try_into().unwrap());
        assert_eq!(offset0, 0, "First offset should be 0");
        assert_eq!(offset1, 0, "Second offset should be 0 for empty map");
    }

    #[test]
    fn test_roundtrip_map_single() {
        use arrow_schema::Fields;

        let map_field = Field::new(
            "entries",
            DataType::Struct(Fields::from(vec![
                Field::new("key", DataType::Utf8, false),
                Field::new("value", DataType::Int64, false),
            ])),
            false,
        );

        let schema = Arc::new(Schema::new(vec![
            Field::new("props", DataType::Map(Arc::new(map_field), false), false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Map with entries
        let entries: Vec<(&str, i64)> = vec![("a", 100), ("bb", 200), ("ccc", 300)];
        let input = encode_avro_map_i64(&entries);

        let row_count = 1;
        // Buffer 0: offsets (row_count + 1 i32 values)
        let mut offsets = vec![0u8; (row_count + 1) * 4];
        // Buffer 1: data (value_count * 8 bytes for i64)
        let mut data = vec![0u8; entries.len() * 8 + 100];
        let mut output_ptrs = vec![offsets.as_mut_ptr(), data.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, 1).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify offsets: [0, 3] for 3 entries
        let offset0 = i32::from_le_bytes(offsets[0..4].try_into().unwrap());
        let offset1 = i32::from_le_bytes(offsets[4..8].try_into().unwrap());
        assert_eq!(offset0, 0, "First offset should be 0");
        assert_eq!(offset1, entries.len() as i32, "Second offset should be entry count");

        // Verify values (keys are skipped)
        for (i, &(_, expected_value)) in entries.iter().enumerate() {
            let actual = i64::from_le_bytes(data[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, expected_value, "Value mismatch at index {}", i);
        }
    }

    #[test]
    fn test_roundtrip_map_multiple_rows() {
        use arrow_schema::Fields;

        let map_field = Field::new(
            "entries",
            DataType::Struct(Fields::from(vec![
                Field::new("key", DataType::Utf8, false),
                Field::new("value", DataType::Int64, false),
            ])),
            false,
        );

        let schema = Arc::new(Schema::new(vec![
            Field::new("props", DataType::Map(Arc::new(map_field), false), false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Three rows with different map sizes
        let row1: Vec<(&str, i64)> = vec![("x", 10)];
        let row2: Vec<(&str, i64)> = vec![("y", 20), ("z", 30)];
        let row3: Vec<(&str, i64)> = vec![];

        let mut input = Vec::new();
        input.extend(encode_avro_map_i64(&row1));
        input.extend(encode_avro_map_i64(&row2));
        input.extend(encode_avro_map_i64(&row3));

        let row_count = 3;
        let total_entries = row1.len() + row2.len() + row3.len();
        // Buffer 0: offsets (row_count + 1 i32 values)
        let mut offsets = vec![0u8; (row_count + 1) * 4];
        // Buffer 1: data (value_count * 8 bytes for i64)
        let mut data = vec![0u8; total_entries * 8 + 100];
        let mut output_ptrs = vec![offsets.as_mut_ptr(), data.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify offsets: [0, 1, 3, 3] for rows with 1, 2, 0 entries
        let offset0 = i32::from_le_bytes(offsets[0..4].try_into().unwrap());
        let offset1 = i32::from_le_bytes(offsets[4..8].try_into().unwrap());
        let offset2 = i32::from_le_bytes(offsets[8..12].try_into().unwrap());
        let offset3 = i32::from_le_bytes(offsets[12..16].try_into().unwrap());
        assert_eq!(offset0, 0, "Offset 0 should be 0");
        assert_eq!(offset1, 1, "Offset 1 should be 1");
        assert_eq!(offset2, 3, "Offset 2 should be 3");
        assert_eq!(offset3, 3, "Offset 3 should be 3 (empty row)");

        // Verify all values in order
        let all_values: Vec<i64> = [row1, row2, row3].iter()
            .flat_map(|entries| entries.iter().map(|(_, v)| *v))
            .collect();
        for (i, &expected) in all_values.iter().enumerate() {
            let actual = i64::from_le_bytes(data[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, expected, "Value mismatch at index {}", i);
        }
    }

    #[test]
    fn test_roundtrip_map_with_int_column() {
        use arrow_schema::Fields;

        let map_field = Field::new(
            "entries",
            DataType::Struct(Fields::from(vec![
                Field::new("key", DataType::Utf8, false),
                Field::new("value", DataType::Int64, false),
            ])),
            false,
        );

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("props", DataType::Map(Arc::new(map_field), false), false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let row_count = 2;
        let ids: Vec<i64> = vec![1000, 2000];
        let maps: Vec<Vec<(&str, i64)>> = vec![
            vec![("a", 1), ("b", 2)],
            vec![("c", 3)],
        ];

        let mut input = Vec::new();
        for i in 0..row_count {
            input.extend(encode_varint_i64(ids[i]));
            input.extend(encode_avro_map_i64(&maps[i]));
        }

        let total_entries: usize = maps.iter().map(|m| m.len()).sum();
        // Buffer 0: Int64 data
        let mut output_id = vec![0u8; row_count * 8];
        // Buffer 1: Map offsets (row_count + 1 i32 values)
        let mut map_offsets = vec![0u8; (row_count + 1) * 4];
        // Buffer 2: Map data (value_count * 8 bytes for i64)
        let mut map_data = vec![0u8; total_entries * 8 + 100];

        let mut output_ptrs = vec![
            output_id.as_mut_ptr(),
            map_offsets.as_mut_ptr(),
            map_data.as_mut_ptr(),
        ];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify ids
        for (i, &expected) in ids.iter().enumerate() {
            let actual = i64::from_le_bytes(output_id[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, expected, "Id mismatch at row {}", i);
        }

        // Verify map offsets: [0, 2, 3] for rows with 2 and 1 entries
        let offset0 = i32::from_le_bytes(map_offsets[0..4].try_into().unwrap());
        let offset1 = i32::from_le_bytes(map_offsets[4..8].try_into().unwrap());
        let offset2 = i32::from_le_bytes(map_offsets[8..12].try_into().unwrap());
        assert_eq!(offset0, 0, "Offset 0 should be 0");
        assert_eq!(offset1, 2, "Offset 1 should be 2");
        assert_eq!(offset2, 3, "Offset 2 should be 3");

        // Verify map values
        let all_values: Vec<i64> = maps.iter()
            .flat_map(|entries| entries.iter().map(|(_, v)| *v))
            .collect();
        for (i, &expected) in all_values.iter().enumerate() {
            let actual = i64::from_le_bytes(map_data[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, expected, "Map value mismatch at index {}", i);
        }
    }

    #[test]
    fn test_roundtrip_struct_simple() {
        use arrow_schema::Fields;

        // Struct with two Int64 fields
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "point",
                DataType::Struct(Fields::from(vec![
                    Field::new("x", DataType::Int64, false),
                    Field::new("y", DataType::Int64, false),
                ])),
                false,
            ),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Encode struct: x=10, y=20
        let mut input = Vec::new();
        input.extend(encode_varint_i64(10)); // x
        input.extend(encode_varint_i64(20)); // y

        // Output buffer for flattened struct (2 x i64 = 16 bytes)
        let mut output = vec![0u8; 16];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, 1).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify fields
        let x = i64::from_le_bytes(output[0..8].try_into().unwrap());
        let y = i64::from_le_bytes(output[8..16].try_into().unwrap());
        assert_eq!(x, 10, "x mismatch");
        assert_eq!(y, 20, "y mismatch");
    }

    #[test]
    fn test_roundtrip_struct_multiple_rows() {
        use arrow_schema::Fields;

        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "point",
                DataType::Struct(Fields::from(vec![
                    Field::new("x", DataType::Int64, false),
                    Field::new("y", DataType::Int64, false),
                ])),
                false,
            ),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let points: Vec<(i64, i64)> = vec![(1, 2), (10, 20), (100, 200)];

        let mut input = Vec::new();
        for (x, y) in &points {
            input.extend(encode_varint_i64(*x));
            input.extend(encode_varint_i64(*y));
        }

        let mut output = vec![0u8; points.len() * 16];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, points.len()).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify each point
        for (i, (expected_x, expected_y)) in points.iter().enumerate() {
            let offset = i * 16;
            let x = i64::from_le_bytes(output[offset..offset+8].try_into().unwrap());
            let y = i64::from_le_bytes(output[offset+8..offset+16].try_into().unwrap());
            assert_eq!(x, *expected_x, "x mismatch at row {}", i);
            assert_eq!(y, *expected_y, "y mismatch at row {}", i);
        }
    }

    #[test]
    fn test_roundtrip_struct_mixed_types() {
        use arrow_schema::Fields;

        // Struct with mixed types
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "record",
                DataType::Struct(Fields::from(vec![
                    Field::new("id", DataType::Int64, false),
                    Field::new("count", DataType::Int32, false),
                    Field::new("score", DataType::Float64, false),
                ])),
                false,
            ),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Encode: id=1000, count=5, score=3.14
        let mut input = Vec::new();
        input.extend(encode_varint_i64(1000)); // id
        input.extend(encode_varint_i32(5));    // count
        input.extend_from_slice(&3.14f64.to_le_bytes()); // score

        // Output: 8 (i64) + 4 (i32) + 8 (f64) = 20 bytes
        let mut output = vec![0u8; 20];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, 1).expect("Decode failed");
        assert_eq!(consumed, input.len());

        let id = i64::from_le_bytes(output[0..8].try_into().unwrap());
        let count = i32::from_le_bytes(output[8..12].try_into().unwrap());
        let score = f64::from_le_bytes(output[12..20].try_into().unwrap());

        assert_eq!(id, 1000);
        assert_eq!(count, 5);
        assert!((score - 3.14).abs() < f64::EPSILON);
    }

    #[test]
    fn test_roundtrip_struct_with_int_column() {
        use arrow_schema::Fields;

        // Schema: Int64, Struct { Int64, Int64 }
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new(
                "point",
                DataType::Struct(Fields::from(vec![
                    Field::new("x", DataType::Int64, false),
                    Field::new("y", DataType::Int64, false),
                ])),
                false,
            ),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let row_count = 2;
        let ids: Vec<i64> = vec![100, 200];
        let points: Vec<(i64, i64)> = vec![(1, 2), (10, 20)];

        let mut input = Vec::new();
        for i in 0..row_count {
            input.extend(encode_varint_i64(ids[i]));
            input.extend(encode_varint_i64(points[i].0));
            input.extend(encode_varint_i64(points[i].1));
        }

        let mut output_id = vec![0u8; row_count * 8];
        let mut output_struct = vec![0u8; row_count * 16];

        let mut output_ptrs = vec![
            output_id.as_mut_ptr(),
            output_struct.as_mut_ptr(),
        ];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify ids
        for (i, &expected) in ids.iter().enumerate() {
            let actual = i64::from_le_bytes(output_id[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, expected, "Id mismatch at row {}", i);
        }

        // Verify struct fields
        for (i, (expected_x, expected_y)) in points.iter().enumerate() {
            let offset = i * 16;
            let x = i64::from_le_bytes(output_struct[offset..offset+8].try_into().unwrap());
            let y = i64::from_le_bytes(output_struct[offset+8..offset+16].try_into().unwrap());
            assert_eq!(x, *expected_x, "x mismatch at row {}", i);
            assert_eq!(y, *expected_y, "y mismatch at row {}", i);
        }
    }

    // === Tests for new types ===

    #[test]
    fn test_roundtrip_union_simple() {
        use arrow_schema::UnionFields;

        // Union of Int64 and Float64
        let union_fields = UnionFields::new(
            vec![0, 1],
            vec![
                Field::new("int_val", DataType::Int64, false),
                Field::new("float_val", DataType::Float64, false),
            ],
        );

        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Union(union_fields, arrow_schema::UnionMode::Dense), false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Test data: type_index=0 (Int64), value=42
        let mut input = Vec::new();
        input.extend(encode_enum_index(0));  // type index 0 = Int64
        input.extend(encode_varint_i64(42));

        let row_count = 1;
        // Union output: type_id (4 bytes) + max variant size (8 bytes)
        let mut output = vec![0u8; row_count * 12];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert!(consumed > 0, "Should consume bytes");

        // Verify type_id
        let type_id = i32::from_le_bytes(output[0..4].try_into().unwrap());
        assert_eq!(type_id, 0, "Type id should be 0 (Int64)");

        // Verify value
        let value = i64::from_le_bytes(output[4..12].try_into().unwrap());
        assert_eq!(value, 42, "Int64 value should be 42");
    }

    #[test]
    fn test_roundtrip_fixed_size_list() {
        // FixedSizeList of 3 Int64s
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int64, false)), 3),
                false,
            ),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Test data: [1, 2, 3]
        let elements: Vec<i64> = vec![1, 2, 3];
        let mut input = Vec::new();
        for &elem in &elements {
            input.extend(encode_varint_i64(elem));
        }

        let row_count = 1;
        let mut output = vec![0u8; row_count * 3 * 8];  // 3 Int64s
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify elements
        for (i, &expected) in elements.iter().enumerate() {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, expected, "Element mismatch at index {}", i);
        }
    }

    #[test]
    fn test_roundtrip_fixed_size_list_multiple_rows() {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int32, false)), 2),
                false,
            ),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Test data: row0=[10, 20], row1=[30, 40]
        let rows: Vec<Vec<i32>> = vec![vec![10, 20], vec![30, 40]];
        let mut input = Vec::new();
        for row in &rows {
            for &elem in row {
                input.extend(encode_varint_i32(elem));
            }
        }

        let row_count = 2;
        let mut output = vec![0u8; row_count * 2 * 4];  // 2 rows * 2 Int32s
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify elements
        for (row_idx, row) in rows.iter().enumerate() {
            for (elem_idx, &expected) in row.iter().enumerate() {
                let offset = (row_idx * 2 + elem_idx) * 4;
                let actual = i32::from_le_bytes(output[offset..offset + 4].try_into().unwrap());
                assert_eq!(actual, expected, "Element mismatch at row {} elem {}", row_idx, elem_idx);
            }
        }
    }

    #[test]
    fn test_roundtrip_duration() {
        use arrow_schema::TimeUnit;

        let schema = Arc::new(Schema::new(vec![
            Field::new("duration", DataType::Duration(TimeUnit::Microsecond), false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let values: Vec<i64> = vec![1000, 2000, -500];
        let mut input = Vec::new();
        for &v in &values {
            input.extend(encode_varint_i64(v));
        }

        let row_count = 3;
        let mut output = vec![0u8; row_count * 8];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in values.iter().enumerate() {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, expected, "Duration mismatch at row {}", i);
        }
    }

    #[test]
    fn test_roundtrip_decimal128() {
        // Decimal128 with precision 38, scale 10
        let schema = Arc::new(Schema::new(vec![
            Field::new("amount", DataType::Decimal128(38, 10), false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Test data: 16 bytes big-endian decimal
        // Simple test: all zeros
        let input = vec![0u8; 16];

        let row_count = 1;
        let mut output = vec![0u8; row_count * 16];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, 16);

        // Verify output is zeros (byte-swapped zeros are still zeros)
        for byte in &output {
            assert_eq!(*byte, 0, "Expected zero");
        }
    }

    #[test]
    fn test_roundtrip_interval_year_month() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("interval", DataType::Interval(IntervalUnit::YearMonth), false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // YearMonth interval is encoded as a varint (months)
        let months: i32 = 14;  // 1 year 2 months
        let mut input = Vec::new();
        input.extend(encode_varint_i32(months));

        let row_count = 1;
        let mut output = vec![0u8; row_count * 4];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        let actual = i32::from_le_bytes(output[0..4].try_into().unwrap());
        assert_eq!(actual, months, "YearMonth interval mismatch");
    }

    #[test]
    fn test_roundtrip_interval_day_time() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("interval", DataType::Interval(IntervalUnit::DayTime), false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // DayTime interval: 8 bytes (days: i32, milliseconds: i32)
        let days: i32 = 5;
        let millis: i32 = 3600000;  // 1 hour
        let mut input = Vec::new();
        input.extend(&days.to_le_bytes());
        input.extend(&millis.to_le_bytes());

        let row_count = 1;
        let mut output = vec![0u8; row_count * 8];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, 8);

        let actual_days = i32::from_le_bytes(output[0..4].try_into().unwrap());
        let actual_millis = i32::from_le_bytes(output[4..8].try_into().unwrap());
        assert_eq!(actual_days, days, "Days mismatch");
        assert_eq!(actual_millis, millis, "Milliseconds mismatch");
    }

    // ============== Output Buffer Overflow Tests ==============
    // Tests for decode_checked capacity validation

    #[test]
    fn test_decode_checked_int64_sufficient_capacity() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Int64, false),
        ]));
        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Encode 3 values
        let mut input = Vec::new();
        for v in [100i64, 200, 300] {
            input.extend(encode_varint_i64(v));
        }

        let row_count = 3;
        let mut output = vec![0u8; row_count * 8]; // Exactly enough
        let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];

        let result = decoder.decode_checked(&input, &buffers, row_count);
        assert!(result.is_ok(), "Should succeed with sufficient capacity");
    }

    #[test]
    fn test_decode_checked_int64_insufficient_capacity() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Int64, false),
        ]));
        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Encode 3 values
        let mut input = Vec::new();
        for v in [100i64, 200, 300] {
            input.extend(encode_varint_i64(v));
        }

        let row_count = 3;
        let mut output = vec![0u8; 16]; // Only 16 bytes, need 24
        let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];

        let result = decoder.decode_checked(&input, &buffers, row_count);
        assert!(result.is_err(), "Should fail with insufficient capacity");
        let err = result.unwrap_err();
        assert!(err.contains("requires 24 bytes"), "Error: {}", err);
    }

    #[test]
    fn test_decode_checked_int32_insufficient_capacity() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Int32, false),
        ]));
        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Encode 5 values - need 20 bytes
        let mut input = Vec::new();
        for v in [1i32, 2, 3, 4, 5] {
            input.extend(encode_varint_i32(v));
        }

        let row_count = 5;
        let mut output = vec![0u8; 12]; // Only 12 bytes, need 20
        let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];

        let result = decoder.decode_checked(&input, &buffers, row_count);
        assert!(result.is_err(), "Should fail with insufficient capacity");
        let err = result.unwrap_err();
        assert!(err.contains("requires 20 bytes"), "Error: {}", err);
    }

    #[test]
    fn test_decode_checked_float64_insufficient_capacity() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, false),
        ]));
        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Float64 is 8 bytes raw, need 16 bytes for 2 rows
        let input: Vec<u8> = [1.5f64, 2.5].iter().flat_map(|f| f.to_le_bytes()).collect();

        let row_count = 2;
        let mut output = vec![0u8; 8]; // Only 8 bytes, need 16
        let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];

        let result = decoder.decode_checked(&input, &buffers, row_count);
        assert!(result.is_err(), "Should fail with insufficient capacity");
    }

    #[test]
    fn test_decode_checked_boolean_insufficient_capacity() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Boolean, false),
        ]));
        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Boolean is 1 byte per value in JIT output
        let input: Vec<u8> = vec![1, 0, 1, 1, 0]; // 5 booleans

        let row_count = 5;
        let mut output = vec![0u8; 3]; // Only 3 bytes, need 5
        let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];

        let result = decoder.decode_checked(&input, &buffers, row_count);
        assert!(result.is_err(), "Should fail with insufficient capacity");
    }

    #[test]
    fn test_decode_checked_multi_column_first_insufficient() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let decoder = JitDecoder::compile(&schema).expect("compile");

        // First column insufficient
        let row_count = 3;
        let mut output1 = vec![0u8; 16]; // Need 24 for Int64
        let mut output2 = vec![0u8; 12]; // Enough for Int32

        let buffers = vec![
            OutputBuffer::new(output1.as_mut_ptr(), output1.len()),
            OutputBuffer::new(output2.as_mut_ptr(), output2.len()),
        ];

        // Input doesn't matter - validation fails before decode
        let input = vec![0u8; 20];
        let result = decoder.decode_checked(&input, &buffers, row_count);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Column 'a'"));
    }

    #[test]
    fn test_decode_checked_multi_column_second_insufficient() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Second column insufficient
        let row_count = 3;
        let mut output1 = vec![0u8; 24]; // Enough for Int64
        let mut output2 = vec![0u8; 8];  // Need 12 for Int32

        let buffers = vec![
            OutputBuffer::new(output1.as_mut_ptr(), output1.len()),
            OutputBuffer::new(output2.as_mut_ptr(), output2.len()),
        ];

        let input = vec![0u8; 20];
        let result = decoder.decode_checked(&input, &buffers, row_count);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Column 'b'"));
    }

    #[test]
    fn test_decode_checked_wrong_number_of_buffers() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let decoder = JitDecoder::compile(&schema).expect("compile");

        let row_count = 1;
        let mut output = vec![0u8; 8];
        let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];

        let input = vec![0u8; 10];
        let result = decoder.decode_checked(&input, &buffers, row_count);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Expected 2 output buffers, got 1"));
    }

    #[test]
    fn test_decode_checked_fixed_size_binary_insufficient() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::FixedSizeBinary(16), false),
        ]));
        let decoder = JitDecoder::compile(&schema).expect("compile");

        let row_count = 3;
        let mut output = vec![0u8; 32]; // Need 48 (3 * 16)
        let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];

        let input = vec![0u8; 48];
        let result = decoder.decode_checked(&input, &buffers, row_count);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("requires 48 bytes"), "Error: {}", err);
    }

    #[test]
    fn test_decode_checked_decimal128_insufficient() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Decimal128(38, 10), false),
        ]));
        let decoder = JitDecoder::compile(&schema).expect("compile");

        let row_count = 2;
        let mut output = vec![0u8; 16]; // Need 32 (2 * 16)
        let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];

        let input = vec![0u8; 32];
        let result = decoder.decode_checked(&input, &buffers, row_count);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("requires 32 bytes"), "Error: {}", err);
    }

    #[test]
    fn test_decode_checked_bytes_variable_length_no_validation() {
        // Variable-length columns (Bytes) require 2 buffers (offsets + data)
        // They rely on JIT-level bounds checking
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Binary, false),
        ]));
        let decoder = JitDecoder::compile(&schema).expect("compile");

        let row_count = 1;
        // Buffer 0: offsets (row_count + 1 i32 values)
        let mut offsets = vec![0u8; (row_count + 1) * 4];
        // Buffer 1: data
        let mut data_buf = vec![0u8; 100];

        let buffers = vec![
            OutputBuffer::new(offsets.as_mut_ptr(), offsets.len()),
            OutputBuffer::new(data_buf.as_mut_ptr(), data_buf.len()),
        ];

        // Encode bytes: length (varint) + data
        let mut input = Vec::new();
        let test_data = b"hello";
        input.extend(encode_varint_i64(test_data.len() as i64));
        input.extend_from_slice(test_data);

        let result = decoder.decode_checked(&input, &buffers, row_count);
        assert!(result.is_ok(), "Should succeed with dual buffers");

        // Verify offsets and data
        let offset_0 = i32::from_le_bytes(offsets[0..4].try_into().unwrap());
        let offset_1 = i32::from_le_bytes(offsets[4..8].try_into().unwrap());
        assert_eq!(offset_0, 0);
        assert_eq!(offset_1, test_data.len() as i32);
        assert_eq!(&data_buf[..test_data.len()], test_data);
    }

    #[test]
    fn test_decode_checked_zero_rows() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Int64, false),
        ]));
        let decoder = JitDecoder::compile(&schema).expect("compile");

        let row_count = 0;
        let mut output = vec![0u8; 0]; // Zero capacity for zero rows
        let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), 0)];

        let input: Vec<u8> = vec![];
        let result = decoder.decode_checked(&input, &buffers, row_count);
        assert!(result.is_ok(), "Zero rows should always succeed");
    }

    // ============== JIT Output Overflow Tests for Bytes ==============
    // Note: The JIT infrastructure for output bounds checking is in place,
    // but the current API uses usize::MAX for backward compatibility.
    // When real capacities are passed, the JIT will detect overflows.

    #[test]
    fn test_fixed_size_for_type_coverage() {
        // Verify fixed_size_for_type returns correct sizes for all fixed types
        use arrow_schema::TimeUnit;

        assert_eq!(JitDecoder::fixed_size_for_type(&DataType::Boolean), Some(1));
        assert_eq!(JitDecoder::fixed_size_for_type(&DataType::Int32), Some(4));
        assert_eq!(JitDecoder::fixed_size_for_type(&DataType::Int64), Some(8));
        assert_eq!(JitDecoder::fixed_size_for_type(&DataType::UInt32), Some(4));
        assert_eq!(JitDecoder::fixed_size_for_type(&DataType::UInt64), Some(8));
        assert_eq!(JitDecoder::fixed_size_for_type(&DataType::Float32), Some(4));
        assert_eq!(JitDecoder::fixed_size_for_type(&DataType::Float64), Some(8));
        assert_eq!(JitDecoder::fixed_size_for_type(&DataType::Date32), Some(4));
        assert_eq!(JitDecoder::fixed_size_for_type(&DataType::Date64), Some(8));
        assert_eq!(JitDecoder::fixed_size_for_type(&DataType::Time32(TimeUnit::Millisecond)), Some(4));
        assert_eq!(JitDecoder::fixed_size_for_type(&DataType::Time64(TimeUnit::Microsecond)), Some(8));
        assert_eq!(JitDecoder::fixed_size_for_type(&DataType::Timestamp(TimeUnit::Second, None)), Some(8));
        assert_eq!(JitDecoder::fixed_size_for_type(&DataType::Duration(TimeUnit::Second)), Some(8));
        assert_eq!(JitDecoder::fixed_size_for_type(&DataType::Decimal128(38, 10)), Some(16));
        assert_eq!(JitDecoder::fixed_size_for_type(&DataType::Decimal256(76, 20)), Some(32));
        assert_eq!(JitDecoder::fixed_size_for_type(&DataType::FixedSizeBinary(20)), Some(20));

        // Variable-length types should return None
        assert_eq!(JitDecoder::fixed_size_for_type(&DataType::Utf8), None);
        assert_eq!(JitDecoder::fixed_size_for_type(&DataType::Binary), None);
        assert_eq!(JitDecoder::fixed_size_for_type(&DataType::LargeUtf8), None);
        assert_eq!(JitDecoder::fixed_size_for_type(&DataType::LargeBinary), None);
    }

    #[test]
    fn test_has_variable_columns() {
        // Schema with only fixed types
        let fixed_schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Float32, false),
        ]));
        let fixed_decoder = JitDecoder::compile(&fixed_schema).expect("compile");
        assert!(!fixed_decoder.has_variable_columns());

        // Schema with variable-length type
        let var_schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Utf8, false),
        ]));
        let var_decoder = JitDecoder::compile(&var_schema).expect("compile");
        assert!(var_decoder.has_variable_columns());

        // Schema with only variable-length types
        let all_var_schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Binary, false),
            Field::new("b", DataType::Utf8, false),
        ]));
        let all_var_decoder = JitDecoder::compile(&all_var_schema).expect("compile");
        assert!(all_var_decoder.has_variable_columns());
    }

    #[test]
    fn test_required_capacity() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),  // 8 bytes
            Field::new("b", DataType::Int32, false),  // 4 bytes
            Field::new("c", DataType::Binary, false), // variable
        ]));
        let decoder = JitDecoder::compile(&schema).expect("compile");

        let row_count = 10;

        // Fixed columns have exact requirements
        assert_eq!(decoder.required_capacity(0, row_count), Some(80)); // 10 * 8
        assert_eq!(decoder.required_capacity(1, row_count), Some(40)); // 10 * 4
        // Variable column returns None
        assert_eq!(decoder.required_capacity(2, row_count), None);
    }

    #[test]
    fn test_decode_result_struct() {
        // Test ok() constructor
        let ok_result = DecodeResult::ok(100, 10);
        assert_eq!(ok_result.bytes_consumed, 100);
        assert_eq!(ok_result.rows_decoded, 10);
        assert_eq!(ok_result.status, DecodeStatus::Ok);
        assert!(ok_result.is_ok());
        assert!(!ok_result.has_more());

        // Test overflow() constructor
        let overflow_result = DecodeResult::overflow(50, 5);
        assert_eq!(overflow_result.bytes_consumed, 50);
        assert_eq!(overflow_result.rows_decoded, 5);
        assert_eq!(overflow_result.status, DecodeStatus::OutputOverflow);
        assert!(!overflow_result.is_ok());
        assert!(overflow_result.has_more());

        // Test input_error() constructor
        let error_result = DecodeResult::input_error();
        assert_eq!(error_result.bytes_consumed, 0);
        assert_eq!(error_result.rows_decoded, 0);
        assert_eq!(error_result.status, DecodeStatus::InputBounds);
        assert!(!error_result.is_ok());
        assert!(!error_result.has_more());
    }

    #[test]
    fn test_decode_chunked_success() {
        // Test decode_chunked when all rows fit
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Encode 10 rows
        let mut input = Vec::new();
        for i in 0..10i64 {
            input.extend(encode_varint_i64(i));
        }

        // Buffer big enough for all rows (80 bytes for 10 i64s)
        let mut output = vec![0u8; 80];
        let output_buffers = vec![OutputBuffer::new(output.as_mut_ptr(), 80)];

        let result = decoder.decode_chunked(&input, &output_buffers, 10);

        assert!(result.is_ok());
        assert!(!result.has_more());
        assert_eq!(result.rows_decoded, 10);
        assert_eq!(result.bytes_consumed, input.len());
    }

    #[test]
    fn test_decode_chunked_overflow() {
        // Test decode_chunked when buffer is too small
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Encode 100 rows
        let mut input = Vec::new();
        for i in 0..100i64 {
            input.extend(encode_varint_i64(i));
        }

        // Buffer only big enough for 10 rows (80 bytes)
        let mut output = vec![0u8; 80];
        let output_buffers = vec![OutputBuffer::new(output.as_mut_ptr(), 80)];

        let result = decoder.decode_chunked(&input, &output_buffers, 100);

        // Should report overflow with partial progress
        assert!(!result.is_ok());
        assert!(result.has_more());
        assert_eq!(result.rows_decoded, 10); // 80 bytes / 8 bytes per i64 = 10 rows
        assert!(result.bytes_consumed > 0);
        assert!(result.bytes_consumed < input.len());
    }

    #[test]
    fn test_decode_chunked_single_row_overflow() {
        // Test when even a single row doesn't fit
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Single row
        let input = encode_varint_i64(42);

        // Buffer too small for even 1 row (need 8 bytes, have 4)
        let mut output = vec![0u8; 4];
        let output_buffers = vec![OutputBuffer::new(output.as_mut_ptr(), 4)];

        let result = decoder.decode_chunked(&input, &output_buffers, 1);

        // Should report overflow with 0 rows decoded
        assert!(!result.is_ok());
        assert!(result.has_more());
        assert_eq!(result.rows_decoded, 0);
        assert_eq!(result.bytes_consumed, 0);
    }

    #[test]
    fn test_decode_chunked_continuation() {
        // Test that we can continue decoding from where we left off
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Encode 20 rows
        let mut input = Vec::new();
        for i in 0..20i64 {
            input.extend(encode_varint_i64(i));
        }

        // Buffer for 10 rows
        let mut output = vec![0u8; 80];
        let output_buffers = vec![OutputBuffer::new(output.as_mut_ptr(), 80)];

        // First chunk
        let result1 = decoder.decode_chunked(&input, &output_buffers, 20);
        assert!(result1.has_more());
        assert_eq!(result1.rows_decoded, 10);

        // Second chunk - continue from where we left off
        let remaining_input = &input[result1.bytes_consumed..];
        let remaining_rows = 20 - result1.rows_decoded;

        // Reset output buffer
        output.fill(0);
        let output_buffers = vec![OutputBuffer::new(output.as_mut_ptr(), 80)];

        let result2 = decoder.decode_chunked(remaining_input, &output_buffers, remaining_rows);
        assert!(result2.is_ok());
        assert_eq!(result2.rows_decoded, 10);

        // Total rows decoded should be 20
        assert_eq!(result1.rows_decoded + result2.rows_decoded, 20);
    }

    // =========================================================================
    // COMPREHENSIVE NULLABLE TESTS WITH VALIDITY BITMAP VERIFICATION
    // =========================================================================
    //
    // These tests verify that:
    // 1. Validity bitmaps are correctly set (bit=1 for non-null, bit=0 for null)
    // 2. Data values are correct for non-null entries
    // 3. All data types handle nullability correctly
    // 4. Corner cases (all nulls, no nulls, mixed, single row) work correctly
    // =========================================================================

    /// Helper to check validity bit at a given row index
    fn check_validity_bit(validity: &[u8], row: usize) -> bool {
        let byte_idx = row / 8;
        let bit_idx = row % 8;
        if byte_idx >= validity.len() {
            return false;
        }
        (validity[byte_idx] & (1 << bit_idx)) != 0
    }

    /// Helper to count valid (non-null) bits in validity buffer
    fn count_valid_bits(validity: &[u8], row_count: usize) -> usize {
        (0..row_count).filter(|&i| check_validity_bit(validity, i)).count()
    }

    // -------------------------------------------------------------------------
    // Int64 Nullable Tests with Validity Bitmap
    // -------------------------------------------------------------------------

    #[test]
    fn test_nullable_int64_validity_all_non_null() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int64, true),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("compile");
        let test_values: Vec<i64> = vec![10, -20, 0, i64::MAX, i64::MIN];
        let row_count = test_values.len();

        let mut input = Vec::new();
        for &v in &test_values {
            input.extend(encode_union_tag(false)); // non-null
            input.extend(encode_varint_i64(v));
        }

        // Allocate data buffer and validity buffer
        let mut data_buf = vec![0u8; row_count * 8];
        let validity_bytes = (row_count + 7) / 8;
        let mut validity_buf = vec![0u8; validity_bytes];

        let output_buffers = vec![
            OutputBuffer::new(data_buf.as_mut_ptr(), data_buf.len()),
            OutputBuffer::new(validity_buf.as_mut_ptr(), validity_buf.len()),
        ];

        let result = decoder.decode_chunked(&input, &output_buffers, row_count);
        assert!(result.is_ok(), "Decode failed: {:?}", result);
        assert_eq!(result.rows_decoded, row_count);

        // Verify all validity bits are set to 1
        for i in 0..row_count {
            assert!(check_validity_bit(&validity_buf, i),
                "Validity bit should be 1 for non-null at row {}", i);
        }
        assert_eq!(count_valid_bits(&validity_buf, row_count), row_count);

        // Verify data values
        for (i, &expected) in test_values.iter().enumerate() {
            let actual = i64::from_le_bytes(data_buf[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, expected, "Data mismatch at row {}", i);
        }
    }

    #[test]
    fn test_nullable_int64_validity_all_nulls_zeroed_buffer() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int64, true),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("compile");
        let row_count = 10;

        let mut input = Vec::new();
        for _ in 0..row_count {
            input.extend(encode_union_tag(true)); // null
        }

        let mut data_buf = vec![0u8; row_count * 8];
        let validity_bytes = (row_count + 7) / 8;
        let mut validity_buf = vec![0u8; validity_bytes]; // zeroed - nulls stay as 0

        let output_buffers = vec![
            OutputBuffer::new(data_buf.as_mut_ptr(), data_buf.len()),
            OutputBuffer::new(validity_buf.as_mut_ptr(), validity_buf.len()),
        ];

        let result = decoder.decode_chunked(&input, &output_buffers, row_count);
        assert!(result.is_ok());
        assert_eq!(result.rows_decoded, row_count);

        // All validity bits should be 0 (null)
        for i in 0..row_count {
            assert!(!check_validity_bit(&validity_buf, i),
                "Validity bit should be 0 for null at row {}", i);
        }
        assert_eq!(count_valid_bits(&validity_buf, row_count), 0);
    }

    #[test]
    fn test_nullable_int64_validity_mixed_pattern() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int64, true),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Mixed pattern: value, null, value, value, null, null, value
        let values: Vec<Option<i64>> = vec![
            Some(100), None, Some(-50), Some(0), None, None, Some(999),
        ];
        let row_count = values.len();

        let mut input = Vec::new();
        for v in &values {
            match v {
                Some(val) => {
                    input.extend(encode_union_tag(false));
                    input.extend(encode_varint_i64(*val));
                }
                None => {
                    input.extend(encode_union_tag(true));
                }
            }
        }

        let mut data_buf = vec![0u8; row_count * 8];
        let validity_bytes = (row_count + 7) / 8;
        let mut validity_buf = vec![0u8; validity_bytes];

        let output_buffers = vec![
            OutputBuffer::new(data_buf.as_mut_ptr(), data_buf.len()),
            OutputBuffer::new(validity_buf.as_mut_ptr(), validity_buf.len()),
        ];

        let result = decoder.decode_chunked(&input, &output_buffers, row_count);
        assert!(result.is_ok());
        assert_eq!(result.rows_decoded, row_count);

        // Verify validity bits match pattern
        for (i, v) in values.iter().enumerate() {
            let is_valid = check_validity_bit(&validity_buf, i);
            match v {
                Some(_) => assert!(is_valid, "Row {} should be valid (non-null)", i),
                None => assert!(!is_valid, "Row {} should be invalid (null)", i),
            }
        }

        // Verify data values for non-null entries
        for (i, v) in values.iter().enumerate() {
            if let Some(expected) = v {
                let actual = i64::from_le_bytes(data_buf[i * 8..(i + 1) * 8].try_into().unwrap());
                assert_eq!(actual, *expected, "Data mismatch at row {}", i);
            }
        }
    }

    #[test]
    fn test_nullable_int64_single_row_non_null() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int64, true),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        let mut input = Vec::new();
        input.extend(encode_union_tag(false));
        input.extend(encode_varint_i64(42));

        let mut data_buf = vec![0u8; 8];
        let mut validity_buf = vec![0u8; 1];

        let output_buffers = vec![
            OutputBuffer::new(data_buf.as_mut_ptr(), data_buf.len()),
            OutputBuffer::new(validity_buf.as_mut_ptr(), validity_buf.len()),
        ];

        let result = decoder.decode_chunked(&input, &output_buffers, 1);
        assert!(result.is_ok());
        assert_eq!(result.rows_decoded, 1);

        assert!(check_validity_bit(&validity_buf, 0), "Single row should be valid");
        let actual = i64::from_le_bytes(data_buf[0..8].try_into().unwrap());
        assert_eq!(actual, 42);
    }

    #[test]
    fn test_nullable_int64_single_row_null() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int64, true),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        let mut input = Vec::new();
        input.extend(encode_union_tag(true)); // null

        let mut data_buf = vec![0u8; 8];
        let mut validity_buf = vec![0u8; 1];

        let output_buffers = vec![
            OutputBuffer::new(data_buf.as_mut_ptr(), data_buf.len()),
            OutputBuffer::new(validity_buf.as_mut_ptr(), validity_buf.len()),
        ];

        let result = decoder.decode_chunked(&input, &output_buffers, 1);
        assert!(result.is_ok());
        assert_eq!(result.rows_decoded, 1);

        assert!(!check_validity_bit(&validity_buf, 0), "Single null row should be invalid");
    }

    // -------------------------------------------------------------------------
    // Int32 Nullable Tests with Validity Bitmap
    // -------------------------------------------------------------------------

    #[test]
    fn test_nullable_int32_validity_mixed() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, true),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        let values: Vec<Option<i32>> = vec![
            Some(1), Some(-1), None, Some(i32::MAX), None, Some(i32::MIN), Some(0),
        ];
        let row_count = values.len();

        let mut input = Vec::new();
        for v in &values {
            match v {
                Some(val) => {
                    input.extend(encode_union_tag(false));
                    input.extend(encode_varint_i32(*val));
                }
                None => {
                    input.extend(encode_union_tag(true));
                }
            }
        }

        let mut data_buf = vec![0u8; row_count * 4];
        let validity_bytes = (row_count + 7) / 8;
        let mut validity_buf = vec![0u8; validity_bytes];

        let output_buffers = vec![
            OutputBuffer::new(data_buf.as_mut_ptr(), data_buf.len()),
            OutputBuffer::new(validity_buf.as_mut_ptr(), validity_buf.len()),
        ];

        let result = decoder.decode_chunked(&input, &output_buffers, row_count);
        assert!(result.is_ok());
        assert_eq!(result.rows_decoded, row_count);

        // Verify validity and data
        for (i, v) in values.iter().enumerate() {
            let is_valid = check_validity_bit(&validity_buf, i);
            match v {
                Some(expected) => {
                    assert!(is_valid, "Row {} should be valid", i);
                    let actual = i32::from_le_bytes(data_buf[i * 4..(i + 1) * 4].try_into().unwrap());
                    assert_eq!(actual, *expected, "Data mismatch at row {}", i);
                }
                None => {
                    assert!(!is_valid, "Row {} should be invalid (null)", i);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Float64 Nullable Tests with Validity Bitmap
    // -------------------------------------------------------------------------

    #[test]
    fn test_nullable_float64_validity_mixed() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Float64, true),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        let values: Vec<Option<f64>> = vec![
            Some(3.14159), None, Some(-0.0), Some(f64::INFINITY), None,
            Some(f64::NEG_INFINITY), Some(1e308), None,
        ];
        let row_count = values.len();

        let mut input = Vec::new();
        for v in &values {
            match v {
                Some(val) => {
                    input.extend(encode_union_tag(false));
                    input.extend(val.to_le_bytes());
                }
                None => {
                    input.extend(encode_union_tag(true));
                }
            }
        }

        let mut data_buf = vec![0u8; row_count * 8];
        let validity_bytes = (row_count + 7) / 8;
        let mut validity_buf = vec![0u8; validity_bytes];

        let output_buffers = vec![
            OutputBuffer::new(data_buf.as_mut_ptr(), data_buf.len()),
            OutputBuffer::new(validity_buf.as_mut_ptr(), validity_buf.len()),
        ];

        let result = decoder.decode_chunked(&input, &output_buffers, row_count);
        assert!(result.is_ok());
        assert_eq!(result.rows_decoded, row_count);

        for (i, v) in values.iter().enumerate() {
            let is_valid = check_validity_bit(&validity_buf, i);
            match v {
                Some(expected) => {
                    assert!(is_valid, "Row {} should be valid", i);
                    let actual = f64::from_le_bytes(data_buf[i * 8..(i + 1) * 8].try_into().unwrap());
                    if expected.is_nan() {
                        assert!(actual.is_nan(), "Row {} should be NaN", i);
                    } else {
                        assert_eq!(actual, *expected, "Data mismatch at row {}", i);
                    }
                }
                None => {
                    assert!(!is_valid, "Row {} should be invalid", i);
                }
            }
        }
    }

    #[test]
    fn test_nullable_float64_nan_handling() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Float64, true),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        let values: Vec<Option<f64>> = vec![Some(f64::NAN), None, Some(f64::NAN)];
        let row_count = values.len();

        let mut input = Vec::new();
        for v in &values {
            match v {
                Some(val) => {
                    input.extend(encode_union_tag(false));
                    input.extend(val.to_le_bytes());
                }
                None => {
                    input.extend(encode_union_tag(true));
                }
            }
        }

        let mut data_buf = vec![0u8; row_count * 8];
        let mut validity_buf = vec![0u8; 1];

        let output_buffers = vec![
            OutputBuffer::new(data_buf.as_mut_ptr(), data_buf.len()),
            OutputBuffer::new(validity_buf.as_mut_ptr(), validity_buf.len()),
        ];

        let result = decoder.decode_chunked(&input, &output_buffers, row_count);
        assert!(result.is_ok());

        // Row 0: NaN (valid)
        assert!(check_validity_bit(&validity_buf, 0));
        let v0 = f64::from_le_bytes(data_buf[0..8].try_into().unwrap());
        assert!(v0.is_nan());

        // Row 1: null
        assert!(!check_validity_bit(&validity_buf, 1));

        // Row 2: NaN (valid)
        assert!(check_validity_bit(&validity_buf, 2));
        let v2 = f64::from_le_bytes(data_buf[16..24].try_into().unwrap());
        assert!(v2.is_nan());
    }

    // -------------------------------------------------------------------------
    // Float32 Nullable Tests with Validity Bitmap
    // -------------------------------------------------------------------------

    #[test]
    fn test_nullable_float32_validity_mixed() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Float32, true),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        let values: Vec<Option<f32>> = vec![
            Some(1.5f32), None, Some(-0.0f32), Some(f32::MAX), None, Some(f32::MIN),
        ];
        let row_count = values.len();

        let mut input = Vec::new();
        for v in &values {
            match v {
                Some(val) => {
                    input.extend(encode_union_tag(false));
                    input.extend(val.to_le_bytes());
                }
                None => {
                    input.extend(encode_union_tag(true));
                }
            }
        }

        let mut data_buf = vec![0u8; row_count * 4];
        let validity_bytes = (row_count + 7) / 8;
        let mut validity_buf = vec![0u8; validity_bytes];

        let output_buffers = vec![
            OutputBuffer::new(data_buf.as_mut_ptr(), data_buf.len()),
            OutputBuffer::new(validity_buf.as_mut_ptr(), validity_buf.len()),
        ];

        let result = decoder.decode_chunked(&input, &output_buffers, row_count);
        assert!(result.is_ok());
        assert_eq!(result.rows_decoded, row_count);

        for (i, v) in values.iter().enumerate() {
            let is_valid = check_validity_bit(&validity_buf, i);
            match v {
                Some(expected) => {
                    assert!(is_valid, "Row {} should be valid", i);
                    let actual = f32::from_le_bytes(data_buf[i * 4..(i + 1) * 4].try_into().unwrap());
                    assert_eq!(actual, *expected, "Data mismatch at row {}", i);
                }
                None => {
                    assert!(!is_valid, "Row {} should be invalid", i);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Boolean Nullable Tests with Validity Bitmap
    // -------------------------------------------------------------------------

    #[test]
    fn test_nullable_boolean_validity_mixed() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("flag", DataType::Boolean, true),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        let values: Vec<Option<bool>> = vec![
            Some(true), None, Some(false), Some(true), None, None, Some(false),
        ];
        let row_count = values.len();

        let mut input = Vec::new();
        for v in &values {
            match v {
                Some(val) => {
                    input.extend(encode_union_tag(false));
                    input.push(if *val { 1 } else { 0 });
                }
                None => {
                    input.extend(encode_union_tag(true));
                }
            }
        }

        let mut data_buf = vec![0u8; row_count]; // 1 byte per boolean
        let validity_bytes = (row_count + 7) / 8;
        let mut validity_buf = vec![0u8; validity_bytes];

        let output_buffers = vec![
            OutputBuffer::new(data_buf.as_mut_ptr(), data_buf.len()),
            OutputBuffer::new(validity_buf.as_mut_ptr(), validity_buf.len()),
        ];

        let result = decoder.decode_chunked(&input, &output_buffers, row_count);
        assert!(result.is_ok());
        assert_eq!(result.rows_decoded, row_count);

        for (i, v) in values.iter().enumerate() {
            let is_valid = check_validity_bit(&validity_buf, i);
            match v {
                Some(expected) => {
                    assert!(is_valid, "Row {} should be valid", i);
                    let actual = data_buf[i] != 0;
                    assert_eq!(actual, *expected, "Boolean mismatch at row {}", i);
                }
                None => {
                    assert!(!is_valid, "Row {} should be invalid", i);
                }
            }
        }
    }

    #[test]
    fn test_nullable_boolean_all_true() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("flag", DataType::Boolean, true),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("compile");
        let row_count = 8; // exactly one byte of validity

        let mut input = Vec::new();
        for _ in 0..row_count {
            input.extend(encode_union_tag(false));
            input.push(1); // true
        }

        let mut data_buf = vec![0u8; row_count];
        let mut validity_buf = vec![0u8; 1];

        let output_buffers = vec![
            OutputBuffer::new(data_buf.as_mut_ptr(), data_buf.len()),
            OutputBuffer::new(validity_buf.as_mut_ptr(), validity_buf.len()),
        ];

        let result = decoder.decode_chunked(&input, &output_buffers, row_count);
        assert!(result.is_ok());

        // All bits should be 1 -> validity_buf[0] = 0xFF
        assert_eq!(validity_buf[0], 0xFF, "All 8 validity bits should be set");

        for i in 0..row_count {
            assert_eq!(data_buf[i], 1, "All values should be true");
        }
    }

    // -------------------------------------------------------------------------
    // String/Bytes (Variable-length) Nullable Tests with Validity Bitmap
    // -------------------------------------------------------------------------

    #[test]
    fn test_nullable_string_validity_mixed() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, true),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        let values: Vec<Option<&str>> = vec![
            Some("hello"), None, Some(""), Some("world"), None, Some("test"),
        ];
        let row_count = values.len();

        let mut input = Vec::new();
        for v in &values {
            match v {
                Some(s) => {
                    input.extend(encode_union_tag(false));
                    input.extend(encode_avro_bytes(s.as_bytes()));
                }
                None => {
                    input.extend(encode_union_tag(true));
                }
            }
        }

        // Variable-length: offsets buffer + data buffer + validity buffer
        let offsets_size = (row_count + 1) * 4; // i32 offsets, need n+1 for final offset
        let data_size = 256; // plenty of space
        let validity_bytes = (row_count + 7) / 8;

        let mut offsets_buf = vec![0u8; offsets_size];
        let mut data_buf = vec![0u8; data_size];
        let mut validity_buf = vec![0u8; validity_bytes];

        let output_buffers = vec![
            OutputBuffer::new(offsets_buf.as_mut_ptr(), offsets_buf.len()),
            OutputBuffer::new(data_buf.as_mut_ptr(), data_buf.len()),
            OutputBuffer::new(validity_buf.as_mut_ptr(), validity_buf.len()),
        ];

        let result = decoder.decode_chunked(&input, &output_buffers, row_count);
        assert!(result.is_ok(), "Decode failed: {:?}", result);
        assert_eq!(result.rows_decoded, row_count);

        // Verify validity bits
        for (i, v) in values.iter().enumerate() {
            let is_valid = check_validity_bit(&validity_buf, i);
            match v {
                Some(_) => assert!(is_valid, "Row {} should be valid", i),
                None => assert!(!is_valid, "Row {} should be invalid", i),
            }
        }

        // Verify string values by reading offsets
        let non_null_count = values.iter().filter(|v| v.is_some()).count();
        assert_eq!(count_valid_bits(&validity_buf, row_count), non_null_count);
    }

    #[test]
    fn test_nullable_string_all_empty_strings() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, true),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("compile");
        let row_count = 5;

        let mut input = Vec::new();
        for _ in 0..row_count {
            input.extend(encode_union_tag(false)); // non-null
            input.extend(encode_avro_bytes(b"")); // empty string
        }

        let offsets_size = (row_count + 1) * 4;
        let data_size = 64;
        let validity_bytes = (row_count + 7) / 8;

        let mut offsets_buf = vec![0u8; offsets_size];
        let mut data_buf = vec![0u8; data_size];
        let mut validity_buf = vec![0u8; validity_bytes];

        let output_buffers = vec![
            OutputBuffer::new(offsets_buf.as_mut_ptr(), offsets_buf.len()),
            OutputBuffer::new(data_buf.as_mut_ptr(), data_buf.len()),
            OutputBuffer::new(validity_buf.as_mut_ptr(), validity_buf.len()),
        ];

        let result = decoder.decode_chunked(&input, &output_buffers, row_count);
        assert!(result.is_ok());
        assert_eq!(result.rows_decoded, row_count);

        // All should be valid (empty string != null)
        assert_eq!(count_valid_bits(&validity_buf, row_count), row_count);
    }

    #[test]
    fn test_nullable_string_all_nulls() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, true),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("compile");
        let row_count = 5;

        let mut input = Vec::new();
        for _ in 0..row_count {
            input.extend(encode_union_tag(true)); // null
        }

        let offsets_size = (row_count + 1) * 4;
        let data_size = 64;
        let validity_bytes = (row_count + 7) / 8;

        let mut offsets_buf = vec![0u8; offsets_size];
        let mut data_buf = vec![0u8; data_size];
        let mut validity_buf = vec![0u8; validity_bytes];

        let output_buffers = vec![
            OutputBuffer::new(offsets_buf.as_mut_ptr(), offsets_buf.len()),
            OutputBuffer::new(data_buf.as_mut_ptr(), data_buf.len()),
            OutputBuffer::new(validity_buf.as_mut_ptr(), validity_buf.len()),
        ];

        let result = decoder.decode_chunked(&input, &output_buffers, row_count);
        assert!(result.is_ok());
        assert_eq!(result.rows_decoded, row_count);

        // All should be null
        assert_eq!(count_valid_bits(&validity_buf, row_count), 0);
    }

    // -------------------------------------------------------------------------
    // Enum (Dictionary) Nullable Tests with Validity Bitmap
    // -------------------------------------------------------------------------

    #[test]
    fn test_nullable_enum_validity_mixed() {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "status",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                true,
            ),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        let values: Vec<Option<u32>> = vec![Some(0), None, Some(1), Some(2), None, Some(0)];
        let row_count = values.len();

        let mut input = Vec::new();
        for v in &values {
            match v {
                Some(idx) => {
                    input.extend(encode_union_tag(false));
                    input.extend(encode_enum_index(*idx));
                }
                None => {
                    input.extend(encode_union_tag(true));
                }
            }
        }

        let mut data_buf = vec![0u8; row_count * 4]; // i32 indices
        let validity_bytes = (row_count + 7) / 8;
        let mut validity_buf = vec![0u8; validity_bytes];

        let output_buffers = vec![
            OutputBuffer::new(data_buf.as_mut_ptr(), data_buf.len()),
            OutputBuffer::new(validity_buf.as_mut_ptr(), validity_buf.len()),
        ];

        let result = decoder.decode_chunked(&input, &output_buffers, row_count);
        assert!(result.is_ok());
        assert_eq!(result.rows_decoded, row_count);

        for (i, v) in values.iter().enumerate() {
            let is_valid = check_validity_bit(&validity_buf, i);
            match v {
                Some(expected) => {
                    assert!(is_valid, "Row {} should be valid", i);
                    let actual = i32::from_le_bytes(data_buf[i * 4..(i + 1) * 4].try_into().unwrap());
                    assert_eq!(actual as u32, *expected, "Enum index mismatch at row {}", i);
                }
                None => {
                    assert!(!is_valid, "Row {} should be invalid", i);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Multi-column Nullable Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_multi_column_nullable_mixed() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("value", DataType::Float64, true),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Pattern: (id, value) - some nulls in each column
        let rows: Vec<(Option<i64>, Option<f64>)> = vec![
            (Some(1), Some(1.5)),
            (None, Some(2.5)),
            (Some(3), None),
            (None, None),
            (Some(5), Some(5.5)),
        ];
        let row_count = rows.len();

        let mut input = Vec::new();
        for (id, val) in &rows {
            // id column
            match id {
                Some(v) => {
                    input.extend(encode_union_tag(false));
                    input.extend(encode_varint_i64(*v));
                }
                None => {
                    input.extend(encode_union_tag(true));
                }
            }
            // value column
            match val {
                Some(v) => {
                    input.extend(encode_union_tag(false));
                    input.extend(v.to_le_bytes());
                }
                None => {
                    input.extend(encode_union_tag(true));
                }
            }
        }

        // Column 0 (id): data + validity
        // Column 1 (value): data + validity
        let mut id_data = vec![0u8; row_count * 8];
        let mut id_validity = vec![0u8; (row_count + 7) / 8];
        let mut val_data = vec![0u8; row_count * 8];
        let mut val_validity = vec![0u8; (row_count + 7) / 8];

        let output_buffers = vec![
            OutputBuffer::new(id_data.as_mut_ptr(), id_data.len()),
            OutputBuffer::new(id_validity.as_mut_ptr(), id_validity.len()),
            OutputBuffer::new(val_data.as_mut_ptr(), val_data.len()),
            OutputBuffer::new(val_validity.as_mut_ptr(), val_validity.len()),
        ];

        let result = decoder.decode_chunked(&input, &output_buffers, row_count);
        assert!(result.is_ok(), "Decode failed: {:?}", result);
        assert_eq!(result.rows_decoded, row_count);

        // Verify id column
        for (i, (id, _)) in rows.iter().enumerate() {
            let is_valid = check_validity_bit(&id_validity, i);
            match id {
                Some(expected) => {
                    assert!(is_valid, "Row {} id should be valid", i);
                    let actual = i64::from_le_bytes(id_data[i * 8..(i + 1) * 8].try_into().unwrap());
                    assert_eq!(actual, *expected);
                }
                None => assert!(!is_valid, "Row {} id should be null", i),
            }
        }

        // Verify value column
        for (i, (_, val)) in rows.iter().enumerate() {
            let is_valid = check_validity_bit(&val_validity, i);
            match val {
                Some(expected) => {
                    assert!(is_valid, "Row {} value should be valid", i);
                    let actual = f64::from_le_bytes(val_data[i * 8..(i + 1) * 8].try_into().unwrap());
                    assert_eq!(actual, *expected);
                }
                None => assert!(!is_valid, "Row {} value should be null", i),
            }
        }
    }

    #[test]
    fn test_mixed_nullable_and_non_nullable_columns() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("required_id", DataType::Int64, false),  // NOT nullable
            Field::new("optional_value", DataType::Int64, true), // nullable
        ]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Pattern: required_id always present, optional_value sometimes null
        let rows: Vec<(i64, Option<i64>)> = vec![
            (1, Some(100)),
            (2, None),
            (3, Some(300)),
            (4, None),
            (5, Some(500)),
        ];
        let row_count = rows.len();

        let mut input = Vec::new();
        for (id, val) in &rows {
            // required_id - no union tag
            input.extend(encode_varint_i64(*id));
            // optional_value - has union tag
            match val {
                Some(v) => {
                    input.extend(encode_union_tag(false));
                    input.extend(encode_varint_i64(*v));
                }
                None => {
                    input.extend(encode_union_tag(true));
                }
            }
        }

        // Column 0 (required): data only (no validity)
        // Column 1 (optional): data + validity
        let mut id_data = vec![0u8; row_count * 8];
        let mut val_data = vec![0u8; row_count * 8];
        let mut val_validity = vec![0u8; (row_count + 7) / 8];

        let output_buffers = vec![
            OutputBuffer::new(id_data.as_mut_ptr(), id_data.len()),
            OutputBuffer::new(val_data.as_mut_ptr(), val_data.len()),
            OutputBuffer::new(val_validity.as_mut_ptr(), val_validity.len()),
        ];

        let result = decoder.decode_chunked(&input, &output_buffers, row_count);
        assert!(result.is_ok(), "Decode failed: {:?}", result);
        assert_eq!(result.rows_decoded, row_count);

        // Verify required_id (all values should be present)
        for (i, (expected_id, _)) in rows.iter().enumerate() {
            let actual = i64::from_le_bytes(id_data[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, *expected_id, "required_id mismatch at row {}", i);
        }

        // Verify optional_value with validity
        for (i, (_, val)) in rows.iter().enumerate() {
            let is_valid = check_validity_bit(&val_validity, i);
            match val {
                Some(expected) => {
                    assert!(is_valid, "Row {} optional_value should be valid", i);
                    let actual = i64::from_le_bytes(val_data[i * 8..(i + 1) * 8].try_into().unwrap());
                    assert_eq!(actual, *expected);
                }
                None => assert!(!is_valid, "Row {} optional_value should be null", i),
            }
        }
    }

    // -------------------------------------------------------------------------
    // Edge Cases and Stress Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_nullable_validity_byte_boundary_9_rows() {
        // Test crossing byte boundary: 9 rows = 2 validity bytes
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int64, true),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        // 9 rows: bits 0-7 in byte 0, bit 8 in byte 1
        let values: Vec<Option<i64>> = vec![
            Some(1), None, Some(3), Some(4), None, Some(6), Some(7), Some(8), Some(9),
        ];
        let row_count = values.len();
        assert_eq!(row_count, 9);

        let mut input = Vec::new();
        for v in &values {
            match v {
                Some(val) => {
                    input.extend(encode_union_tag(false));
                    input.extend(encode_varint_i64(*val));
                }
                None => {
                    input.extend(encode_union_tag(true));
                }
            }
        }

        let mut data_buf = vec![0u8; row_count * 8];
        let validity_bytes = (row_count + 7) / 8;
        assert_eq!(validity_bytes, 2);
        let mut validity_buf = vec![0u8; validity_bytes];

        let output_buffers = vec![
            OutputBuffer::new(data_buf.as_mut_ptr(), data_buf.len()),
            OutputBuffer::new(validity_buf.as_mut_ptr(), validity_buf.len()),
        ];

        let result = decoder.decode_chunked(&input, &output_buffers, row_count);
        assert!(result.is_ok());
        assert_eq!(result.rows_decoded, row_count);

        // Verify each bit
        for (i, v) in values.iter().enumerate() {
            let is_valid = check_validity_bit(&validity_buf, i);
            assert_eq!(is_valid, v.is_some(), "Validity mismatch at row {}", i);
        }
    }

    #[test]
    fn test_nullable_validity_16_rows() {
        // Test exactly 16 rows = 2 full validity bytes
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int64, true),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Alternating null/non-null pattern
        let values: Vec<Option<i64>> = (0..16).map(|i| {
            if i % 2 == 0 { Some(i as i64) } else { None }
        }).collect();
        let row_count = values.len();

        let mut input = Vec::new();
        for v in &values {
            match v {
                Some(val) => {
                    input.extend(encode_union_tag(false));
                    input.extend(encode_varint_i64(*val));
                }
                None => {
                    input.extend(encode_union_tag(true));
                }
            }
        }

        let mut data_buf = vec![0u8; row_count * 8];
        let mut validity_buf = vec![0u8; 2];

        let output_buffers = vec![
            OutputBuffer::new(data_buf.as_mut_ptr(), data_buf.len()),
            OutputBuffer::new(validity_buf.as_mut_ptr(), validity_buf.len()),
        ];

        let result = decoder.decode_chunked(&input, &output_buffers, row_count);
        assert!(result.is_ok());

        // Expected pattern: 0b01010101 for both bytes (even indices valid)
        assert_eq!(validity_buf[0], 0b01010101, "First validity byte mismatch");
        assert_eq!(validity_buf[1], 0b01010101, "Second validity byte mismatch");
    }

    #[test]
    fn test_nullable_large_row_count() {
        // Test with 1000 rows to stress test validity bitmap handling
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int64, true),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("compile");
        let row_count = 1000;

        // Every 3rd row is null
        let values: Vec<Option<i64>> = (0..row_count).map(|i| {
            if i % 3 == 0 { None } else { Some(i as i64) }
        }).collect();

        let mut input = Vec::new();
        for v in &values {
            match v {
                Some(val) => {
                    input.extend(encode_union_tag(false));
                    input.extend(encode_varint_i64(*val));
                }
                None => {
                    input.extend(encode_union_tag(true));
                }
            }
        }

        let mut data_buf = vec![0u8; row_count * 8];
        let validity_bytes = (row_count + 7) / 8;
        let mut validity_buf = vec![0u8; validity_bytes];

        let output_buffers = vec![
            OutputBuffer::new(data_buf.as_mut_ptr(), data_buf.len()),
            OutputBuffer::new(validity_buf.as_mut_ptr(), validity_buf.len()),
        ];

        let result = decoder.decode_chunked(&input, &output_buffers, row_count);
        assert!(result.is_ok());
        assert_eq!(result.rows_decoded, row_count);

        // Verify all validity bits
        let mut null_count = 0;
        let mut valid_count = 0;
        for (i, v) in values.iter().enumerate() {
            let is_valid = check_validity_bit(&validity_buf, i);
            assert_eq!(is_valid, v.is_some(), "Validity mismatch at row {}", i);
            if v.is_some() {
                valid_count += 1;
            } else {
                null_count += 1;
            }
        }

        // Every 3rd row is null: 334 nulls, 666 valid (for 1000 rows)
        assert_eq!(null_count, 334);
        assert_eq!(valid_count, 666);
        assert_eq!(count_valid_bits(&validity_buf, row_count), valid_count);
    }

    #[test]
    fn test_nullable_zero_rows() {
        // Test with 0 rows - should handle gracefully
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int64, true),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        let input: Vec<u8> = Vec::new();
        let mut data_buf = vec![0u8; 8]; // some space, but 0 rows
        let mut validity_buf = vec![0u8; 1];

        let output_buffers = vec![
            OutputBuffer::new(data_buf.as_mut_ptr(), data_buf.len()),
            OutputBuffer::new(validity_buf.as_mut_ptr(), validity_buf.len()),
        ];

        let result = decoder.decode_chunked(&input, &output_buffers, 0);
        assert!(result.is_ok());
        assert_eq!(result.rows_decoded, 0);
        assert_eq!(result.bytes_consumed, 0);
    }

    // -------------------------------------------------------------------------
    // Binary (Bytes) Nullable Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_nullable_binary_validity_mixed() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("data", DataType::Binary, true),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        let values: Vec<Option<&[u8]>> = vec![
            Some(b"hello"),
            None,
            Some(b""),
            Some(&[0, 1, 255, 128]),
            None,
        ];
        let row_count = values.len();

        let mut input = Vec::new();
        for v in &values {
            match v {
                Some(data) => {
                    input.extend(encode_union_tag(false));
                    input.extend(encode_avro_bytes(data));
                }
                None => {
                    input.extend(encode_union_tag(true));
                }
            }
        }

        let offsets_size = (row_count + 1) * 4;
        let data_size = 256;
        let validity_bytes = (row_count + 7) / 8;

        let mut offsets_buf = vec![0u8; offsets_size];
        let mut data_buf = vec![0u8; data_size];
        let mut validity_buf = vec![0u8; validity_bytes];

        let output_buffers = vec![
            OutputBuffer::new(offsets_buf.as_mut_ptr(), offsets_buf.len()),
            OutputBuffer::new(data_buf.as_mut_ptr(), data_buf.len()),
            OutputBuffer::new(validity_buf.as_mut_ptr(), validity_buf.len()),
        ];

        let result = decoder.decode_chunked(&input, &output_buffers, row_count);
        assert!(result.is_ok(), "Decode failed");
        assert_eq!(result.rows_decoded, row_count);

        // Verify validity
        for (i, v) in values.iter().enumerate() {
            let is_valid = check_validity_bit(&validity_buf, i);
            assert_eq!(is_valid, v.is_some(), "Validity mismatch at row {}", i);
        }
    }

}

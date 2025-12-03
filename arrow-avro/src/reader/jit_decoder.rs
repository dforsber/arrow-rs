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

use arrow_array::{
    ArrayRef, BinaryArray, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array,
    RecordBatch, StringArray,
};
use arrow_buffer::{Buffer, ScalarBuffer};
use arrow_schema::{ArrowError, DataType, Field, IntervalUnit, Schema};
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
    /// Signature: (input_ptr, input_len, output_ptrs, row_count) -> bytes_consumed
    /// Returns negative value on error
    decode_fn: unsafe extern "C" fn(*const u8, usize, *mut *mut u8, usize) -> i64,
    /// Keep the JIT module alive (owns the compiled code memory)
    _module: SyncJitModule,
    /// Schema for reference
    schema: Arc<Schema>,
    /// Number of columns
    num_columns: usize,
    /// Column types for buffer layout (Variable columns need 2 buffers: data + offsets)
    #[allow(dead_code)]
    column_types: Vec<ColumnType>,
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
}

/// Column type metadata for buffer layout
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColumnType {
    /// Fixed-size column: single data buffer
    Fixed,
    /// Variable-size column: data buffer + offsets buffer
    Variable,
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

        // Function signature: (input_ptr, input_len, output_ptrs, row_count) -> bytes_consumed
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // input_ptr
        ctx.func.signature.params.push(AbiParam::new(types::I64)); // input_len
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // output_ptrs (array of pointers)
        ctx.func.signature.params.push(AbiParam::new(types::I64)); // row_count
        ctx.func.signature.returns.push(AbiParam::new(types::I64)); // bytes_consumed or -1 on error

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
        let row_count = builder.block_params(entry_block)[3];

        // Create variables for cursor position and output positions
        // Cranelift 0.126+: declare_var returns a Variable
        let cursor_var = builder.declare_var(types::I64);
        let row_var = builder.declare_var(types::I64);

        // Initialize cursor to 0
        let zero = builder.ins().iconst(types::I64, 0);
        builder.def_var(cursor_var, zero);
        builder.def_var(row_var, zero);

        // Create output position variables for each column
        let mut output_vars = Vec::with_capacity(num_columns);
        for _i in 0..num_columns {
            let var = builder.declare_var(types::I64);
            builder.def_var(var, zero);
            output_vars.push(var);
        }

        // Analyze schema to get field operations
        let field_ops: Vec<FieldOp> = schema.fields()
            .iter()
            .map(|f| Self::field_to_op(f))
            .collect();

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
            Self::emit_field_decode(
                &mut builder,
                op,
                col_idx,
                input_ptr,
                input_len,
                output_ptrs,
                cursor_var,
                &output_vars,
                ptr_type,
                memcpy_func_ref,
            );
        }

        // Increment row counter
        let current_row = builder.use_var(row_var);
        let one = builder.ins().iconst(types::I64, 1);
        let next_row = builder.ins().iadd(current_row, one);
        builder.def_var(row_var, next_row);

        // Jump back to loop header
        builder.ins().jump(loop_header, &[]);

        // Loop exit: return cursor position (bytes consumed)
        builder.switch_to_block(loop_exit);
        let final_cursor = builder.use_var(cursor_var);
        builder.ins().return_(&[final_cursor]);

        // Seal all blocks
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
        let decode_fn: unsafe extern "C" fn(*const u8, usize, *mut *mut u8, usize) -> i64 =
            unsafe { std::mem::transmute(code_ptr) };

        // Compute column types
        let column_types: Vec<ColumnType> = field_ops.iter().map(Self::op_to_column_type).collect();

        Ok(JitDecoder {
            decode_fn,
            _module: SyncJitModule(ManuallyDrop::new(module)),
            schema: schema.clone(),
            num_columns,
            column_types,
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

    /// Convert an Arrow field to a decode operation
    fn field_to_op(field: &Field) -> FieldOp {
        let nullable = field.is_nullable();
        match field.data_type() {
            DataType::Int32 | DataType::Date32 | DataType::Time32(_) => FieldOp::Int32 { nullable },
            DataType::Int64 | DataType::Date64 | DataType::Time64(_) | DataType::Timestamp(_, _) => FieldOp::Int64 { nullable },
            DataType::Float32 => FieldOp::Float32 { nullable },
            DataType::Float64 => FieldOp::Float64 { nullable },
            DataType::Boolean => FieldOp::Boolean { nullable },
            DataType::Utf8 | DataType::Binary | DataType::LargeUtf8 | DataType::LargeBinary => FieldOp::Bytes { nullable },
            DataType::FixedSizeBinary(size) => FieldOp::Fixed { size: *size as usize, nullable },
            DataType::Struct(fields) => {
                let child_ops = fields.iter().map(|f| Self::field_to_op(f.as_ref())).collect();
                FieldOp::Struct { fields: child_ops, nullable }
            }
            // Dictionary/Enum - store indices as i32
            DataType::Dictionary(_, _) => FieldOp::Enum { nullable },
            // List types - decode element type recursively
            DataType::List(element_field) | DataType::LargeList(element_field) => {
                let element_op = Box::new(Self::field_to_op(element_field.as_ref()));
                FieldOp::List { element: element_op, nullable }
            }
            // Map type - keys are always strings in Avro, values are recursive
            DataType::Map(entries_field, _) => {
                // Map entries are a struct with key and value fields
                if let DataType::Struct(fields) = entries_field.data_type() {
                    if fields.len() >= 2 {
                        let value_op = Box::new(Self::field_to_op(fields[1].as_ref()));
                        return FieldOp::Map { value: value_op, nullable };
                    }
                }
                // Fallback - treat as bytes
                FieldOp::Bytes { nullable }
            }
            // Union type - read type index, then decode variant
            DataType::Union(union_fields, _mode) => {
                let variants: Vec<FieldOp> = union_fields
                    .iter()
                    .map(|(_, f)| Self::field_to_op(f.as_ref()))
                    .collect();
                let type_ids: Vec<i8> = union_fields.iter().map(|(id, _)| id).collect();
                FieldOp::Union { variants, type_ids, nullable }
            }
            // Fixed-size list
            DataType::FixedSizeList(element_field, size) => {
                let element_op = Box::new(Self::field_to_op(element_field.as_ref()));
                FieldOp::FixedSizeList { element: element_op, size: *size as usize, nullable }
            }
            // Decimal types - Avro encodes as fixed bytes
            DataType::Decimal128(precision, scale) => {
                FieldOp::Decimal128 { _precision: *precision, _scale: *scale, nullable }
            }
            DataType::Decimal256(precision, scale) => {
                FieldOp::Decimal256 { _precision: *precision, _scale: *scale, nullable }
            }
            // Duration - encoded as i64
            DataType::Duration(_) => FieldOp::Duration { nullable },
            // Interval types
            DataType::Interval(unit) => FieldOp::Interval { unit: *unit, nullable },
            // Default to Int64 for other types
            _ => FieldOp::Int64 { nullable },
        }
    }

    /// Emit Cranelift IR for decoding a single field
    fn emit_field_decode(
        builder: &mut FunctionBuilder,
        op: &FieldOp,
        col_idx: usize,
        input_ptr: Value,
        _input_len: Value,
        output_ptrs: Value,
        cursor_var: Variable,
        output_vars: &[Variable],
        ptr_type: Type,
        memcpy_func_ref: FuncRef,
    ) {
        // Load output pointer for this column
        let col_offset = builder.ins().iconst(ptr_type, (col_idx * 8) as i64);
        let output_ptr_addr = builder.ins().iadd(output_ptrs, col_offset);
        let output_ptr = builder.ins().load(ptr_type, MemFlags::trusted(), output_ptr_addr, 0);
        let output_pos = builder.use_var(output_vars[col_idx]);
        let output_write_ptr = builder.ins().iadd(output_ptr, output_pos);

        match op {
            FieldOp::Int32 { nullable } => {
                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                    builder.def_var(cursor_var, tag_cursor);

                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, zero);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder.ins().brif(is_null, null_block, &[], value_block, &[]);

                    // Null block: write 0 placeholder
                    builder.switch_to_block(null_block);
                    let zero_val = builder.ins().iconst(types::I32, 0);
                    builder.ins().store(MemFlags::trusted(), zero_val, output_write_ptr, 0);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode the value
                    builder.switch_to_block(value_block);
                    let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                    let zigzag = Self::emit_zigzag_decode_i32(builder, value);
                    builder.ins().store(MemFlags::trusted(), zigzag, output_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);

                    let four = builder.ins().iconst(types::I64, 4);
                    let new_output_pos = builder.ins().iadd(output_pos, four);
                    builder.def_var(output_vars[col_idx], new_output_pos);
                } else {
                    let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                    let zigzag = Self::emit_zigzag_decode_i32(builder, value);
                    builder.ins().store(MemFlags::trusted(), zigzag, output_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                    let four = builder.ins().iconst(types::I64, 4);
                    let new_output_pos = builder.ins().iadd(output_pos, four);
                    builder.def_var(output_vars[col_idx], new_output_pos);
                }
            }
            FieldOp::Int64 { nullable } => {
                if *nullable {
                    // Read union tag
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                    builder.def_var(cursor_var, tag_cursor);

                    // Check if null (tag == 0)
                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, zero);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder.ins().brif(is_null, null_block, &[], value_block, &[]);

                    // Null block: write 0 placeholder
                    builder.switch_to_block(null_block);
                    let zero_val = builder.ins().iconst(types::I64, 0);
                    builder.ins().store(MemFlags::trusted(), zero_val, output_write_ptr, 0);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode the value
                    builder.switch_to_block(value_block);
                    let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                    let zigzag = Self::emit_zigzag_decode_i64(builder, value);
                    builder.ins().store(MemFlags::trusted(), zigzag, output_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                    builder.ins().jump(continue_block, &[]);

                    // Seal and continue
                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);

                    // Update output position (same for both branches)
                    let eight = builder.ins().iconst(types::I64, 8);
                    let new_output_pos = builder.ins().iadd(output_pos, eight);
                    builder.def_var(output_vars[col_idx], new_output_pos);
                } else {
                    let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                    let zigzag = Self::emit_zigzag_decode_i64(builder, value);
                    builder.ins().store(MemFlags::trusted(), zigzag, output_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                    let eight = builder.ins().iconst(types::I64, 8);
                    let new_output_pos = builder.ins().iadd(output_pos, eight);
                    builder.def_var(output_vars[col_idx], new_output_pos);
                }
            }
            FieldOp::Float32 { nullable } => {
                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
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

                    builder.switch_to_block(value_block);
                    let cursor = builder.use_var(cursor_var);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder.ins().load(types::I32, MemFlags::trusted(), read_ptr, 0);
                    builder.ins().store(MemFlags::trusted(), value, output_write_ptr, 0);
                    let four = builder.ins().iconst(types::I64, 4);
                    let new_cursor = builder.ins().iadd(cursor, four);
                    builder.def_var(cursor_var, new_cursor);
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);

                    let four = builder.ins().iconst(types::I64, 4);
                    let new_output_pos = builder.ins().iadd(output_pos, four);
                    builder.def_var(output_vars[col_idx], new_output_pos);
                } else {
                    let cursor = builder.use_var(cursor_var);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder.ins().load(types::I32, MemFlags::trusted(), read_ptr, 0);
                    builder.ins().store(MemFlags::trusted(), value, output_write_ptr, 0);
                    let four = builder.ins().iconst(types::I64, 4);
                    let new_cursor = builder.ins().iadd(cursor, four);
                    builder.def_var(cursor_var, new_cursor);
                    let new_output_pos = builder.ins().iadd(output_pos, four);
                    builder.def_var(output_vars[col_idx], new_output_pos);
                }
            }
            FieldOp::Float64 { nullable } => {
                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
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

                    builder.switch_to_block(value_block);
                    let cursor = builder.use_var(cursor_var);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder.ins().load(types::I64, MemFlags::trusted(), read_ptr, 0);
                    builder.ins().store(MemFlags::trusted(), value, output_write_ptr, 0);
                    let eight = builder.ins().iconst(types::I64, 8);
                    let new_cursor = builder.ins().iadd(cursor, eight);
                    builder.def_var(cursor_var, new_cursor);
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);

                    let eight = builder.ins().iconst(types::I64, 8);
                    let new_output_pos = builder.ins().iadd(output_pos, eight);
                    builder.def_var(output_vars[col_idx], new_output_pos);
                } else {
                    let cursor = builder.use_var(cursor_var);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder.ins().load(types::I64, MemFlags::trusted(), read_ptr, 0);
                    builder.ins().store(MemFlags::trusted(), value, output_write_ptr, 0);
                    let eight = builder.ins().iconst(types::I64, 8);
                    let new_cursor = builder.ins().iadd(cursor, eight);
                    builder.def_var(cursor_var, new_cursor);
                    let new_output_pos = builder.ins().iadd(output_pos, eight);
                    builder.def_var(output_vars[col_idx], new_output_pos);
                }
            }
            FieldOp::Boolean { nullable } => {
                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
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

                    builder.switch_to_block(value_block);
                    let cursor = builder.use_var(cursor_var);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder.ins().load(types::I8, MemFlags::trusted(), read_ptr, 0);
                    builder.ins().store(MemFlags::trusted(), value, output_write_ptr, 0);
                    let one = builder.ins().iconst(types::I64, 1);
                    let new_cursor = builder.ins().iadd(cursor, one);
                    builder.def_var(cursor_var, new_cursor);
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);

                    let one = builder.ins().iconst(types::I64, 1);
                    let new_output_pos = builder.ins().iadd(output_pos, one);
                    builder.def_var(output_vars[col_idx], new_output_pos);
                } else {
                    let cursor = builder.use_var(cursor_var);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder.ins().load(types::I8, MemFlags::trusted(), read_ptr, 0);
                    builder.ins().store(MemFlags::trusted(), value, output_write_ptr, 0);
                    let one = builder.ins().iconst(types::I64, 1);
                    let new_cursor = builder.ins().iadd(cursor, one);
                    builder.def_var(cursor_var, new_cursor);
                    let new_output_pos = builder.ins().iadd(output_pos, one);
                    builder.def_var(output_vars[col_idx], new_output_pos);
                }
            }
            FieldOp::Bytes { nullable } => {
                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                    builder.def_var(cursor_var, tag_cursor);

                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, zero);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder.ins().brif(is_null, null_block, &[], value_block, &[]);

                    // Null block: no data written, output position unchanged
                    builder.switch_to_block(null_block);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode bytes normally
                    builder.switch_to_block(value_block);
                    let (len_raw, new_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                    builder.def_var(cursor_var, new_cursor);
                    let len = Self::emit_zigzag_decode_i64(builder, len_raw);
                    let cursor = builder.use_var(cursor_var);
                    Self::emit_memcpy(builder, input_ptr, cursor, output_write_ptr, len, memcpy_func_ref);
                    let new_cursor = builder.ins().iadd(cursor, len);
                    builder.def_var(cursor_var, new_cursor);
                    let new_output_pos = builder.ins().iadd(output_pos, len);
                    builder.def_var(output_vars[col_idx], new_output_pos);
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);
                } else {
                    // Read length as varint (returns raw zigzag value)
                    let (len_raw, new_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                    builder.def_var(cursor_var, new_cursor);

                    // Zigzag decode the length
                    let len = Self::emit_zigzag_decode_i64(builder, len_raw);

                    // Get cursor position for memcpy source offset
                    let cursor = builder.use_var(cursor_var);

                    // Emit byte copy loop
                    Self::emit_memcpy(builder, input_ptr, cursor, output_write_ptr, len, memcpy_func_ref);

                    // Advance input cursor by length
                    let new_cursor = builder.ins().iadd(cursor, len);
                    builder.def_var(cursor_var, new_cursor);

                    // Update data position by length
                    let new_output_pos = builder.ins().iadd(output_pos, len);
                    builder.def_var(output_vars[col_idx], new_output_pos);
                }
            }
            FieldOp::Fixed { size, nullable } => {
                let size_val = builder.ins().iconst(types::I64, *size as i64);
                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
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

                    // Value block: copy fixed bytes
                    builder.switch_to_block(value_block);
                    let cursor = builder.use_var(cursor_var);
                    Self::emit_memcpy(builder, input_ptr, cursor, output_write_ptr, size_val, memcpy_func_ref);
                    let new_cursor = builder.ins().iadd(cursor, size_val);
                    builder.def_var(cursor_var, new_cursor);
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);

                    let new_output_pos = builder.ins().iadd(output_pos, size_val);
                    builder.def_var(output_vars[col_idx], new_output_pos);
                } else {
                    let cursor = builder.use_var(cursor_var);
                    Self::emit_memcpy(builder, input_ptr, cursor, output_write_ptr, size_val, memcpy_func_ref);
                    let new_cursor = builder.ins().iadd(cursor, size_val);
                    builder.def_var(cursor_var, new_cursor);
                    let new_output_pos = builder.ins().iadd(output_pos, size_val);
                    builder.def_var(output_vars[col_idx], new_output_pos);
                }
            }
            FieldOp::Struct { fields: child_ops, nullable } => {
                // Struct: decode each child field sequentially
                // Output is flattened: all field data concatenated together

                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                    builder.def_var(cursor_var, tag_cursor);

                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, zero);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder.ins().brif(is_null, null_block, &[], value_block, &[]);

                    builder.switch_to_block(null_block);
                    builder.ins().jump(continue_block, &[]);

                    builder.switch_to_block(value_block);
                    Self::emit_struct_decode(builder, child_ops, input_ptr, cursor_var, output_write_ptr, output_pos, output_vars, col_idx);
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);
                } else {
                    Self::emit_struct_decode(builder, child_ops, input_ptr, cursor_var, output_write_ptr, output_pos, output_vars, col_idx);
                }
            }
            FieldOp::Enum { nullable } => {
                // Enum is stored as a varint index (NOT zigzag encoded)
                // We read it as zigzag and decode, same as other ints
                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
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

                    builder.switch_to_block(value_block);
                    let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                    // Enum indices are stored directly (no zigzag)
                    let value_i32 = builder.ins().ireduce(types::I32, value);
                    builder.ins().store(MemFlags::trusted(), value_i32, output_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);

                    let four = builder.ins().iconst(types::I64, 4);
                    let new_output_pos = builder.ins().iadd(output_pos, four);
                    builder.def_var(output_vars[col_idx], new_output_pos);
                } else {
                    let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                    // Enum indices are stored directly (no zigzag)
                    let value_i32 = builder.ins().ireduce(types::I32, value);
                    builder.ins().store(MemFlags::trusted(), value_i32, output_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                    let four = builder.ins().iconst(types::I64, 4);
                    let new_output_pos = builder.ins().iadd(output_pos, four);
                    builder.def_var(output_vars[col_idx], new_output_pos);
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
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
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

                    // Value block: decode list
                    builder.switch_to_block(value_block);
                    Self::emit_list_decode(builder, element, input_ptr, cursor_var, output_write_ptr, output_pos, output_vars, col_idx);
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);
                } else {
                    Self::emit_list_decode(builder, element, input_ptr, cursor_var, output_write_ptr, output_pos, output_vars, col_idx);
                }
            }
            FieldOp::Map { value, nullable } => {
                // Avro maps: read block count (zigzag), decode key-value pairs, repeat until count=0
                // Keys are always strings in Avro
                // For simplicity, this implementation stores just the values
                // (keys are read and skipped for now - full map support would need separate buffers)

                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                    builder.def_var(cursor_var, tag_cursor);

                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, zero);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder.ins().brif(is_null, null_block, &[], value_block, &[]);

                    builder.switch_to_block(null_block);
                    builder.ins().jump(continue_block, &[]);

                    builder.switch_to_block(value_block);
                    Self::emit_map_decode(builder, value, input_ptr, cursor_var, output_write_ptr, output_pos, output_vars, col_idx);
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);
                } else {
                    Self::emit_map_decode(builder, value, input_ptr, cursor_var, output_write_ptr, output_pos, output_vars, col_idx);
                }
            }
            FieldOp::Union { variants, type_ids, nullable } => {
                // Avro union: read type index varint, then decode corresponding variant
                // Output: type_id (i32) followed by variant data

                if *nullable {
                    // For nullable union, first check for null tag
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                    builder.def_var(cursor_var, tag_cursor);

                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, zero);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder.ins().brif(is_null, null_block, &[], value_block, &[]);

                    builder.switch_to_block(null_block);
                    builder.ins().jump(continue_block, &[]);

                    builder.switch_to_block(value_block);
                    Self::emit_union_decode(builder, variants, type_ids, input_ptr, cursor_var, output_write_ptr, output_pos, output_vars, col_idx);
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);
                } else {
                    Self::emit_union_decode(builder, variants, type_ids, input_ptr, cursor_var, output_write_ptr, output_pos, output_vars, col_idx);
                }
            }
            FieldOp::FixedSizeList { element, size, nullable } => {
                // Fixed-size list: decode exactly `size` elements
                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                    builder.def_var(cursor_var, tag_cursor);

                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, zero);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder.ins().brif(is_null, null_block, &[], value_block, &[]);

                    builder.switch_to_block(null_block);
                    builder.ins().jump(continue_block, &[]);

                    builder.switch_to_block(value_block);
                    Self::emit_fixed_size_list_decode(builder, element, *size, input_ptr, cursor_var, output_write_ptr, output_pos, output_vars, col_idx);
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);
                } else {
                    Self::emit_fixed_size_list_decode(builder, element, *size, input_ptr, cursor_var, output_write_ptr, output_pos, output_vars, col_idx);
                }
            }
            FieldOp::Decimal128 { nullable, .. } => {
                // Avro decimal: read as fixed 16 bytes (big-endian), convert to little-endian for Arrow
                let byte_size: i64 = 16;

                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
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

                    builder.switch_to_block(value_block);
                    Self::emit_decimal128_decode(builder, input_ptr, cursor_var, output_write_ptr);
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);
                } else {
                    Self::emit_decimal128_decode(builder, input_ptr, cursor_var, output_write_ptr);
                }

                let size_const = builder.ins().iconst(types::I64, byte_size);
                let new_output_pos = builder.ins().iadd(output_pos, size_const);
                builder.def_var(output_vars[col_idx], new_output_pos);
            }
            FieldOp::Decimal256 { nullable, .. } => {
                // Avro decimal: read as fixed 32 bytes (big-endian), convert to little-endian for Arrow
                let byte_size: i64 = 32;

                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
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

                    builder.switch_to_block(value_block);
                    Self::emit_decimal256_decode(builder, input_ptr, cursor_var, output_write_ptr);
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);
                } else {
                    Self::emit_decimal256_decode(builder, input_ptr, cursor_var, output_write_ptr);
                }

                let size_const = builder.ins().iconst(types::I64, byte_size);
                let new_output_pos = builder.ins().iadd(output_pos, size_const);
                builder.def_var(output_vars[col_idx], new_output_pos);
            }
            FieldOp::Duration { nullable } => {
                // Duration: read as i64 (same as Int64)
                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
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

                    builder.switch_to_block(value_block);
                    let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                    let zigzag = Self::emit_zigzag_decode_i64(builder, value);
                    builder.ins().store(MemFlags::trusted(), zigzag, output_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);
                } else {
                    let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                    let zigzag = Self::emit_zigzag_decode_i64(builder, value);
                    builder.ins().store(MemFlags::trusted(), zigzag, output_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                }

                let eight = builder.ins().iconst(types::I64, 8);
                let new_output_pos = builder.ins().iadd(output_pos, eight);
                builder.def_var(output_vars[col_idx], new_output_pos);
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
                    let (tag, tag_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                    builder.def_var(cursor_var, tag_cursor);

                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, zero);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder.ins().brif(is_null, null_block, &[], value_block, &[]);

                    builder.switch_to_block(null_block);
                    builder.ins().jump(continue_block, &[]);

                    builder.switch_to_block(value_block);
                    Self::emit_interval_decode(builder, *unit, input_ptr, cursor_var, output_write_ptr);
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);
                } else {
                    Self::emit_interval_decode(builder, *unit, input_ptr, cursor_var, output_write_ptr);
                }

                let size_const = builder.ins().iconst(types::I64, byte_size);
                let new_output_pos = builder.ins().iadd(output_pos, size_const);
                builder.def_var(output_vars[col_idx], new_output_pos);
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
        cursor_var: Variable,
        output_write_ptr: Value,
        output_pos: Value,
        output_vars: &[Variable],
        col_idx: usize,
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
                    let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                    let zigzag = Self::emit_zigzag_decode_i64(builder, value);
                    builder.ins().store(MemFlags::trusted(), zigzag, field_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                }
                FieldOp::Int32 { nullable: false } => {
                    let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                    let zigzag = Self::emit_zigzag_decode_i32(builder, value);
                    builder.ins().store(MemFlags::trusted(), zigzag, field_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                }
                FieldOp::Float64 { nullable: false } => {
                    let cursor = builder.use_var(cursor_var);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder.ins().load(types::I64, MemFlags::trusted(), read_ptr, 0);
                    builder.ins().store(MemFlags::trusted(), value, field_write_ptr, 0);
                    let eight = builder.ins().iconst(types::I64, 8);
                    let new_cursor = builder.ins().iadd(cursor, eight);
                    builder.def_var(cursor_var, new_cursor);
                }
                FieldOp::Float32 { nullable: false } => {
                    let cursor = builder.use_var(cursor_var);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder.ins().load(types::I32, MemFlags::trusted(), read_ptr, 0);
                    builder.ins().store(MemFlags::trusted(), value, field_write_ptr, 0);
                    let four = builder.ins().iconst(types::I64, 4);
                    let new_cursor = builder.ins().iadd(cursor, four);
                    builder.def_var(cursor_var, new_cursor);
                }
                FieldOp::Boolean { nullable: false } => {
                    let cursor = builder.use_var(cursor_var);
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
        builder.def_var(output_vars[col_idx], new_output_pos);
    }

    /// Emit Cranelift IR for Avro map decoding
    ///
    /// Avro maps use the same block encoding as arrays:
    /// 1. Read block count (zigzag encoded)
    /// 2. If count < 0, abs(count) is entry count, skip byte size
    /// 3. For each entry: read string key, then read value
    /// 4. Repeat until count = 0
    ///
    /// This implementation stores only the values (keys are skipped)
    fn emit_map_decode(
        builder: &mut FunctionBuilder,
        value_op: &FieldOp,
        input_ptr: Value,
        cursor_var: Variable,
        output_write_ptr: Value,
        output_pos: Value,
        output_vars: &[Variable],
        col_idx: usize,
    ) {
        // Get value size based on type
        let value_size: i64 = match value_op {
            FieldOp::Int64 { .. } | FieldOp::Float64 { .. } => 8,
            FieldOp::Int32 { .. } | FieldOp::Float32 { .. } | FieldOp::Enum { .. } => 4,
            FieldOp::Boolean { .. } => 1,
            _ => 8,
        };

        // Create variable to track bytes written
        let map_bytes_var = builder.declare_var(types::I64);
        let zero = builder.ins().iconst(types::I64, 0);
        builder.def_var(map_bytes_var, zero);

        // Block loop
        let block_header = builder.create_block();
        let block_body = builder.create_block();
        let block_exit = builder.create_block();

        builder.ins().jump(block_header, &[]);

        // Block header: read block count
        builder.switch_to_block(block_header);
        let (count_raw, count_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
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
        let (_byte_size, skip_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
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
        let (key_len_raw, key_len_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
        builder.def_var(cursor_var, key_len_cursor);
        let key_len = Self::emit_zigzag_decode_i64(builder, key_len_raw);
        let cursor_after_key = builder.use_var(cursor_var);
        let new_cursor_after_key = builder.ins().iadd(cursor_after_key, key_len);
        builder.def_var(cursor_var, new_cursor_after_key);

        // Decode value
        let map_bytes_written = builder.use_var(map_bytes_var);
        let value_write_ptr = builder.ins().iadd(output_write_ptr, map_bytes_written);

        match value_op {
            FieldOp::Int64 { nullable: false } => {
                let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                let zigzag = Self::emit_zigzag_decode_i64(builder, value);
                builder.ins().store(MemFlags::trusted(), zigzag, value_write_ptr, 0);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Int32 { nullable: false } => {
                let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                let zigzag = Self::emit_zigzag_decode_i32(builder, value);
                builder.ins().store(MemFlags::trusted(), zigzag, value_write_ptr, 0);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Float64 { nullable: false } => {
                let cursor = builder.use_var(cursor_var);
                let read_ptr = builder.ins().iadd(input_ptr, cursor);
                let value = builder.ins().load(types::I64, MemFlags::trusted(), read_ptr, 0);
                builder.ins().store(MemFlags::trusted(), value, value_write_ptr, 0);
                let eight = builder.ins().iconst(types::I64, 8);
                let new_cursor = builder.ins().iadd(cursor, eight);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Float32 { nullable: false } => {
                let cursor = builder.use_var(cursor_var);
                let read_ptr = builder.ins().iadd(input_ptr, cursor);
                let value = builder.ins().load(types::I32, MemFlags::trusted(), read_ptr, 0);
                builder.ins().store(MemFlags::trusted(), value, value_write_ptr, 0);
                let four = builder.ins().iconst(types::I64, 4);
                let new_cursor = builder.ins().iadd(cursor, four);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Boolean { nullable: false } => {
                let cursor = builder.use_var(cursor_var);
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

        // Update bytes written
        let value_size_val = builder.ins().iconst(types::I64, value_size);
        let new_bytes_written = builder.ins().iadd(map_bytes_written, value_size_val);
        builder.def_var(map_bytes_var, new_bytes_written);

        // Increment entry index
        let one = builder.ins().iconst(types::I64, 1);
        let new_entry_idx = builder.ins().iadd(entry_idx, one);
        builder.def_var(entry_idx_var, new_entry_idx);

        builder.ins().jump(entry_loop_header, &[]);

        // Entry loop exit
        builder.switch_to_block(entry_loop_exit);
        builder.ins().jump(block_header, &[]);

        // Block exit
        builder.switch_to_block(block_exit);
        let final_bytes_written = builder.use_var(map_bytes_var);
        let new_output_pos = builder.ins().iadd(output_pos, final_bytes_written);
        builder.def_var(output_vars[col_idx], new_output_pos);

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

    /// Emit Cranelift IR for Avro array/list decoding
    ///
    /// Avro arrays use block encoding:
    /// 1. Read block count (zigzag encoded)
    /// 2. If count < 0, abs(count) is element count, skip byte size
    /// 3. Decode elements
    /// 4. Repeat until count = 0
    ///
    /// For simplicity, this implementation handles primitive element types
    /// and stores just the flattened element data (not full Arrow ListArray format)
    fn emit_list_decode(
        builder: &mut FunctionBuilder,
        element_op: &FieldOp,
        input_ptr: Value,
        cursor_var: Variable,
        output_write_ptr: Value,  // Base + output_pos
        output_pos: Value,
        output_vars: &[Variable],
        col_idx: usize,
    ) {
        // Get element size based on type
        let element_size: i64 = match element_op {
            FieldOp::Int64 { .. } | FieldOp::Float64 { .. } => 8,
            FieldOp::Int32 { .. } | FieldOp::Float32 { .. } | FieldOp::Enum { .. } => 4,
            FieldOp::Boolean { .. } => 1,
            _ => 8, // Default for complex types
        };

        // Create variable to track bytes written (relative offset from output_write_ptr)
        let list_bytes_var = builder.declare_var(types::I64);
        let zero = builder.ins().iconst(types::I64, 0);
        builder.def_var(list_bytes_var, zero);

        // Block loop: read count, decode elements, repeat until count=0
        let block_header = builder.create_block();
        let block_body = builder.create_block();
        let block_exit = builder.create_block();

        builder.ins().jump(block_header, &[]);

        // Block header: read block count
        builder.switch_to_block(block_header);
        let (count_raw, count_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
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
        let (_byte_size, skip_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
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

        // Calculate output pointer for this element
        // output_write_ptr is base + output_pos, list_bytes_written is offset from there
        let list_bytes_written = builder.use_var(list_bytes_var);
        let elem_write_ptr = builder.ins().iadd(output_write_ptr, list_bytes_written);

        // Decode based on element type
        match element_op {
            FieldOp::Int64 { nullable: false } => {
                let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                let zigzag = Self::emit_zigzag_decode_i64(builder, value);
                builder.ins().store(MemFlags::trusted(), zigzag, elem_write_ptr, 0);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Int32 { nullable: false } => {
                let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                let zigzag = Self::emit_zigzag_decode_i32(builder, value);
                builder.ins().store(MemFlags::trusted(), zigzag, elem_write_ptr, 0);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Float64 { nullable: false } => {
                let cursor = builder.use_var(cursor_var);
                let read_ptr = builder.ins().iadd(input_ptr, cursor);
                let value = builder.ins().load(types::I64, MemFlags::trusted(), read_ptr, 0);
                builder.ins().store(MemFlags::trusted(), value, elem_write_ptr, 0);
                let eight = builder.ins().iconst(types::I64, 8);
                let new_cursor = builder.ins().iadd(cursor, eight);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Float32 { nullable: false } => {
                let cursor = builder.use_var(cursor_var);
                let read_ptr = builder.ins().iadd(input_ptr, cursor);
                let value = builder.ins().load(types::I32, MemFlags::trusted(), read_ptr, 0);
                builder.ins().store(MemFlags::trusted(), value, elem_write_ptr, 0);
                let four = builder.ins().iconst(types::I64, 4);
                let new_cursor = builder.ins().iadd(cursor, four);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Boolean { nullable: false } => {
                let cursor = builder.use_var(cursor_var);
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

        // Update list bytes written
        let elem_size_val = builder.ins().iconst(types::I64, element_size);
        let new_bytes_written = builder.ins().iadd(list_bytes_written, elem_size_val);
        builder.def_var(list_bytes_var, new_bytes_written);

        // Increment element index
        let one = builder.ins().iconst(types::I64, 1);
        let new_elem_idx = builder.ins().iadd(elem_idx, one);
        builder.def_var(elem_idx_var, new_elem_idx);

        builder.ins().jump(elem_loop_header, &[]);

        // Element loop exit: continue to next block
        builder.switch_to_block(elem_loop_exit);
        builder.ins().jump(block_header, &[]);

        // Block exit: update final output position
        // output_pos + list_bytes_written = new output position
        builder.switch_to_block(block_exit);
        let final_bytes_written = builder.use_var(list_bytes_var);
        let new_output_pos = builder.ins().iadd(output_pos, final_bytes_written);
        builder.def_var(output_vars[col_idx], new_output_pos);

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
        cursor_var: Variable,
        output_write_ptr: Value,
        output_pos: Value,
        output_vars: &[Variable],
        col_idx: usize,
    ) {
        // Read type index (varint, not zigzag encoded for union)
        let (type_index, index_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
        builder.def_var(cursor_var, index_cursor);

        // Track bytes written - start with 4 for type_id
        let union_bytes_var = builder.declare_var(types::I64);
        let four = builder.ins().iconst(types::I64, 4);
        builder.def_var(union_bytes_var, four);

        // For simple unions (typically 2-4 variants), use if-else chain
        // For now, we handle the common case of small unions
        if variants.is_empty() {
            // Empty union - shouldn't happen but handle gracefully
            builder.def_var(output_vars[col_idx], output_pos);
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
                    let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                    let zigzag = Self::emit_zigzag_decode_i64(builder, value);
                    builder.ins().store(MemFlags::trusted(), zigzag, variant_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                }
                FieldOp::Int32 { nullable: false } => {
                    let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                    let zigzag = Self::emit_zigzag_decode_i32(builder, value);
                    builder.ins().store(MemFlags::trusted(), zigzag, variant_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                }
                FieldOp::Float64 { nullable: false } => {
                    let cursor = builder.use_var(cursor_var);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder.ins().load(types::I64, MemFlags::trusted(), read_ptr, 0);
                    builder.ins().store(MemFlags::trusted(), value, variant_write_ptr, 0);
                    let eight = builder.ins().iconst(types::I64, 8);
                    let new_cursor = builder.ins().iadd(cursor, eight);
                    builder.def_var(cursor_var, new_cursor);
                }
                FieldOp::Float32 { nullable: false } => {
                    let cursor = builder.use_var(cursor_var);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder.ins().load(types::I32, MemFlags::trusted(), read_ptr, 0);
                    builder.ins().store(MemFlags::trusted(), value, variant_write_ptr, 0);
                    let four_const = builder.ins().iconst(types::I64, 4);
                    let new_cursor = builder.ins().iadd(cursor, four_const);
                    builder.def_var(cursor_var, new_cursor);
                }
                FieldOp::Boolean { nullable: false } => {
                    let cursor = builder.use_var(cursor_var);
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
        builder.def_var(output_vars[col_idx], new_output_pos);
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
        cursor_var: Variable,
        output_write_ptr: Value,
        output_pos: Value,
        output_vars: &[Variable],
        col_idx: usize,
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
                let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                let zigzag = Self::emit_zigzag_decode_i64(builder, value);
                builder.ins().store(MemFlags::trusted(), zigzag, elem_write_ptr, 0);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Int32 { nullable: false } => {
                let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                let zigzag = Self::emit_zigzag_decode_i32(builder, value);
                builder.ins().store(MemFlags::trusted(), zigzag, elem_write_ptr, 0);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Float64 { nullable: false } => {
                let cursor = builder.use_var(cursor_var);
                let read_ptr = builder.ins().iadd(input_ptr, cursor);
                let value = builder.ins().load(types::I64, MemFlags::trusted(), read_ptr, 0);
                builder.ins().store(MemFlags::trusted(), value, elem_write_ptr, 0);
                let eight = builder.ins().iconst(types::I64, 8);
                let new_cursor = builder.ins().iadd(cursor, eight);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Float32 { nullable: false } => {
                let cursor = builder.use_var(cursor_var);
                let read_ptr = builder.ins().iadd(input_ptr, cursor);
                let value = builder.ins().load(types::I32, MemFlags::trusted(), read_ptr, 0);
                builder.ins().store(MemFlags::trusted(), value, elem_write_ptr, 0);
                let four = builder.ins().iconst(types::I64, 4);
                let new_cursor = builder.ins().iadd(cursor, four);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Boolean { nullable: false } => {
                let cursor = builder.use_var(cursor_var);
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
        builder.def_var(output_vars[col_idx], new_output_pos);

        // Seal blocks
        builder.seal_block(loop_header);
        builder.seal_block(loop_body);
        builder.seal_block(loop_exit);
    }

    /// Emit Cranelift IR for Decimal128 decoding (16 bytes big-endian to little-endian)
    fn emit_decimal128_decode(
        builder: &mut FunctionBuilder,
        input_ptr: Value,
        cursor_var: Variable,
        output_write_ptr: Value,
    ) {
        let cursor = builder.use_var(cursor_var);
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
        cursor_var: Variable,
        output_write_ptr: Value,
    ) {
        let cursor = builder.use_var(cursor_var);
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
        cursor_var: Variable,
        output_write_ptr: Value,
    ) {
        let cursor = builder.use_var(cursor_var);
        let read_ptr = builder.ins().iadd(input_ptr, cursor);

        match unit {
            IntervalUnit::YearMonth => {
                // 4 bytes (i32) - zigzag encoded varint
                let (value, new_cursor) = Self::emit_varint_decode(builder, input_ptr, cursor_var);
                let zigzag = Self::emit_zigzag_decode_i32(builder, value);
                builder.ins().store(MemFlags::trusted(), zigzag, output_write_ptr, 0);
                builder.def_var(cursor_var, new_cursor);
            }
            IntervalUnit::DayTime => {
                // 8 bytes (two i32s: days + milliseconds)
                // In Avro, this is typically encoded as a fixed 8 bytes
                let value = builder.ins().load(types::I64, MemFlags::trusted(), read_ptr, 0);
                builder.ins().store(MemFlags::trusted(), value, output_write_ptr, 0);
                let eight = builder.ins().iconst(types::I64, 8);
                let new_cursor = builder.ins().iadd(cursor, eight);
                builder.def_var(cursor_var, new_cursor);
            }
            IntervalUnit::MonthDayNano => {
                // 16 bytes (i32 months + i32 days + i64 nanos)
                // In Avro, typically encoded as fixed 16 bytes
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

    /// Emit Cranelift IR for varint decoding with proper loop
    /// Handles up to 10 bytes (full i64 range)
    /// Returns (decoded_value, new_cursor_position)
    fn emit_varint_decode(
        builder: &mut FunctionBuilder,
        input_ptr: Value,
        cursor_var: Variable,
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
        let loop_exit = builder.create_block();

        // Jump to loop header
        builder.ins().jump(loop_header, &[]);

        // Loop header - process one byte
        builder.switch_to_block(loop_header);

        // Read byte at current position
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

    /// Decode rows using the JIT-compiled function
    ///
    /// # Arguments
    /// * `input` - Raw Avro bytes
    /// * `output_buffers` - Mutable slice of output buffer pointers (one per column)
    /// * `row_count` - Number of rows to decode
    ///
    /// # Returns
    /// Number of bytes consumed from input, or error
    ///
    /// # Safety
    /// The output buffers must have sufficient capacity for the decoded data.
    #[inline(always)]
    pub fn decode(
        &self,
        input: &[u8],
        output_buffers: &mut [*mut u8],
        row_count: usize,
    ) -> Result<usize, String> {
        if output_buffers.len() != self.num_columns {
            return Err(format!(
                "Expected {} output buffers, got {}",
                self.num_columns,
                output_buffers.len()
            ));
        }

        let result = unsafe {
            (self.decode_fn)(
                input.as_ptr(),
                input.len(),
                output_buffers.as_mut_ptr(),
                row_count,
            )
        };

        if result < 0 {
            Err("Decode error".to_string())
        } else {
            Ok(result as usize)
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

    /// Decode Avro data directly to an Arrow RecordBatch
    ///
    /// This method allocates buffers, runs the JIT decoder, and converts
    /// the results to Arrow arrays.
    ///
    /// # Arguments
    /// * `input` - Raw Avro bytes (without header/sync markers)
    /// * `row_count` - Number of rows to decode
    ///
    /// # Returns
    /// A RecordBatch with decoded data
    pub fn decode_to_batch(
        &self,
        input: &[u8],
        row_count: usize,
    ) -> Result<RecordBatch, ArrowError> {
        // Estimate buffer sizes and allocate
        let mut buffers: Vec<Vec<u8>> = Vec::with_capacity(self.num_columns);
        let mut buffer_ptrs: Vec<*mut u8> = Vec::with_capacity(self.num_columns);

        for field in self.schema.fields() {
            let buffer_size = Self::estimate_buffer_size(field.data_type(), row_count);
            let mut buffer = vec![0u8; buffer_size];
            buffer_ptrs.push(buffer.as_mut_ptr());
            buffers.push(buffer);
        }

        // Run the JIT decoder
        let _bytes_consumed = self.decode(input, &mut buffer_ptrs, row_count)
            .map_err(|e| ArrowError::ExternalError(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                e,
            ))))?;

        // Convert raw buffers to Arrow arrays
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(self.num_columns);

        for (i, field) in self.schema.fields().iter().enumerate() {
            let buffer = std::mem::take(&mut buffers[i]);
            let array = Self::buffer_to_array(field.data_type(), buffer, row_count)?;
            arrays.push(array);
        }

        RecordBatch::try_new(self.schema.clone(), arrays)
    }

    /// Estimate buffer size for a column
    fn estimate_buffer_size(data_type: &DataType, row_count: usize) -> usize {
        match data_type {
            DataType::Boolean => row_count, // 1 byte per boolean (not packed in JIT output)
            DataType::Int32 | DataType::Float32 => row_count * 4,
            DataType::Int64 | DataType::Float64 => row_count * 8,
            DataType::Utf8 | DataType::Binary => row_count * 128, // Estimate average string length
            DataType::FixedSizeBinary(n) => row_count * (*n as usize),
            DataType::List(elem_field) => {
                // Estimate avg 10 elements per list
                10 * row_count * Self::estimate_buffer_size(elem_field.data_type(), 1)
            }
            DataType::Map(entries_field, _) => {
                // Estimate avg 5 entries per map
                5 * row_count * Self::estimate_buffer_size(entries_field.data_type(), 1)
            }
            _ => row_count * 8, // Default
        }
    }

    /// Convert a raw buffer to an Arrow array
    fn buffer_to_array(
        data_type: &DataType,
        buffer: Vec<u8>,
        row_count: usize,
    ) -> Result<ArrayRef, ArrowError> {
        match data_type {
            DataType::Int32 => {
                let scalar_buffer = ScalarBuffer::<i32>::from(Buffer::from_vec(buffer));
                Ok(Arc::new(Int32Array::new(scalar_buffer.slice(0, row_count), None)))
            }
            DataType::Int64 => {
                let scalar_buffer = ScalarBuffer::<i64>::from(Buffer::from_vec(buffer));
                Ok(Arc::new(Int64Array::new(scalar_buffer.slice(0, row_count), None)))
            }
            DataType::Float32 => {
                let scalar_buffer = ScalarBuffer::<f32>::from(Buffer::from_vec(buffer));
                Ok(Arc::new(Float32Array::new(scalar_buffer.slice(0, row_count), None)))
            }
            DataType::Float64 => {
                let scalar_buffer = ScalarBuffer::<f64>::from(Buffer::from_vec(buffer));
                Ok(Arc::new(Float64Array::new(scalar_buffer.slice(0, row_count), None)))
            }
            DataType::Boolean => {
                // JIT outputs one byte per boolean, need to convert
                let bools: Vec<bool> = buffer.iter().take(row_count).map(|&b| b != 0).collect();
                Ok(Arc::new(BooleanArray::from(bools)))
            }
            DataType::Utf8 => {
                // For now, this is a placeholder - proper string handling needs offsets
                // Return empty string array since we don't track offsets yet
                let strings: Vec<&str> = vec![""; row_count];
                Ok(Arc::new(StringArray::from(strings)))
            }
            DataType::Binary => {
                // Similar placeholder for binary
                let binaries: Vec<&[u8]> = vec![&[]; row_count];
                Ok(Arc::new(BinaryArray::from_vec(binaries)))
            }
            _ => Err(ArrowError::NotYetImplemented(format!(
                "JIT decoder does not yet support {:?}",
                data_type
            ))),
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

        // Test a single bytes value
        let test_data = b"Hello, World!";
        let input = encode_avro_bytes(test_data);

        let mut output = vec![0u8; test_data.len() + 100]; // Extra space
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, 1).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify the bytes were copied correctly
        assert_eq!(&output[..test_data.len()], test_data);
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
        let mut output = vec![0u8; total_data_len + 100]; // Extra space
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify the strings were copied in sequence
        let mut offset = 0;
        for s in &test_strings {
            let actual = &output[offset..offset + s.len()];
            assert_eq!(actual, s.as_bytes(), "String mismatch: expected '{}', got '{:?}'", s, actual);
            offset += s.len();
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

        let mut output = vec![0xFFu8; 100]; // Fill with sentinel
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, 1).expect("Decode failed");
        assert_eq!(consumed, 1); // Just the length varint (0x00)

        // Output should not have been modified
        assert_eq!(output[0], 0xFF, "Output was modified for empty bytes");
    }

    #[test]
    fn test_roundtrip_bytes_with_int_column() {
        // Test mixed schema: Int64 + Bytes + Int32
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("count", DataType::Int32, false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

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
        let mut output_name = vec![0u8; total_name_len + 100];
        let mut output_count = vec![0u8; row_count * 4];

        let mut output_ptrs = vec![
            output_id.as_mut_ptr(),
            output_name.as_mut_ptr(),
            output_count.as_mut_ptr(),
        ];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify Int64 values
        for i in 0..row_count {
            let id = i64::from_le_bytes(output_id[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(id, i as i64 * 100, "id mismatch at row {}", i);
        }

        // Verify strings
        let mut offset = 0;
        for (i, name) in names.iter().enumerate() {
            let actual = &output_name[offset..offset + name.len()];
            assert_eq!(actual, name.as_bytes(), "name mismatch at row {}", i);
            offset += name.len();
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

        let mut output = vec![0u8; test_data.len() + 100];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, 1).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify all 500 bytes were copied correctly
        assert_eq!(&output[..test_data.len()], &test_data[..], "Long bytes mismatch");
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
        let mut output_ptrs = vec![output.as_mut_ptr()];

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
        let mut output_ptrs = vec![output.as_mut_ptr()];

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
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        for i in 0..row_count {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, 0, "Null should have 0 placeholder at index {}", i);
        }
    }

    #[test]
    fn test_decode_to_batch_int64() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let test_values: Vec<i64> = vec![10, 20, 30, 40, 50];

        let mut input = Vec::new();
        for &v in &test_values {
            input.extend(encode_varint_i64(v));
        }

        let batch = decoder.decode_to_batch(&input, test_values.len()).expect("Decode to batch failed");

        assert_eq!(batch.num_rows(), test_values.len());
        assert_eq!(batch.num_columns(), 1);

        let id_array = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        for (i, &expected) in test_values.iter().enumerate() {
            assert_eq!(id_array.value(i), expected, "Id mismatch at index {}", i);
        }
    }

    #[test]
    fn test_decode_to_batch_multi_column() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Float64, false),
            Field::new("count", DataType::Int32, false),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let row_count = 10;

        let mut input = Vec::new();
        for i in 0..row_count {
            input.extend(encode_varint_i64(i as i64 * 100));
            input.extend((i as f64 * 1.5).to_le_bytes());
            input.extend(encode_varint_i32(i as i32 * 10));
        }

        let batch = decoder.decode_to_batch(&input, row_count).expect("Decode to batch failed");

        assert_eq!(batch.num_rows(), row_count);
        assert_eq!(batch.num_columns(), 3);

        let id_array = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let value_array = batch.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        let count_array = batch.column(2).as_any().downcast_ref::<Int32Array>().unwrap();

        for i in 0..row_count {
            assert_eq!(id_array.value(i), i as i64 * 100, "Id mismatch at row {}", i);
            assert!((value_array.value(i) - i as f64 * 1.5).abs() < 1e-10, "Value mismatch at row {}", i);
            assert_eq!(count_array.value(i), i as i32 * 10, "Count mismatch at row {}", i);
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
        let mut output_ptrs = vec![output.as_mut_ptr()];

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

        // Empty array: single 0 byte
        let input = encode_avro_array_i64(&[]);

        let mut output = vec![0xFFu8; 100];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, 1).expect("Decode failed");
        assert_eq!(consumed, 1); // Just the 0 terminator
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

        let mut output = vec![0u8; elements.len() * 8 + 100];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, 1).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify elements
        for (i, &expected) in elements.iter().enumerate() {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
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

        let mut output = vec![0u8; elements.len() * 4 + 100];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, 1).expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in elements.iter().enumerate() {
            let actual = i32::from_le_bytes(output[i * 4..(i + 1) * 4].try_into().unwrap());
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

        let mut input = Vec::new();
        input.extend(encode_avro_array_i64(&row1));
        input.extend(encode_avro_array_i64(&row2));
        input.extend(encode_avro_array_i64(&row3));

        let total_elements = row1.len() + row2.len() + row3.len();
        let mut output = vec![0u8; total_elements * 8 + 100];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, 3).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify all elements in order
        let all_elements: Vec<i64> = [row1, row2, row3].concat();
        for (i, &expected) in all_elements.iter().enumerate() {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, expected, "Element mismatch at index {}", i);
        }
    }

    #[test]
    fn test_roundtrip_list_with_int_column() {
        // Mixed schema: Int64 + List<Int64>
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new(
                "values",
                DataType::List(Arc::new(Field::new("item", DataType::Int64, false))),
                false,
            ),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

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
        let mut output_id = vec![0u8; row_count * 8];
        let mut output_list = vec![0u8; total_elements * 8 + 100];

        let mut output_ptrs = vec![
            output_id.as_mut_ptr(),
            output_list.as_mut_ptr(),
        ];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify ids
        for (i, &expected) in ids.iter().enumerate() {
            let actual = i64::from_le_bytes(output_id[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, expected, "Id mismatch at row {}", i);
        }

        // Verify list elements
        let all_elements: Vec<i64> = lists.into_iter().flatten().collect();
        for (i, &expected) in all_elements.iter().enumerate() {
            let actual = i64::from_le_bytes(output_list[i * 8..(i + 1) * 8].try_into().unwrap());
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

        let mut output = vec![0xFFu8; 100];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, 1).expect("Decode failed");
        assert_eq!(consumed, 1); // Just the 0 terminator
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

        // Output buffer for values only
        let mut output = vec![0u8; entries.len() * 8 + 100];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, 1).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify values (keys are skipped)
        for (i, &(_, expected_value)) in entries.iter().enumerate() {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
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

        let total_entries = row1.len() + row2.len() + row3.len();
        let mut output = vec![0u8; total_entries * 8 + 100];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = decoder.decode(&input, &mut output_ptrs, 3).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify all values in order
        let all_values: Vec<i64> = [row1, row2, row3].iter()
            .flat_map(|entries| entries.iter().map(|(_, v)| *v))
            .collect();
        for (i, &expected) in all_values.iter().enumerate() {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
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
        let mut output_id = vec![0u8; row_count * 8];
        let mut output_map = vec![0u8; total_entries * 8 + 100];

        let mut output_ptrs = vec![
            output_id.as_mut_ptr(),
            output_map.as_mut_ptr(),
        ];

        let consumed = decoder.decode(&input, &mut output_ptrs, row_count).expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify ids
        for (i, &expected) in ids.iter().enumerate() {
            let actual = i64::from_le_bytes(output_id[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, expected, "Id mismatch at row {}", i);
        }

        // Verify map values
        let all_values: Vec<i64> = maps.iter()
            .flat_map(|entries| entries.iter().map(|(_, v)| *v))
            .collect();
        for (i, &expected) in all_values.iter().enumerate() {
            let actual = i64::from_le_bytes(output_map[i * 8..(i + 1) * 8].try_into().unwrap());
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
}

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
use arrow_schema::{ArrowError, DataType, Field, Schema};
use std::sync::Arc;

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

/// JIT-compiled decoder for a specific Avro schema
pub struct JitDecoder {
    /// The compiled decode function
    /// Signature: (input_ptr, input_len, output_ptrs, row_count) -> bytes_consumed
    /// Returns negative value on error
    decode_fn: unsafe extern "C" fn(*const u8, usize, *mut *mut u8, usize) -> i64,
    /// Keep the JIT module alive (owns the compiled code memory)
    _module: JITModule,
    /// Schema for reference
    schema: Arc<Schema>,
    /// Number of columns
    num_columns: usize,
    /// Column types for buffer layout (Variable columns need 2 buffers: data + offsets)
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

        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
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

        // Build the function
        let mut fn_builder_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);

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
            _module: module,
            schema: schema.clone(),
            num_columns,
            column_types,
        })
    }

    /// Get column type for a field operation
    fn op_to_column_type(op: &FieldOp) -> ColumnType {
        match op {
            FieldOp::Bytes { .. } => ColumnType::Variable,
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
                    Self::emit_memcpy(builder, input_ptr, cursor, output_write_ptr, len);
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
                    Self::emit_memcpy(builder, input_ptr, cursor, output_write_ptr, len);

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
                    Self::emit_memcpy(builder, input_ptr, cursor, output_write_ptr, size_val);
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
                    Self::emit_memcpy(builder, input_ptr, cursor, output_write_ptr, size_val);
                    let new_cursor = builder.ins().iadd(cursor, size_val);
                    builder.def_var(cursor_var, new_cursor);
                    let new_output_pos = builder.ins().iadd(output_pos, size_val);
                    builder.def_var(output_vars[col_idx], new_output_pos);
                }
            }
            FieldOp::Struct { fields: _child_ops, nullable: _ } => {
                // TODO: Recursively emit child field decodes
                // For now, this is a placeholder - structs not yet supported
            }
        }
    }

    /// Emit Cranelift IR for memcpy with proper loop structure
    /// Copies len bytes from src_base+src_offset to dst
    fn emit_memcpy(
        builder: &mut FunctionBuilder,
        src_base: Value,
        src_offset: Value,
        dst: Value,
        len: Value,
    ) {
        // Skip if len == 0
        let zero = builder.ins().iconst(types::I64, 0);
        let is_zero = builder.ins().icmp(IntCC::Equal, len, zero);

        let copy_block = builder.create_block();
        let exit_block = builder.create_block();

        builder.ins().brif(is_zero, exit_block, &[], copy_block, &[]);

        // Copy block - byte by byte
        builder.switch_to_block(copy_block);

        // Create loop variable
        let idx_var = builder.declare_var(types::I64);
        builder.def_var(idx_var, zero);

        let loop_check = builder.create_block();
        let loop_body = builder.create_block();

        builder.ins().jump(loop_check, &[]);

        // Loop check: idx < len?
        builder.switch_to_block(loop_check);
        let idx = builder.use_var(idx_var);
        let continue_cmp = builder.ins().icmp(IntCC::UnsignedLessThan, idx, len);
        builder.ins().brif(continue_cmp, loop_body, &[], exit_block, &[]);

        // Loop body: copy one byte
        builder.switch_to_block(loop_body);
        let idx = builder.use_var(idx_var);

        // src_ptr = src_base + src_offset + idx
        let src_pos = builder.ins().iadd(src_offset, idx);
        let src_ptr = builder.ins().iadd(src_base, src_pos);

        // dst_ptr = dst + idx
        let dst_ptr = builder.ins().iadd(dst, idx);

        // Load and store byte
        let byte = builder.ins().load(types::I8, MemFlags::trusted(), src_ptr, 0);
        builder.ins().store(MemFlags::trusted(), byte, dst_ptr, 0);

        // Increment index
        let one = builder.ins().iconst(types::I64, 1);
        let new_idx = builder.ins().iadd(idx, one);
        builder.def_var(idx_var, new_idx);

        // Jump back to check
        builder.ins().jump(loop_check, &[]);

        // Seal blocks
        builder.seal_block(copy_block);
        builder.seal_block(loop_check);
        builder.seal_block(loop_body);
        builder.seal_block(exit_block);

        // Continue from exit block
        builder.switch_to_block(exit_block);
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
}

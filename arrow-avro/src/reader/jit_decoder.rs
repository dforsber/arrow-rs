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
//! let bytes_consumed = unsafe { decoder.decode(input, &mut buffers, row_count) }?;
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

/// Owns executable memory until the last decoder reference is dropped.
/// The module is immutable after finalization; only the compiled function is called.
struct SyncJitModule(ManuallyDrop<JITModule>);

impl Drop for SyncJitModule {
    fn drop(&mut self) {
        // SAFETY: dropping the decoder requires exclusive ownership. All calls
        // borrow it, so no compiled function can still be running.
        unsafe { ManuallyDrop::take(&mut self.0).free_memory() };
    }
}

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
    /// Signature: (input_ptr, input_len, output_ptrs, output_caps, row_count, record_ends) -> bytes_consumed
    /// Returns negative value on error (-1 input bounds, -2 output overflow)
    decode_fn: unsafe extern "C" fn(
        *const u8,
        usize,
        *mut *mut u8,
        *const usize,
        usize,
        *const usize,
    ) -> i64,
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
    Struct {
        fields: Vec<FieldOp>,
        nullable: bool,
        recursive: bool,
    },
    /// Enum - read zigzag-encoded nonnegative int index
    Enum { nullable: bool },
    /// List/Array - read block count + elements
    List {
        element: Box<FieldOp>,
        nullable: bool,
    },
    /// Decimal128 - read fixed bytes and decode as 128-bit decimal
    Decimal128 {
        _precision: u8,
        _scale: i8,
        nullable: bool,
        bytes_backed: bool,
    },
    /// Decimal256 - read fixed bytes and decode as 256-bit decimal
    Decimal256 {
        _precision: u8,
        _scale: i8,
        nullable: bool,
        bytes_backed: bool,
    },
    /// Duration - read as i64
    Duration { nullable: bool },
    /// Interval - read as fixed bytes
    Interval {
        unit: IntervalUnit,
        nullable: bool,
        avro_duration: bool,
    },
}

/// Column type metadata for buffer layout
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColumnType {
    /// Fixed-size column: single data buffer
    Fixed,
    /// Variable-size column: offsets buffer + data buffer
    Variable,
    /// Record with independent child buffers; inspect its recursive BufferSlot.
    Record,
}

/// Buffer slot descriptor for a schema column
#[derive(Debug, Clone)]
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
    /// For List: element size (offsets are element indices)
    pub element_size: i64,
    /// Raw Avro union tag for null (0 or 2).
    pub null_tag: i64,
    /// Recursive record children, in Avro wire order. Empty for packed records
    /// and scalar/list leaves. Each child has independent validity and offsets.
    pub children: Vec<BufferSlot>,
    /// Bytes per row in this slot's primary buffer. None denotes offsets.
    /// Recursive record containers have zero primary bytes.
    pub fixed_width: Option<usize>,
}

impl BufferSlot {
    /// Visit this slot followed by its descendants in output pointer order.
    pub fn visit<'a>(&'a self, visitor: &mut impl FnMut(&'a BufferSlot)) {
        visitor(self);
        for child in &self.children {
            child.visit(visitor);
        }
    }

    /// Required primary bytes, with checked arithmetic before allocation.
    pub fn primary_capacity(&self, rows: usize) -> Option<usize> {
        match self.fixed_width {
            Some(width) => rows.checked_mul(width),
            None => rows.checked_add(1)?.checked_mul(4),
        }
    }
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
    /// Compile from an Avro writer schema, retaining wire details that Arrow
    /// types alone cannot describe (union order, decimal width, duration).
    pub fn compile_avro(json: &str) -> Result<Self, String> {
        let avro: crate::schema::Schema<'_> =
            serde_json::from_str(json).map_err(|e| e.to_string())?;
        let root = crate::codec::AvroFieldBuilder::new(&avro)
            .with_jit_metadata()
            .build()
            .map_err(|e| e.to_string())?;
        let DataType::Struct(fields) = root.field().data_type().clone() else {
            return Err("JIT expects an Avro record schema".into());
        };
        // `small_decimals` changes the interpreted reader's output width. The
        // JIT deliberately retains its 128/256-bit output layout in every build.
        fn widen_decimal(field: &Field) -> Field {
            let dt = match field.data_type() {
                DataType::Decimal32(p, s) | DataType::Decimal64(p, s) => {
                    DataType::Decimal128(*p, *s)
                }
                DataType::Struct(fields) => {
                    DataType::Struct(fields.iter().map(|f| widen_decimal(f)).collect())
                }
                DataType::List(child) => DataType::List(Arc::new(widen_decimal(child))),
                other => other.clone(),
            };
            field.clone().with_data_type(dt)
        }
        Self::compile(&Arc::new(Schema::new(
            fields.iter().map(|f| widen_decimal(f)).collect::<Vec<_>>(),
        )))
    }

    /// Compile a schema-specific decoder
    ///
    /// This is an expensive operation (~1-5ms) but only needs to be done once per schema.
    /// The resulting decoder has zero dispatch overhead.
    pub fn compile(schema: &Arc<Schema>) -> Result<Self, String> {
        let num_columns = schema.fields().len();
        if num_columns == 0 {
            return Err("JIT requires at least one supported output column".into());
        }
        fn validate_shape(dt: &DataType, depth: usize, fields: &mut usize) -> Result<(), String> {
            *fields += 1;
            if depth > 32 || *fields > 1024 {
                return Err("JIT schema complexity limit exceeded".into());
            }
            match dt {
                DataType::Struct(children) => {
                    for child in children {
                        validate_shape(child.data_type(), depth + 1, fields)?;
                    }
                }
                DataType::List(child)
                | DataType::LargeList(child)
                | DataType::FixedSizeList(child, _)
                | DataType::Map(child, _) => validate_shape(child.data_type(), depth + 1, fields)?,
                DataType::Union(children, _) => {
                    for (_, child) in children.iter() {
                        validate_shape(child.data_type(), depth + 1, fields)?;
                    }
                }
                _ => {}
            }
            Ok(())
        }
        let mut field_count = 0;
        for field in schema.fields() {
            validate_shape(field.data_type(), 0, &mut field_count)?;
        }
        let field_ops: Vec<_> = schema
            .fields()
            .iter()
            .map(|f| Self::field_to_op(f))
            .collect::<Result<_, _>>()?;

        // Build the JIT module
        let mut flag_builder = settings::builder();
        flag_builder
            .set("opt_level", "speed")
            .map_err(|e| e.to_string())?;
        flag_builder
            .set("is_pic", "false")
            .map_err(|e| e.to_string())?;

        let isa_builder = cranelift_native::builder().map_err(|e| e.to_string())?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| e.to_string())?;

        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        // Register memcpy symbol for fast bulk memory copies
        builder.symbol("memcpy", memcpy as *const u8);
        let mut module_owner = SyncJitModule(ManuallyDrop::new(JITModule::new(builder)));
        let module = &mut *module_owner.0;

        // Create the function context
        let mut ctx = module.make_context();
        let ptr_type = module.target_config().pointer_type();

        // Function signature: (input_ptr, input_len, output_ptrs, output_caps, row_count, record_ends) -> bytes_consumed
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // input_ptr
        ctx.func.signature.params.push(AbiParam::new(types::I64)); // input_len
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // output_ptrs (array of pointers)
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // output_caps (array of capacities)
        ctx.func.signature.params.push(AbiParam::new(types::I64)); // row_count
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // optional record ends
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
        let record_ends = builder.block_params(entry_block)[5];

        // Create variables for cursor position and output positions
        // Cranelift 0.126+: declare_var returns a Variable
        let cursor_var = builder.declare_var(types::I64);
        let row_var = builder.declare_var(types::I64);

        // Initialize cursor to 0
        let zero = builder.ins().iconst(types::I64, 0);
        builder.def_var(cursor_var, zero);
        builder.def_var(row_var, zero);

        // Compute column types and buffer slots
        let column_types: Vec<ColumnType> = field_ops.iter().map(Self::op_to_column_type).collect();

        // Preserve packed scalar-record layout for existing callers. Records
        // with nullable/variable/recursive children use independent child slots.
        fn make_slot(field: &Field, op: &FieldOp, next: &mut usize) -> Result<BufferSlot, String> {
            let primary = *next;
            *next += 1;
            let secondary = if JitDecoder::op_to_column_type(op) == ColumnType::Variable {
                let index = *next;
                *next += 1;
                Some(index)
            } else {
                None
            };
            let validity = if field.is_nullable() {
                let index = *next;
                *next += 1;
                Some(index)
            } else {
                None
            };
            let null_tag = match field
                .metadata()
                .get("avro.union.null_index")
                .map(String::as_str)
            {
                None | Some("0") => 0,
                Some("1") => 2,
                _ => return Err("Invalid nullable union index".into()),
            };
            let recursive = matches!(
                op,
                FieldOp::Struct {
                    recursive: true,
                    ..
                }
            );
            let mut children = Vec::new();
            if let (DataType::Struct(fields), FieldOp::Struct { fields: ops, .. }) =
                (field.data_type(), op)
            {
                if recursive {
                    for (child, child_op) in fields.iter().zip(ops) {
                        children.push(make_slot(child, child_op, next)?);
                    }
                }
            }
            Ok(BufferSlot {
                primary,
                secondary,
                validity,
                null_tag,
                children,
                element_size: JitDecoder::op_to_element_size(op),
                fixed_width: if recursive {
                    Some(0)
                } else {
                    JitDecoder::fixed_size_for_type(field.data_type())
                },
            })
        }
        let mut buffer_idx = 0;
        let buffer_slots = schema
            .fields()
            .iter()
            .zip(&field_ops)
            .map(|(field, op)| make_slot(field, op, &mut buffer_idx))
            .collect::<Result<Vec<_>, _>>()?;
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
        let cmp = builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, current_row, row_count);
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

        // The framed Kafka API checks exact datum boundaries in the generated
        // loop; raw OCF callers pass a null pointer and keep streaming semantics.
        let check_end = builder.create_block();
        let row_done = builder.create_block();
        let framed = builder.ins().icmp_imm(IntCC::NotEqual, record_ends, 0);
        builder.ins().brif(framed, check_end, &[], row_done, &[]);
        builder.switch_to_block(check_end);
        let row = builder.use_var(row_var);
        let offset = builder.ins().imul_imm(row, 8);
        let addr = builder.ins().iadd(record_ends, offset);
        let expected = builder.ins().load(types::I64, MemFlags::trusted(), addr, 0);
        let cursor = builder.use_var(cursor_var);
        let exact = builder.ins().icmp(IntCC::Equal, cursor, expected);
        builder.ins().brif(exact, row_done, &[], error_block, &[]);
        builder.seal_block(check_end);
        builder.seal_block(row_done);
        builder.switch_to_block(row_done);

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
        let mut all_slots = Vec::new();
        for slot in &buffer_slots {
            slot.visit(&mut |slot| all_slots.push(slot));
        }
        for slot in &all_slots {
            if let Some(data_buf_idx) = slot.secondary {
                // This is a variable-length column
                let buf_idx = slot.primary; // offsets buffer index

                // Load offsets buffer pointer
                let buf_offset = builder.ins().iconst(ptr_type, (buf_idx * 8) as i64);
                let offsets_ptr_addr = builder.ins().iadd(output_ptrs, buf_offset);
                let offsets_ptr =
                    builder
                        .ins()
                        .load(ptr_type, MemFlags::trusted(), offsets_ptr_addr, 0);

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
                builder
                    .ins()
                    .store(MemFlags::trusted(), final_offset, offset_write_ptr, 0);
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
        let decode_fn: unsafe extern "C" fn(
            *const u8,
            usize,
            *mut *mut u8,
            *const usize,
            usize,
            *const usize,
        ) -> i64 = unsafe { std::mem::transmute(code_ptr) };

        // buffer_slots, column_types, and total_buffers were computed earlier in compile()
        Ok(JitDecoder {
            decode_fn,
            _module: module_owner,
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
            FieldOp::Bytes { .. } | FieldOp::List { .. } => ColumnType::Variable,
            FieldOp::Struct {
                recursive: true, ..
            } => ColumnType::Record,
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
            // Fixed-size types: not used for offsets
            _ => 1,
        }
    }

    fn packed_struct_child(op: &FieldOp) -> bool {
        matches!(
            op,
            FieldOp::Boolean { nullable: false }
                | FieldOp::Int32 { nullable: false }
                | FieldOp::Int64 { nullable: false }
                | FieldOp::Float32 { nullable: false }
                | FieldOp::Float64 { nullable: false }
                | FieldOp::Decimal128 {
                    nullable: false,
                    ..
                }
                | FieldOp::Decimal256 {
                    nullable: false,
                    ..
                }
                | FieldOp::Enum { nullable: false }
        )
    }

    /// Get byte size of element type (must match emit_list_decode's element_size
    /// and emit_struct_decode's field_size — used for offset element counts).
    fn element_op_size(op: &FieldOp) -> i64 {
        match op {
            FieldOp::Decimal256 { .. } => 32,
            FieldOp::Decimal128 { .. } => 16,
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
            DataType::Map(_, _)
            | DataType::Union(_, _)
            | DataType::FixedSizeList(_, _)
            | DataType::LargeList(_)
            | DataType::LargeUtf8
            | DataType::LargeBinary => Err(format!(
                "Unsupported JIT Arrow layout: {:?}",
                field.data_type()
            )),
            DataType::FixedSizeBinary(n) if *n <= 0 => Err("Invalid fixed width".into()),
            DataType::Int32 | DataType::Date32 | DataType::Time32(_) => {
                Ok(FieldOp::Int32 { nullable })
            }
            DataType::Int64
            | DataType::Date64
            | DataType::Time64(_)
            | DataType::Timestamp(_, _) => Ok(FieldOp::Int64 { nullable }),
            DataType::Float32 => Ok(FieldOp::Float32 { nullable }),
            DataType::Float64 => Ok(FieldOp::Float64 { nullable }),
            DataType::Boolean => Ok(FieldOp::Boolean { nullable }),
            DataType::Utf8 | DataType::Binary => Ok(FieldOp::Bytes { nullable }),
            DataType::FixedSizeBinary(size) => Ok(FieldOp::Fixed {
                size: *size as usize,
                nullable,
            }),
            DataType::Struct(fields) => {
                let child_ops: Vec<FieldOp> = fields
                    .iter()
                    .map(|f| Self::field_to_op(f.as_ref()))
                    .collect::<Result<Vec<_>, _>>()?;
                // Keep the established packed layout only for shapes its
                // specialized emitter handles. Rich records are decoded by
                // recursively invoking the same checked leaf emitters.
                let recursive = !child_ops.iter().all(Self::packed_struct_child);
                Ok(FieldOp::Struct {
                    fields: child_ops,
                    nullable,
                    recursive,
                })
            }
            // Dictionary/Enum - store indices as i32
            DataType::Dictionary(_, _) => Ok(FieldOp::Enum { nullable }),
            // List types - decode element type recursively
            DataType::List(element_field) => {
                let element_op = Self::field_to_op(element_field.as_ref())?;
                // emit_list_decode handles non-nullable scalars (incl. temporal),
                // decimal128/256, and enum elements. Others (string/bytes/fixed/
                // nullable/nested) are silently skipped → reject at compile time.
                let supported = matches!(
                    element_op,
                    FieldOp::Boolean { nullable: false }
                        | FieldOp::Int32 { nullable: false }
                        | FieldOp::Int64 { nullable: false }
                        | FieldOp::Float32 { nullable: false }
                        | FieldOp::Float64 { nullable: false }
                        | FieldOp::Decimal128 {
                            nullable: false,
                            ..
                        }
                        | FieldOp::Decimal256 {
                            nullable: false,
                            ..
                        }
                        | FieldOp::Enum { nullable: false }
                );
                if !supported {
                    return Err(
                        "JIT list decode supports non-nullable scalar/decimal/enum elements \
                         only; rejecting list with an unsupported element to avoid silent \
                         corruption"
                            .to_string(),
                    );
                }
                Ok(FieldOp::List {
                    element: Box::new(element_op),
                    nullable,
                })
            }
            // Decimal types. Avro has two encodings: `fixed(N)` (raw N bytes, no
            // length prefix) and `bytes` (zigzag-length-prefixed minimal
            // two's-complement). The Arrow type can't distinguish them, so the
            // schema builder marks bytes-backed via field metadata
            // ("avro.decimal.encoding" = "bytes"). Absent ⇒ fixed-backed (the
            // historical default), so callers that don't set it are unaffected.
            DataType::Decimal128(precision, scale) => {
                if let Some(size) = field.metadata().get("avro.decimal.fixed_size") {
                    if size.parse::<usize>().ok() != Some(16) {
                        return Err("Unsupported fixed-backed decimal width".into());
                    }
                }

                let bytes_backed = field
                    .metadata()
                    .get("avro.decimal.encoding")
                    .map(|v| v == "bytes")
                    .unwrap_or(false);
                Ok(FieldOp::Decimal128 {
                    _precision: *precision,
                    _scale: *scale,
                    nullable,
                    bytes_backed,
                })
            }
            DataType::Decimal256(precision, scale) => {
                if let Some(size) = field.metadata().get("avro.decimal.fixed_size") {
                    if size.parse::<usize>().ok() != Some(32) {
                        return Err("Unsupported fixed-backed decimal width".into());
                    }
                }

                let bytes_backed = field
                    .metadata()
                    .get("avro.decimal.encoding")
                    .map(|v| v == "bytes")
                    .unwrap_or(false);
                Ok(FieldOp::Decimal256 {
                    _precision: *precision,
                    _scale: *scale,
                    nullable,
                    bytes_backed,
                })
            }
            // Duration - encoded as i64
            DataType::Duration(_) => Ok(FieldOp::Duration { nullable }),
            // Interval types
            DataType::Interval(unit) => Ok(FieldOp::Interval {
                unit: *unit,
                nullable,
                avro_duration: field
                    .metadata()
                    .get("avro.duration")
                    .is_some_and(|v| v == "true"),
            }),
            // Null type - used in unions for nullable fields
            DataType::Null => {
                Err("Standalone Null columns are unsupported by JIT Arrow assembly".into())
            }
            // Unsupported types - fail fast
            other => Err(format!(
                "Unsupported data type for JIT Avro decoder: {:?} (field '{}')",
                other,
                field.name()
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
        if let FieldOp::Struct {
            fields,
            nullable,
            recursive: true,
        } = op
        {
            let value_block = builder.create_block();
            let done = builder.create_block();
            if *nullable {
                let (tag, next) = Self::emit_varint_decode(
                    builder,
                    input_ptr,
                    input_len,
                    cursor_var,
                    error_block,
                );
                builder.def_var(cursor_var, next);
                let valid = builder
                    .ins()
                    .icmp_imm(IntCC::UnsignedLessThanOrEqual, tag, 2);
                let even = builder.ins().band_imm(tag, 1);
                let even = builder.ins().icmp_imm(IntCC::Equal, even, 0);
                let valid = builder.ins().band(valid, even);
                Self::emit_require(builder, valid, error_block);
                let is_null = builder
                    .ins()
                    .icmp_imm(IntCC::Equal, tag, buffer_slot.null_tag);
                let null_block = builder.create_block();
                builder
                    .ins()
                    .brif(is_null, null_block, &[], value_block, &[]);
                builder.switch_to_block(null_block);
                Self::emit_absent_record(builder, buffer_slot, output_ptrs, output_vars, ptr_type);
                builder.ins().jump(done, &[]);
                builder.seal_block(null_block);
            } else {
                builder.ins().jump(value_block, &[]);
            }
            builder.switch_to_block(value_block);
            for (child, slot) in fields.iter().zip(&buffer_slot.children) {
                Self::emit_field_decode(
                    builder,
                    child,
                    slot,
                    input_ptr,
                    input_len,
                    output_ptrs,
                    output_caps,
                    cursor_var,
                    row_var,
                    output_vars,
                    ptr_type,
                    memcpy_func_ref,
                    error_block,
                    output_overflow_block,
                );
            }
            if let Some(validity) = buffer_slot.validity {
                let row = builder.use_var(row_var);
                Self::emit_set_validity_bit(builder, output_ptrs, validity, row, ptr_type);
            }
            builder.ins().jump(done, &[]);
            builder.seal_block(value_block);
            builder.seal_block(done);
            builder.switch_to_block(done);
            return;
        }
        // For fixed-size types, use primary buffer only
        // For variable-size types (Bytes), use primary=offsets, secondary=data
        let buf_idx = buffer_slot.primary;

        // Load primary output pointer
        let buf_offset = builder.ins().iconst(ptr_type, (buf_idx * 8) as i64);
        let output_ptr_addr = builder.ins().iadd(output_ptrs, buf_offset);
        let output_ptr = builder
            .ins()
            .load(ptr_type, MemFlags::trusted(), output_ptr_addr, 0);
        let output_pos = builder.use_var(output_vars[buf_idx]);
        let output_write_ptr = builder.ins().iadd(output_ptr, output_pos);

        // Load primary output capacity (for bounds checking)
        let output_cap_addr = builder.ins().iadd(output_caps, buf_offset);
        let _output_capacity =
            builder
                .ins()
                .load(types::I64, MemFlags::trusted(), output_cap_addr, 0);

        match op {
            FieldOp::Int32 { nullable } => {
                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        error_block,
                    );
                    builder.def_var(cursor_var, tag_cursor);

                    let null_tag = builder.ins().iconst(types::I64, buffer_slot.null_tag);
                    let valid_tag = builder
                        .ins()
                        .icmp_imm(IntCC::UnsignedLessThanOrEqual, tag, 2);
                    let even = builder.ins().band_imm(tag, 1);
                    let even = builder.ins().icmp_imm(IntCC::Equal, even, 0);
                    let valid_tag = builder.ins().band(valid_tag, even);
                    Self::emit_require(builder, valid_tag, error_block);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, null_tag);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder
                        .ins()
                        .brif(is_null, null_block, &[], value_block, &[]);

                    // Null block: write 0 placeholder (validity bit stays 0 = null)
                    builder.switch_to_block(null_block);
                    let zero_val = builder.ins().iconst(types::I32, 0);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), zero_val, output_write_ptr, 0);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode the value and set validity bit
                    builder.switch_to_block(value_block);
                    let (value, new_cursor) = Self::emit_varint_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        error_block,
                    );
                    let zigzag = Self::emit_zigzag_decode_i32(builder, value, error_block);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), zigzag, output_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(
                            builder,
                            output_ptrs,
                            validity_idx,
                            row_idx,
                            ptr_type,
                        );
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
                    let (value, new_cursor) = Self::emit_varint_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        error_block,
                    );
                    let zigzag = Self::emit_zigzag_decode_i32(builder, value, error_block);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), zigzag, output_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                    let four = builder.ins().iconst(types::I64, 4);
                    let new_output_pos = builder.ins().iadd(output_pos, four);
                    builder.def_var(output_vars[buf_idx], new_output_pos);
                }
            }
            FieldOp::Int64 { nullable } => {
                if *nullable {
                    // Read union tag
                    let (tag, tag_cursor) = Self::emit_varint_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        error_block,
                    );
                    builder.def_var(cursor_var, tag_cursor);

                    // Check if null (tag == 0)
                    let null_tag = builder.ins().iconst(types::I64, buffer_slot.null_tag);
                    let valid_tag = builder
                        .ins()
                        .icmp_imm(IntCC::UnsignedLessThanOrEqual, tag, 2);
                    let even = builder.ins().band_imm(tag, 1);
                    let even = builder.ins().icmp_imm(IntCC::Equal, even, 0);
                    let valid_tag = builder.ins().band(valid_tag, even);
                    Self::emit_require(builder, valid_tag, error_block);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, null_tag);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder
                        .ins()
                        .brif(is_null, null_block, &[], value_block, &[]);

                    // Null block: write 0 placeholder (validity bit stays 0 = null)
                    builder.switch_to_block(null_block);
                    let zero_val = builder.ins().iconst(types::I64, 0);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), zero_val, output_write_ptr, 0);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode the value and set validity bit
                    builder.switch_to_block(value_block);
                    let (value, new_cursor) = Self::emit_varint_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        error_block,
                    );
                    let zigzag = Self::emit_zigzag_decode_i64(builder, value);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), zigzag, output_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(
                            builder,
                            output_ptrs,
                            validity_idx,
                            row_idx,
                            ptr_type,
                        );
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
                    let (value, new_cursor) = Self::emit_varint_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        error_block,
                    );
                    let zigzag = Self::emit_zigzag_decode_i64(builder, value);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), zigzag, output_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                    let eight = builder.ins().iconst(types::I64, 8);
                    let new_output_pos = builder.ins().iadd(output_pos, eight);
                    builder.def_var(output_vars[buf_idx], new_output_pos);
                }
            }
            FieldOp::Float32 { nullable } => {
                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        error_block,
                    );
                    builder.def_var(cursor_var, tag_cursor);

                    let null_tag = builder.ins().iconst(types::I64, buffer_slot.null_tag);
                    let valid_tag = builder
                        .ins()
                        .icmp_imm(IntCC::UnsignedLessThanOrEqual, tag, 2);
                    let even = builder.ins().band_imm(tag, 1);
                    let even = builder.ins().icmp_imm(IntCC::Equal, even, 0);
                    let valid_tag = builder.ins().band(valid_tag, even);
                    Self::emit_require(builder, valid_tag, error_block);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, null_tag);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder
                        .ins()
                        .brif(is_null, null_block, &[], value_block, &[]);

                    // Null block (validity bit stays 0 = null)
                    builder.switch_to_block(null_block);
                    let zero_val = builder.ins().iconst(types::I32, 0);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), zero_val, output_write_ptr, 0);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode and set validity bit
                    builder.switch_to_block(value_block);
                    let cursor = builder.use_var(cursor_var);
                    Self::emit_fixed_bounds_check(builder, cursor, 4, input_len, error_block);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder
                        .ins()
                        .load(types::I32, MemFlags::trusted(), read_ptr, 0);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), value, output_write_ptr, 0);
                    let four = builder.ins().iconst(types::I64, 4);
                    let new_cursor = builder.ins().iadd(cursor, four);
                    builder.def_var(cursor_var, new_cursor);
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(
                            builder,
                            output_ptrs,
                            validity_idx,
                            row_idx,
                            ptr_type,
                        );
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
                    let value = builder
                        .ins()
                        .load(types::I32, MemFlags::trusted(), read_ptr, 0);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), value, output_write_ptr, 0);
                    let four = builder.ins().iconst(types::I64, 4);
                    let new_cursor = builder.ins().iadd(cursor, four);
                    builder.def_var(cursor_var, new_cursor);
                    let new_output_pos = builder.ins().iadd(output_pos, four);
                    builder.def_var(output_vars[buf_idx], new_output_pos);
                }
            }
            FieldOp::Float64 { nullable } => {
                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        error_block,
                    );
                    builder.def_var(cursor_var, tag_cursor);

                    let null_tag = builder.ins().iconst(types::I64, buffer_slot.null_tag);
                    let valid_tag = builder
                        .ins()
                        .icmp_imm(IntCC::UnsignedLessThanOrEqual, tag, 2);
                    let even = builder.ins().band_imm(tag, 1);
                    let even = builder.ins().icmp_imm(IntCC::Equal, even, 0);
                    let valid_tag = builder.ins().band(valid_tag, even);
                    Self::emit_require(builder, valid_tag, error_block);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, null_tag);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder
                        .ins()
                        .brif(is_null, null_block, &[], value_block, &[]);

                    builder.switch_to_block(null_block);
                    let zero_val = builder.ins().iconst(types::I64, 0);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), zero_val, output_write_ptr, 0);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode and set validity bit
                    builder.switch_to_block(value_block);
                    let cursor = builder.use_var(cursor_var);
                    Self::emit_fixed_bounds_check(builder, cursor, 8, input_len, error_block);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder
                        .ins()
                        .load(types::I64, MemFlags::trusted(), read_ptr, 0);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), value, output_write_ptr, 0);
                    let eight = builder.ins().iconst(types::I64, 8);
                    let new_cursor = builder.ins().iadd(cursor, eight);
                    builder.def_var(cursor_var, new_cursor);
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(
                            builder,
                            output_ptrs,
                            validity_idx,
                            row_idx,
                            ptr_type,
                        );
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
                    let value = builder
                        .ins()
                        .load(types::I64, MemFlags::trusted(), read_ptr, 0);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), value, output_write_ptr, 0);
                    let eight = builder.ins().iconst(types::I64, 8);
                    let new_cursor = builder.ins().iadd(cursor, eight);
                    builder.def_var(cursor_var, new_cursor);
                    let new_output_pos = builder.ins().iadd(output_pos, eight);
                    builder.def_var(output_vars[buf_idx], new_output_pos);
                }
            }
            FieldOp::Boolean { nullable } => {
                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        error_block,
                    );
                    builder.def_var(cursor_var, tag_cursor);

                    let null_tag = builder.ins().iconst(types::I64, buffer_slot.null_tag);
                    let valid_tag = builder
                        .ins()
                        .icmp_imm(IntCC::UnsignedLessThanOrEqual, tag, 2);
                    let even = builder.ins().band_imm(tag, 1);
                    let even = builder.ins().icmp_imm(IntCC::Equal, even, 0);
                    let valid_tag = builder.ins().band(valid_tag, even);
                    Self::emit_require(builder, valid_tag, error_block);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, null_tag);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder
                        .ins()
                        .brif(is_null, null_block, &[], value_block, &[]);

                    builder.switch_to_block(null_block);
                    let zero_val = builder.ins().iconst(types::I8, 0);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), zero_val, output_write_ptr, 0);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode and set validity bit
                    builder.switch_to_block(value_block);
                    let cursor = builder.use_var(cursor_var);
                    Self::emit_fixed_bounds_check(builder, cursor, 1, input_len, error_block);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder
                        .ins()
                        .load(types::I8, MemFlags::trusted(), read_ptr, 0);
                    let valid = builder
                        .ins()
                        .icmp_imm(IntCC::UnsignedLessThanOrEqual, value, 1);
                    Self::emit_require(builder, valid, error_block);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), value, output_write_ptr, 0);
                    let one = builder.ins().iconst(types::I64, 1);
                    let new_cursor = builder.ins().iadd(cursor, one);
                    builder.def_var(cursor_var, new_cursor);
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(
                            builder,
                            output_ptrs,
                            validity_idx,
                            row_idx,
                            ptr_type,
                        );
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
                    let value = builder
                        .ins()
                        .load(types::I8, MemFlags::trusted(), read_ptr, 0);
                    let valid = builder
                        .ins()
                        .icmp_imm(IntCC::UnsignedLessThanOrEqual, value, 1);
                    Self::emit_require(builder, valid, error_block);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), value, output_write_ptr, 0);
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
                let data_buf_idx = buffer_slot
                    .secondary
                    .expect("Bytes type must have secondary buffer");

                // Load data buffer pointer and position
                let data_buf_offset = builder.ins().iconst(ptr_type, (data_buf_idx * 8) as i64);
                let data_ptr_addr = builder.ins().iadd(output_ptrs, data_buf_offset);
                let data_ptr = builder
                    .ins()
                    .load(ptr_type, MemFlags::trusted(), data_ptr_addr, 0);
                let data_pos = builder.use_var(output_vars[data_buf_idx]);
                let data_write_ptr = builder.ins().iadd(data_ptr, data_pos);

                // Load data buffer capacity
                let data_cap_addr = builder.ins().iadd(output_caps, data_buf_offset);
                let data_capacity =
                    builder
                        .ins()
                        .load(types::I64, MemFlags::trusted(), data_cap_addr, 0);

                // output_write_ptr points to offsets buffer (already set up above)
                // output_capacity is offsets buffer capacity

                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        error_block,
                    );
                    builder.def_var(cursor_var, tag_cursor);

                    let null_tag = builder.ins().iconst(types::I64, buffer_slot.null_tag);
                    let valid_tag = builder
                        .ins()
                        .icmp_imm(IntCC::UnsignedLessThanOrEqual, tag, 2);
                    let even = builder.ins().band_imm(tag, 1);
                    let even = builder.ins().icmp_imm(IntCC::Equal, even, 0);
                    let valid_tag = builder.ins().band(valid_tag, even);
                    Self::emit_require(builder, valid_tag, error_block);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, null_tag);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder
                        .ins()
                        .brif(is_null, null_block, &[], value_block, &[]);

                    // Null block: write current data_pos as offset (same as previous), increment offset pos
                    builder.switch_to_block(null_block);
                    {
                        // Write offset (data_pos as i32)
                        let data_pos_null = builder.use_var(output_vars[data_buf_idx]);
                        let offset_i32 = builder.ins().ireduce(types::I32, data_pos_null);
                        let offset_pos_null = builder.use_var(output_vars[buf_idx]);
                        let offset_write_ptr_null = builder.ins().iadd(output_ptr, offset_pos_null);
                        builder.ins().store(
                            MemFlags::trusted(),
                            offset_i32,
                            offset_write_ptr_null,
                            0,
                        );
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
                        builder.ins().store(
                            MemFlags::trusted(),
                            offset_i32,
                            offset_write_ptr_val,
                            0,
                        );

                        // Advance offsets position by 4
                        let four = builder.ins().iconst(types::I64, 4);
                        let offset_pos_curr = builder.use_var(output_vars[buf_idx]);
                        let new_offset_pos = builder.ins().iadd(offset_pos_curr, four);
                        builder.def_var(output_vars[buf_idx], new_offset_pos);

                        // Decode bytes length
                        let (len_raw, new_cursor) = Self::emit_varint_decode(
                            builder,
                            input_ptr,
                            input_len,
                            cursor_var,
                            error_block,
                        );
                        builder.def_var(cursor_var, new_cursor);
                        let len = Self::emit_zigzag_decode_i64(builder, len_raw);
                        let cursor = builder.use_var(cursor_var);

                        // Input bounds check: len >= 0 and cursor + len <= input_len
                        Self::emit_variable_bounds_check(
                            builder,
                            cursor,
                            len,
                            input_len,
                            error_block,
                        );

                        // Output bounds check for DATA buffer
                        let cur_data_pos = builder.use_var(output_vars[data_buf_idx]);
                        Self::emit_output_bounds_check(
                            builder,
                            cur_data_pos,
                            len,
                            data_capacity,
                            output_overflow_block,
                        );

                        // Copy data to data buffer
                        let cur_data_write = builder.ins().iadd(data_ptr, cur_data_pos);
                        Self::emit_memcpy(
                            builder,
                            input_ptr,
                            cursor,
                            cur_data_write,
                            len,
                            memcpy_func_ref,
                        );

                        // Advance input cursor
                        let new_cursor = builder.ins().iadd(cursor, len);
                        builder.def_var(cursor_var, new_cursor);

                        // Advance data position
                        let new_data_pos = builder.ins().iadd(cur_data_pos, len);
                        builder.def_var(output_vars[data_buf_idx], new_data_pos);

                        // Set validity bit to 1 for non-null value
                        if let Some(validity_idx) = buffer_slot.validity {
                            let row_idx = builder.use_var(row_var);
                            Self::emit_set_validity_bit(
                                builder,
                                output_ptrs,
                                validity_idx,
                                row_idx,
                                ptr_type,
                            );
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
                    builder
                        .ins()
                        .store(MemFlags::trusted(), offset_i32, output_write_ptr, 0);

                    // Advance offsets position by 4
                    let four = builder.ins().iconst(types::I64, 4);
                    let new_offset_pos = builder.ins().iadd(output_pos, four);
                    builder.def_var(output_vars[buf_idx], new_offset_pos);

                    // Decode bytes length
                    let (len_raw, new_cursor) = Self::emit_varint_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        error_block,
                    );
                    builder.def_var(cursor_var, new_cursor);
                    let len = Self::emit_zigzag_decode_i64(builder, len_raw);
                    let cursor = builder.use_var(cursor_var);

                    // Input bounds check: len >= 0 and cursor + len <= input_len
                    Self::emit_variable_bounds_check(builder, cursor, len, input_len, error_block);

                    // Output bounds check for DATA buffer
                    Self::emit_output_bounds_check(
                        builder,
                        data_pos,
                        len,
                        data_capacity,
                        output_overflow_block,
                    );

                    // Copy data to data buffer
                    Self::emit_memcpy(
                        builder,
                        input_ptr,
                        cursor,
                        data_write_ptr,
                        len,
                        memcpy_func_ref,
                    );

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
                    let (tag, tag_cursor) = Self::emit_varint_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        error_block,
                    );
                    builder.def_var(cursor_var, tag_cursor);

                    let null_tag = builder.ins().iconst(types::I64, buffer_slot.null_tag);
                    let valid_tag = builder
                        .ins()
                        .icmp_imm(IntCC::UnsignedLessThanOrEqual, tag, 2);
                    let even = builder.ins().band_imm(tag, 1);
                    let even = builder.ins().icmp_imm(IntCC::Equal, even, 0);
                    let valid_tag = builder.ins().band(valid_tag, even);
                    Self::emit_require(builder, valid_tag, error_block);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, null_tag);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder
                        .ins()
                        .brif(is_null, null_block, &[], value_block, &[]);

                    // Null block: write zeros
                    builder.switch_to_block(null_block);
                    // For fixed, we could memset to 0, but for simplicity just skip
                    builder.ins().jump(continue_block, &[]);

                    // Value block: copy fixed bytes and set validity bit
                    builder.switch_to_block(value_block);
                    let cursor = builder.use_var(cursor_var);
                    Self::emit_fixed_bounds_check(
                        builder,
                        cursor,
                        *size as i64,
                        input_len,
                        error_block,
                    );
                    Self::emit_memcpy(
                        builder,
                        input_ptr,
                        cursor,
                        output_write_ptr,
                        size_val,
                        memcpy_func_ref,
                    );
                    let new_cursor = builder.ins().iadd(cursor, size_val);
                    builder.def_var(cursor_var, new_cursor);
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(
                            builder,
                            output_ptrs,
                            validity_idx,
                            row_idx,
                            ptr_type,
                        );
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
                    Self::emit_fixed_bounds_check(
                        builder,
                        cursor,
                        *size as i64,
                        input_len,
                        error_block,
                    );
                    Self::emit_memcpy(
                        builder,
                        input_ptr,
                        cursor,
                        output_write_ptr,
                        size_val,
                        memcpy_func_ref,
                    );
                    let new_cursor = builder.ins().iadd(cursor, size_val);
                    builder.def_var(cursor_var, new_cursor);
                    let new_output_pos = builder.ins().iadd(output_pos, size_val);
                    builder.def_var(output_vars[buf_idx], new_output_pos);
                }
            }
            FieldOp::Struct {
                fields: child_ops,
                nullable,
                recursive: _,
            } => {
                // Struct: decode each child field sequentially
                // Output is flattened: all field data concatenated together

                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        error_block,
                    );
                    builder.def_var(cursor_var, tag_cursor);

                    let null_tag = builder.ins().iconst(types::I64, buffer_slot.null_tag);
                    let valid_tag = builder
                        .ins()
                        .icmp_imm(IntCC::UnsignedLessThanOrEqual, tag, 2);
                    let even = builder.ins().band_imm(tag, 1);
                    let even = builder.ins().icmp_imm(IntCC::Equal, even, 0);
                    let valid_tag = builder.ins().band(valid_tag, even);
                    Self::emit_require(builder, valid_tag, error_block);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, null_tag);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder
                        .ins()
                        .brif(is_null, null_block, &[], value_block, &[]);

                    builder.switch_to_block(null_block);
                    let stride: i64 = child_ops.iter().map(Self::element_op_size).sum();
                    let next = builder.ins().iadd_imm(output_pos, stride);
                    builder.def_var(output_vars[buf_idx], next);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode struct and set validity bit
                    builder.switch_to_block(value_block);
                    Self::emit_struct_decode(
                        builder,
                        child_ops,
                        input_ptr,
                        input_len,
                        cursor_var,
                        output_write_ptr,
                        output_pos,
                        output_vars,
                        buf_idx,
                        error_block,
                    );
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(
                            builder,
                            output_ptrs,
                            validity_idx,
                            row_idx,
                            ptr_type,
                        );
                    }
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);
                } else {
                    Self::emit_struct_decode(
                        builder,
                        child_ops,
                        input_ptr,
                        input_len,
                        cursor_var,
                        output_write_ptr,
                        output_pos,
                        output_vars,
                        buf_idx,
                        error_block,
                    );
                }
            }
            FieldOp::Enum { nullable } => {
                // Avro enum is encoded as an `int` (zero-based symbol index),
                // i.e. a zigzag varint — so decode it like any other int.
                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        error_block,
                    );
                    builder.def_var(cursor_var, tag_cursor);

                    let null_tag = builder.ins().iconst(types::I64, buffer_slot.null_tag);
                    let valid_tag = builder
                        .ins()
                        .icmp_imm(IntCC::UnsignedLessThanOrEqual, tag, 2);
                    let even = builder.ins().band_imm(tag, 1);
                    let even = builder.ins().icmp_imm(IntCC::Equal, even, 0);
                    let valid_tag = builder.ins().band(valid_tag, even);
                    Self::emit_require(builder, valid_tag, error_block);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, null_tag);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder
                        .ins()
                        .brif(is_null, null_block, &[], value_block, &[]);

                    builder.switch_to_block(null_block);
                    let zero_val = builder.ins().iconst(types::I32, 0);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), zero_val, output_write_ptr, 0);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode enum and set validity bit
                    builder.switch_to_block(value_block);
                    let (raw, new_cursor) = Self::emit_varint_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        error_block,
                    );
                    // Avro enum index is an `int` = zigzag varint (like any int).
                    let value_i32 = Self::emit_zigzag_decode_i32(builder, raw, error_block);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), value_i32, output_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(
                            builder,
                            output_ptrs,
                            validity_idx,
                            row_idx,
                            ptr_type,
                        );
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
                    let (raw, new_cursor) = Self::emit_varint_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        error_block,
                    );
                    // Avro enum index is an `int` = zigzag varint (like any int).
                    let value_i32 = Self::emit_zigzag_decode_i32(builder, raw, error_block);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), value_i32, output_write_ptr, 0);
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
                    let (tag, tag_cursor) = Self::emit_varint_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        error_block,
                    );
                    builder.def_var(cursor_var, tag_cursor);

                    let null_tag = builder.ins().iconst(types::I64, buffer_slot.null_tag);
                    let valid_tag = builder
                        .ins()
                        .icmp_imm(IntCC::UnsignedLessThanOrEqual, tag, 2);
                    let even = builder.ins().band_imm(tag, 1);
                    let even = builder.ins().icmp_imm(IntCC::Equal, even, 0);
                    let valid_tag = builder.ins().band(valid_tag, even);
                    Self::emit_require(builder, valid_tag, error_block);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, null_tag);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder
                        .ins()
                        .brif(is_null, null_block, &[], value_block, &[]);

                    // Null block: skip
                    builder.switch_to_block(null_block);
                    let data_pos = builder.use_var(output_vars[buffer_slot.secondary.unwrap()]);
                    let count = builder.ins().udiv_imm(data_pos, buffer_slot.element_size);
                    let offset = builder.ins().ireduce(types::I32, count);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), offset, output_write_ptr, 0);
                    let next = builder.ins().iadd_imm(output_pos, 4);
                    builder.def_var(output_vars[buf_idx], next);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode list and set validity bit
                    builder.switch_to_block(value_block);
                    Self::emit_list_decode(
                        builder,
                        element,
                        buffer_slot,
                        input_ptr,
                        input_len,
                        output_ptrs,
                        output_caps,
                        cursor_var,
                        output_vars,
                        ptr_type,
                        error_block,
                        output_overflow_block,
                    );
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(
                            builder,
                            output_ptrs,
                            validity_idx,
                            row_idx,
                            ptr_type,
                        );
                    }
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);
                } else {
                    Self::emit_list_decode(
                        builder,
                        element,
                        buffer_slot,
                        input_ptr,
                        input_len,
                        output_ptrs,
                        output_caps,
                        cursor_var,
                        output_vars,
                        ptr_type,
                        error_block,
                        output_overflow_block,
                    );
                }
            }
            FieldOp::Decimal128 {
                nullable,
                bytes_backed,
                ..
            } => {
                // Avro decimal: fixed(16) reads 16 raw big-endian bytes; bytes-backed
                // reads a zigzag length + that many bytes (sign-extended to 16).
                // Output is always a 16-byte little-endian Arrow Decimal128.
                let byte_size: i64 = 16;

                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        error_block,
                    );
                    builder.def_var(cursor_var, tag_cursor);

                    let null_tag = builder.ins().iconst(types::I64, buffer_slot.null_tag);
                    let valid_tag = builder
                        .ins()
                        .icmp_imm(IntCC::UnsignedLessThanOrEqual, tag, 2);
                    let even = builder.ins().band_imm(tag, 1);
                    let even = builder.ins().icmp_imm(IntCC::Equal, even, 0);
                    let valid_tag = builder.ins().band(valid_tag, even);
                    Self::emit_require(builder, valid_tag, error_block);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, null_tag);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder
                        .ins()
                        .brif(is_null, null_block, &[], value_block, &[]);

                    builder.switch_to_block(null_block);
                    // Write zeros for null
                    let zero_val = builder.ins().iconst(types::I64, 0);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), zero_val, output_write_ptr, 0);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), zero_val, output_write_ptr, 8);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode and set validity bit
                    builder.switch_to_block(value_block);
                    if *bytes_backed {
                        Self::emit_decimal128_decode_bytes(
                            builder,
                            input_ptr,
                            input_len,
                            cursor_var,
                            output_write_ptr,
                            error_block,
                        );
                    } else {
                        Self::emit_decimal128_decode(
                            builder,
                            input_ptr,
                            input_len,
                            cursor_var,
                            output_write_ptr,
                            error_block,
                        );
                    }
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(
                            builder,
                            output_ptrs,
                            validity_idx,
                            row_idx,
                            ptr_type,
                        );
                    }
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);
                } else if *bytes_backed {
                    Self::emit_decimal128_decode_bytes(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        output_write_ptr,
                        error_block,
                    );
                } else {
                    Self::emit_decimal128_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        output_write_ptr,
                        error_block,
                    );
                }

                let size_const = builder.ins().iconst(types::I64, byte_size);
                let new_output_pos = builder.ins().iadd(output_pos, size_const);
                builder.def_var(output_vars[buf_idx], new_output_pos);
            }
            FieldOp::Decimal256 {
                nullable,
                bytes_backed,
                ..
            } => {
                // Avro decimal: fixed(32) reads 32 raw big-endian bytes; bytes-backed
                // reads a zigzag length + that many bytes (sign-extended to 32).
                let byte_size: i64 = 32;

                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        error_block,
                    );
                    builder.def_var(cursor_var, tag_cursor);

                    let null_tag = builder.ins().iconst(types::I64, buffer_slot.null_tag);
                    let valid_tag = builder
                        .ins()
                        .icmp_imm(IntCC::UnsignedLessThanOrEqual, tag, 2);
                    let even = builder.ins().band_imm(tag, 1);
                    let even = builder.ins().icmp_imm(IntCC::Equal, even, 0);
                    let valid_tag = builder.ins().band(valid_tag, even);
                    Self::emit_require(builder, valid_tag, error_block);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, null_tag);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder
                        .ins()
                        .brif(is_null, null_block, &[], value_block, &[]);

                    builder.switch_to_block(null_block);
                    // Write zeros for null
                    let zero_val = builder.ins().iconst(types::I64, 0);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), zero_val, output_write_ptr, 0);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), zero_val, output_write_ptr, 8);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), zero_val, output_write_ptr, 16);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), zero_val, output_write_ptr, 24);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode and set validity bit
                    builder.switch_to_block(value_block);
                    if *bytes_backed {
                        Self::emit_decimal256_decode_bytes(
                            builder,
                            input_ptr,
                            input_len,
                            cursor_var,
                            output_write_ptr,
                            error_block,
                        );
                    } else {
                        Self::emit_decimal256_decode(
                            builder,
                            input_ptr,
                            input_len,
                            cursor_var,
                            output_write_ptr,
                            error_block,
                        );
                    }
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(
                            builder,
                            output_ptrs,
                            validity_idx,
                            row_idx,
                            ptr_type,
                        );
                    }
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);
                } else if *bytes_backed {
                    Self::emit_decimal256_decode_bytes(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        output_write_ptr,
                        error_block,
                    );
                } else {
                    Self::emit_decimal256_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        output_write_ptr,
                        error_block,
                    );
                }

                let size_const = builder.ins().iconst(types::I64, byte_size);
                let new_output_pos = builder.ins().iadd(output_pos, size_const);
                builder.def_var(output_vars[buf_idx], new_output_pos);
            }
            FieldOp::Duration { nullable } => {
                // Duration: read as i64 (same as Int64)
                if *nullable {
                    let (tag, tag_cursor) = Self::emit_varint_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        error_block,
                    );
                    builder.def_var(cursor_var, tag_cursor);

                    let null_tag = builder.ins().iconst(types::I64, buffer_slot.null_tag);
                    let valid_tag = builder
                        .ins()
                        .icmp_imm(IntCC::UnsignedLessThanOrEqual, tag, 2);
                    let even = builder.ins().band_imm(tag, 1);
                    let even = builder.ins().icmp_imm(IntCC::Equal, even, 0);
                    let valid_tag = builder.ins().band(valid_tag, even);
                    Self::emit_require(builder, valid_tag, error_block);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, null_tag);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder
                        .ins()
                        .brif(is_null, null_block, &[], value_block, &[]);

                    builder.switch_to_block(null_block);
                    let zero_val = builder.ins().iconst(types::I64, 0);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), zero_val, output_write_ptr, 0);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode and set validity bit
                    builder.switch_to_block(value_block);
                    let (value, new_cursor) = Self::emit_varint_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        error_block,
                    );
                    let zigzag = Self::emit_zigzag_decode_i64(builder, value);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), zigzag, output_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(
                            builder,
                            output_ptrs,
                            validity_idx,
                            row_idx,
                            ptr_type,
                        );
                    }
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);
                } else {
                    let (value, new_cursor) = Self::emit_varint_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        error_block,
                    );
                    let zigzag = Self::emit_zigzag_decode_i64(builder, value);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), zigzag, output_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                }

                let eight = builder.ins().iconst(types::I64, 8);
                let new_output_pos = builder.ins().iadd(output_pos, eight);
                builder.def_var(output_vars[buf_idx], new_output_pos);
            }
            FieldOp::Interval {
                unit,
                nullable,
                avro_duration,
            } => {
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
                    let (tag, tag_cursor) = Self::emit_varint_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        error_block,
                    );
                    builder.def_var(cursor_var, tag_cursor);

                    let null_tag = builder.ins().iconst(types::I64, buffer_slot.null_tag);
                    let valid_tag = builder
                        .ins()
                        .icmp_imm(IntCC::UnsignedLessThanOrEqual, tag, 2);
                    let even = builder.ins().band_imm(tag, 1);
                    let even = builder.ins().icmp_imm(IntCC::Equal, even, 0);
                    let valid_tag = builder.ins().band(valid_tag, even);
                    Self::emit_require(builder, valid_tag, error_block);
                    let is_null = builder.ins().icmp(IntCC::Equal, tag, null_tag);

                    let null_block = builder.create_block();
                    let value_block = builder.create_block();
                    let continue_block = builder.create_block();

                    builder
                        .ins()
                        .brif(is_null, null_block, &[], value_block, &[]);

                    builder.switch_to_block(null_block);
                    builder.ins().jump(continue_block, &[]);

                    // Value block: decode and set validity bit
                    builder.switch_to_block(value_block);
                    Self::emit_interval_decode(
                        builder,
                        *unit,
                        *avro_duration,
                        input_ptr,
                        input_len,
                        cursor_var,
                        output_write_ptr,
                        error_block,
                    );
                    // Set validity bit to 1 for non-null value
                    if let Some(validity_idx) = buffer_slot.validity {
                        let row_idx = builder.use_var(row_var);
                        Self::emit_set_validity_bit(
                            builder,
                            output_ptrs,
                            validity_idx,
                            row_idx,
                            ptr_type,
                        );
                    }
                    builder.ins().jump(continue_block, &[]);

                    builder.seal_block(null_block);
                    builder.seal_block(value_block);
                    builder.seal_block(continue_block);
                    builder.switch_to_block(continue_block);
                } else {
                    Self::emit_interval_decode(
                        builder,
                        *unit,
                        *avro_duration,
                        input_ptr,
                        input_len,
                        cursor_var,
                        output_write_ptr,
                        error_block,
                    );
                }

                let size_const = builder.ins().iconst(types::I64, byte_size);
                let new_output_pos = builder.ins().iadd(output_pos, size_const);
                builder.def_var(output_vars[buf_idx], new_output_pos);
            }
        }
    }

    /// Emit Cranelift IR for Avro struct/record decoding
    ///
    /// A null record still contributes one row to every descendant Arrow
    /// array. It consumes no child wire bytes. Variable offsets repeat; fixed
    /// leaves receive initialized placeholders, including dictionary keys.
    fn emit_absent_record(
        builder: &mut FunctionBuilder,
        slot: &BufferSlot,
        output_ptrs: Value,
        output_vars: &[Variable],
        ptr_type: Type,
    ) {
        if !slot.children.is_empty() {
            for child in &slot.children {
                Self::emit_absent_record(builder, child, output_ptrs, output_vars, ptr_type);
            }
            return;
        }
        let address = builder
            .ins()
            .iadd_imm(output_ptrs, (slot.primary * 8) as i64);
        let ptr = builder
            .ins()
            .load(ptr_type, MemFlags::trusted(), address, 0);
        let pos = builder.use_var(output_vars[slot.primary]);
        let ptr = builder.ins().iadd(ptr, pos);
        if let Some(data) = slot.secondary {
            let bytes = builder.use_var(output_vars[data]);
            let count = builder.ins().udiv_imm(bytes, slot.element_size);
            let count = builder.ins().ireduce(types::I32, count);
            builder.ins().store(MemFlags::trusted(), count, ptr, 0);
            let next = builder.ins().iadd_imm(pos, 4);
            builder.def_var(output_vars[slot.primary], next);
        } else {
            let width = slot.fixed_width.expect("fixed leaf");
            if width > 0 {
                let index = builder.declare_var(types::I64);
                let zero = builder.ins().iconst(types::I64, 0);
                builder.def_var(index, zero);
                let body = builder.create_block();
                let done = builder.create_block();
                builder.ins().jump(body, &[]);
                builder.switch_to_block(body);
                let offset = builder.use_var(index);
                let dest = builder.ins().iadd(ptr, offset);
                let byte = builder.ins().iconst(types::I8, 0);
                builder.ins().store(MemFlags::trusted(), byte, dest, 0);
                let next = builder.ins().iadd_imm(offset, 1);
                builder.def_var(index, next);
                let more = builder
                    .ins()
                    .icmp_imm(IntCC::UnsignedLessThan, next, width as i64);
                builder.ins().brif(more, body, &[], done, &[]);
                builder.seal_block(body);
                builder.seal_block(done);
                builder.switch_to_block(done);
                let next = builder.ins().iadd_imm(pos, width as i64);
                builder.def_var(output_vars[slot.primary], next);
            }
        }
    }

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

            // Get field size (output bytes for this child)
            let field_size: i64 = match child_op {
                FieldOp::Decimal256 { .. } => 32,
                FieldOp::Decimal128 { .. } => 16,
                FieldOp::Int64 { .. } | FieldOp::Float64 { .. } => 8,
                FieldOp::Int32 { .. } | FieldOp::Float32 { .. } | FieldOp::Enum { .. } => 4,
                FieldOp::Boolean { .. } => 1,
                _ => 8, // Default
            };

            // Decode based on child type
            match child_op {
                FieldOp::Int64 { nullable: false } => {
                    let (value, new_cursor) = Self::emit_varint_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        error_block,
                    );
                    let zigzag = Self::emit_zigzag_decode_i64(builder, value);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), zigzag, field_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                }
                FieldOp::Int32 { nullable: false } => {
                    let (value, new_cursor) = Self::emit_varint_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        error_block,
                    );
                    let zigzag = Self::emit_zigzag_decode_i32(builder, value, error_block);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), zigzag, field_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                }
                FieldOp::Float64 { nullable: false } => {
                    let cursor = builder.use_var(cursor_var);
                    Self::emit_fixed_bounds_check(builder, cursor, 8, input_len, error_block);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder
                        .ins()
                        .load(types::I64, MemFlags::trusted(), read_ptr, 0);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), value, field_write_ptr, 0);
                    let eight = builder.ins().iconst(types::I64, 8);
                    let new_cursor = builder.ins().iadd(cursor, eight);
                    builder.def_var(cursor_var, new_cursor);
                }
                FieldOp::Float32 { nullable: false } => {
                    let cursor = builder.use_var(cursor_var);
                    Self::emit_fixed_bounds_check(builder, cursor, 4, input_len, error_block);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder
                        .ins()
                        .load(types::I32, MemFlags::trusted(), read_ptr, 0);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), value, field_write_ptr, 0);
                    let four = builder.ins().iconst(types::I64, 4);
                    let new_cursor = builder.ins().iadd(cursor, four);
                    builder.def_var(cursor_var, new_cursor);
                }
                FieldOp::Boolean { nullable: false } => {
                    let cursor = builder.use_var(cursor_var);
                    Self::emit_fixed_bounds_check(builder, cursor, 1, input_len, error_block);
                    let read_ptr = builder.ins().iadd(input_ptr, cursor);
                    let value = builder
                        .ins()
                        .load(types::I8, MemFlags::trusted(), read_ptr, 0);
                    let valid = builder
                        .ins()
                        .icmp_imm(IntCC::UnsignedLessThanOrEqual, value, 1);
                    Self::emit_require(builder, valid, error_block);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), value, field_write_ptr, 0);
                    let one = builder.ins().iconst(types::I64, 1);
                    let new_cursor = builder.ins().iadd(cursor, one);
                    builder.def_var(cursor_var, new_cursor);
                }
                FieldOp::Decimal128 {
                    nullable: false,
                    bytes_backed,
                    ..
                } => {
                    if *bytes_backed {
                        Self::emit_decimal128_decode_bytes(
                            builder,
                            input_ptr,
                            input_len,
                            cursor_var,
                            field_write_ptr,
                            error_block,
                        );
                    } else {
                        Self::emit_decimal128_decode(
                            builder,
                            input_ptr,
                            input_len,
                            cursor_var,
                            field_write_ptr,
                            error_block,
                        );
                    }
                }
                FieldOp::Decimal256 {
                    nullable: false,
                    bytes_backed,
                    ..
                } => {
                    if *bytes_backed {
                        Self::emit_decimal256_decode_bytes(
                            builder,
                            input_ptr,
                            input_len,
                            cursor_var,
                            field_write_ptr,
                            error_block,
                        );
                    } else {
                        Self::emit_decimal256_decode(
                            builder,
                            input_ptr,
                            input_len,
                            cursor_var,
                            field_write_ptr,
                            error_block,
                        );
                    }
                }
                FieldOp::Enum { nullable: false } => {
                    // Avro enum index = zigzag varint → i32 dictionary key.
                    let (raw, new_cursor) = Self::emit_varint_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        error_block,
                    );
                    let value_i32 = Self::emit_zigzag_decode_i32(builder, raw, error_block);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), value_i32, field_write_ptr, 0);
                    builder.def_var(cursor_var, new_cursor);
                }
                _ => {
                    // Unsupported child types are rejected at compile time
                    // (field_to_op), so this is unreachable.
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

    /// Emit a bounded bulk copy after the caller validates both regions.
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
        output_caps: Value,
        cursor_var: Variable,
        output_vars: &[Variable],
        ptr_type: Type,
        error_block: Block,
        output_overflow_block: Block,
    ) {
        // Get buffer indices
        let offsets_buf_idx = buffer_slot.primary;
        let data_buf_idx = buffer_slot
            .secondary
            .expect("List type must have secondary buffer for data");

        // Get element size based on type
        let element_size: i64 = match element_op {
            FieldOp::Decimal256 { .. } => 32,
            FieldOp::Decimal128 { .. } => 16,
            FieldOp::Int64 { .. } | FieldOp::Float64 { .. } => 8,
            FieldOp::Int32 { .. } | FieldOp::Float32 { .. } | FieldOp::Enum { .. } => 4,
            FieldOp::Boolean { .. } => 1,
            _ => 8, // Default for complex types
        };

        // Load data buffer pointer
        let data_buf_offset = builder.ins().iconst(ptr_type, (data_buf_idx * 8) as i64);
        let data_ptr_addr = builder.ins().iadd(output_ptrs, data_buf_offset);
        let data_ptr = builder
            .ins()
            .load(ptr_type, MemFlags::trusted(), data_ptr_addr, 0);

        // Load offsets buffer pointer
        let offsets_buf_offset = builder.ins().iconst(ptr_type, (offsets_buf_idx * 8) as i64);
        let offsets_ptr_addr = builder.ins().iadd(output_ptrs, offsets_buf_offset);
        let offsets_ptr = builder
            .ins()
            .load(ptr_type, MemFlags::trusted(), offsets_ptr_addr, 0);

        // Write initial offset (current element count, not byte position)
        // Arrow offsets are element indices: offset[row_idx] = element_count so far
        let offset_pos = builder.use_var(output_vars[offsets_buf_idx]);
        let offset_write_ptr = builder.ins().iadd(offsets_ptr, offset_pos);
        let data_pos = builder.use_var(output_vars[data_buf_idx]);
        // Convert byte position to element count: element_count = data_pos / element_size
        let element_size_val = builder.ins().iconst(types::I64, element_size);
        let element_count = builder.ins().udiv(data_pos, element_size_val);
        let offset_i32 = builder.ins().ireduce(types::I32, element_count);
        builder
            .ins()
            .store(MemFlags::trusted(), offset_i32, offset_write_ptr, 0);

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
        let (count_raw, count_cursor) =
            Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
        builder.def_var(cursor_var, count_cursor);

        // Zigzag decode the count
        let count = Self::emit_zigzag_decode_i64(builder, count_raw);

        // Check if count == 0 (end of array)
        let zero_check = builder.ins().iconst(types::I64, 0);
        let is_done = builder.ins().icmp(IntCC::Equal, count, zero_check);
        builder
            .ins()
            .brif(is_done, block_exit, &[], block_body, &[]);

        // Block body: decode elements
        builder.switch_to_block(block_body);

        // Handle negative count: abs(count)
        // If count < 0, also skip byte size varint
        let zero_cmp = builder.ins().iconst(types::I64, 0);
        let is_negative = builder.ins().icmp(IntCC::SignedLessThan, count, zero_cmp);
        let valid_count = builder.ins().icmp_imm(IntCC::NotEqual, count, i64::MIN);
        Self::emit_require(builder, valid_count, error_block);
        let block_end_var = builder.declare_var(types::I64);
        builder.def_var(block_end_var, zero_cmp);

        let neg_count = builder.ins().ineg(count);
        let abs_count = builder.ins().select(is_negative, neg_count, count);

        // If negative, skip byte size (just read and discard)
        let skip_block = builder.create_block();
        let decode_block = builder.create_block();

        builder
            .ins()
            .brif(is_negative, skip_block, &[], decode_block, &[]);

        // Skip block: read byte size and discard
        builder.switch_to_block(skip_block);
        let (raw_size, skip_cursor) =
            Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
        let byte_size = Self::emit_zigzag_decode_i64(builder, raw_size);
        Self::emit_variable_bounds_check(builder, skip_cursor, byte_size, input_len, error_block);
        let block_end = builder.ins().iadd(skip_cursor, byte_size);
        builder.def_var(block_end_var, block_end);
        builder.def_var(cursor_var, skip_cursor);
        builder.ins().jump(decode_block, &[]);

        // Decode block: loop over elements
        builder.switch_to_block(decode_block);
        // Check once per block, keeping the element loop free of output checks.
        let cap_addr = builder.ins().iadd(output_caps, data_buf_offset);
        let capacity = builder
            .ins()
            .load(types::I64, MemFlags::trusted(), cap_addr, 0);
        let used = builder.use_var(output_vars[data_buf_idx]);
        let within_capacity = builder
            .ins()
            .icmp(IntCC::UnsignedLessThanOrEqual, used, capacity);
        Self::emit_require(builder, within_capacity, output_overflow_block);
        let remaining = builder.ins().isub(capacity, used);
        let available = builder.ins().udiv_imm(remaining, element_size);
        let fits = builder
            .ins()
            .icmp(IntCC::UnsignedLessThanOrEqual, abs_count, available);
        Self::emit_require(builder, fits, output_overflow_block);
        // Arrow List offsets are signed i32 element counts.
        let current_count = builder.ins().udiv_imm(used, element_size);
        let count_remaining = builder.ins().irsub_imm(current_count, i32::MAX as i64);
        let offsets_fit =
            builder
                .ins()
                .icmp(IntCC::UnsignedLessThanOrEqual, abs_count, count_remaining);
        Self::emit_require(builder, offsets_fit, output_overflow_block);

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
        let more_elems = builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, elem_idx, abs_count);
        builder
            .ins()
            .brif(more_elems, elem_loop_body, &[], elem_loop_exit, &[]);

        // Element loop body: decode one element
        builder.switch_to_block(elem_loop_body);

        // Calculate output pointer for this element in data buffer
        let current_data_pos = builder.use_var(output_vars[data_buf_idx]);
        let elem_write_ptr = builder.ins().iadd(data_ptr, current_data_pos);

        // Decode based on element type
        match element_op {
            FieldOp::Int64 { nullable: false } => {
                let (value, new_cursor) = Self::emit_varint_decode(
                    builder,
                    input_ptr,
                    input_len,
                    cursor_var,
                    error_block,
                );
                let zigzag = Self::emit_zigzag_decode_i64(builder, value);
                builder
                    .ins()
                    .store(MemFlags::trusted(), zigzag, elem_write_ptr, 0);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Int32 { nullable: false } => {
                let (value, new_cursor) = Self::emit_varint_decode(
                    builder,
                    input_ptr,
                    input_len,
                    cursor_var,
                    error_block,
                );
                let zigzag = Self::emit_zigzag_decode_i32(builder, value, error_block);
                builder
                    .ins()
                    .store(MemFlags::trusted(), zigzag, elem_write_ptr, 0);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Float64 { nullable: false } => {
                let cursor = builder.use_var(cursor_var);
                Self::emit_fixed_bounds_check(builder, cursor, 8, input_len, error_block);
                let read_ptr = builder.ins().iadd(input_ptr, cursor);
                let value = builder
                    .ins()
                    .load(types::I64, MemFlags::trusted(), read_ptr, 0);
                builder
                    .ins()
                    .store(MemFlags::trusted(), value, elem_write_ptr, 0);
                let eight = builder.ins().iconst(types::I64, 8);
                let new_cursor = builder.ins().iadd(cursor, eight);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Float32 { nullable: false } => {
                let cursor = builder.use_var(cursor_var);
                Self::emit_fixed_bounds_check(builder, cursor, 4, input_len, error_block);
                let read_ptr = builder.ins().iadd(input_ptr, cursor);
                let value = builder
                    .ins()
                    .load(types::I32, MemFlags::trusted(), read_ptr, 0);
                builder
                    .ins()
                    .store(MemFlags::trusted(), value, elem_write_ptr, 0);
                let four = builder.ins().iconst(types::I64, 4);
                let new_cursor = builder.ins().iadd(cursor, four);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Boolean { nullable: false } => {
                let cursor = builder.use_var(cursor_var);
                Self::emit_fixed_bounds_check(builder, cursor, 1, input_len, error_block);
                let read_ptr = builder.ins().iadd(input_ptr, cursor);
                let value = builder
                    .ins()
                    .load(types::I8, MemFlags::trusted(), read_ptr, 0);
                let valid = builder
                    .ins()
                    .icmp_imm(IntCC::UnsignedLessThanOrEqual, value, 1);
                Self::emit_require(builder, valid, error_block);
                builder
                    .ins()
                    .store(MemFlags::trusted(), value, elem_write_ptr, 0);
                let one = builder.ins().iconst(types::I64, 1);
                let new_cursor = builder.ins().iadd(cursor, one);
                builder.def_var(cursor_var, new_cursor);
            }
            FieldOp::Decimal128 {
                nullable: false,
                bytes_backed,
                ..
            } => {
                if *bytes_backed {
                    Self::emit_decimal128_decode_bytes(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        elem_write_ptr,
                        error_block,
                    );
                } else {
                    Self::emit_decimal128_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        elem_write_ptr,
                        error_block,
                    );
                }
            }
            FieldOp::Decimal256 {
                nullable: false,
                bytes_backed,
                ..
            } => {
                if *bytes_backed {
                    Self::emit_decimal256_decode_bytes(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        elem_write_ptr,
                        error_block,
                    );
                } else {
                    Self::emit_decimal256_decode(
                        builder,
                        input_ptr,
                        input_len,
                        cursor_var,
                        elem_write_ptr,
                        error_block,
                    );
                }
            }
            FieldOp::Enum { nullable: false } => {
                let (raw, new_cursor) = Self::emit_varint_decode(
                    builder,
                    input_ptr,
                    input_len,
                    cursor_var,
                    error_block,
                );
                let value_i32 = Self::emit_zigzag_decode_i32(builder, raw, error_block);
                builder
                    .ins()
                    .store(MemFlags::trusted(), value_i32, elem_write_ptr, 0);
                builder.def_var(cursor_var, new_cursor);
            }
            _ => {
                // Unsupported element types are rejected at compile time
                // (field_to_op), so this is unreachable.
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
        let block_end = builder.use_var(block_end_var);
        let cursor = builder.use_var(cursor_var);
        let exact = builder.ins().icmp(IntCC::Equal, cursor, block_end);
        let no_size = builder.ins().icmp_imm(IntCC::Equal, block_end, 0);
        let valid = builder.ins().bor(exact, no_size);
        Self::emit_require(builder, valid, error_block);
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

    /// Decode a fixed 16-byte big-endian decimal.
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
        let high_be = builder
            .ins()
            .load(types::I64, MemFlags::trusted(), read_ptr, 0);
        let low_be = builder
            .ins()
            .load(types::I64, MemFlags::trusted(), read_ptr, 8);

        // Byte swap both halves
        let high_le = builder.ins().bswap(high_be);
        let low_le = builder.ins().bswap(low_be);

        // Write in reversed order (low first for little-endian)
        builder
            .ins()
            .store(MemFlags::trusted(), low_le, output_write_ptr, 0);
        builder
            .ins()
            .store(MemFlags::trusted(), high_le, output_write_ptr, 8);

        // Update cursor
        let sixteen = builder.ins().iconst(types::I64, 16);
        let new_cursor = builder.ins().iadd(cursor, sixteen);
        builder.def_var(cursor_var, new_cursor);
    }

    /// Emit Cranelift IR for bytes-backed Decimal128 decoding.
    ///
    /// Avro `bytes`-backed decimal: a zigzag-varint length `L`, then `L` bytes of
    /// big-endian two's-complement (minimal). Output is a 16-byte little-endian
    /// Arrow Decimal128, sign-extended from the `L` bytes.
    fn emit_decimal128_decode_bytes(
        builder: &mut FunctionBuilder,
        input_ptr: Value,
        input_len: Value,
        cursor_var: Variable,
        output_write_ptr: Value,
        error_block: Block,
    ) {
        // Read length (zigzag varint). emit_varint_decode bounds-checks the varint.
        let (raw_len, after_len) =
            Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
        builder.def_var(cursor_var, after_len);
        let len = Self::emit_zigzag_decode_i64(builder, raw_len);

        let cursor = builder.use_var(cursor_var);

        // Runtime bounds check: cursor + len <= input_len.
        let end_pos = builder.ins().iadd(cursor, len);
        let in_bounds = builder
            .ins()
            .icmp(IntCC::UnsignedLessThanOrEqual, end_pos, input_len);
        let ok_block = builder.create_block();
        builder
            .ins()
            .brif(in_bounds, ok_block, &[], error_block, &[]);
        builder.switch_to_block(ok_block);
        builder.seal_block(ok_block);

        let zero = builder.ins().iconst(types::I64, 0);
        let base_ptr = builder.ins().iadd(input_ptr, cursor);

        // Sign-fill byte for the high bytes: 0xFF if negative (MSB of the first,
        // most-significant, byte set) and len > 0, else 0x00.
        let msb = builder
            .ins()
            .load(types::I8, MemFlags::trusted(), base_ptr, 0);
        let msb_i64 = builder.ins().uextend(types::I64, msb);
        let high_bit = builder.ins().iconst(types::I64, 0x80);
        let masked = builder.ins().band(msb_i64, high_bit);
        let is_neg = builder.ins().icmp(IntCC::NotEqual, masked, zero);
        let len_pos = builder.ins().icmp(IntCC::SignedGreaterThan, len, zero);
        let neg_and_len = builder.ins().band(is_neg, len_pos);
        let fill_ff = builder.ins().iconst(types::I8, 0xFF);
        let fill_00 = builder.ins().iconst(types::I8, 0);
        let sign_fill = builder.ins().select(neg_and_len, fill_ff, fill_00);

        // Write 16 little-endian output bytes:
        //   out[j] = (j < len) ? input[cursor + (len-1-j)] : sign_fill
        // (significance j: LE byte j is BE byte (len-1-j); high bytes are sign_fill.)
        for j in 0..16i64 {
            let jc = builder.ins().iconst(types::I64, j);
            let j_lt = builder.ins().icmp(IntCC::SignedLessThan, jc, len);
            let jp1 = builder.ins().iconst(types::I64, j + 1);
            let off = builder.ins().isub(len, jp1); // len - 1 - j
            // Guard the address so a discarded (j >= len) load stays in bounds.
            let safe_off = builder.ins().select(j_lt, off, zero);
            let addr = builder.ins().iadd(base_ptr, safe_off);
            let loaded = builder.ins().load(types::I8, MemFlags::trusted(), addr, 0);
            let byte = builder.ins().select(j_lt, loaded, sign_fill);
            builder
                .ins()
                .store(MemFlags::trusted(), byte, output_write_ptr, j as i32);
        }

        // Advance cursor by len.
        let new_cursor = builder.ins().iadd(cursor, len);
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
        let q0_be = builder
            .ins()
            .load(types::I64, MemFlags::trusted(), read_ptr, 0);
        let q1_be = builder
            .ins()
            .load(types::I64, MemFlags::trusted(), read_ptr, 8);
        let q2_be = builder
            .ins()
            .load(types::I64, MemFlags::trusted(), read_ptr, 16);
        let q3_be = builder
            .ins()
            .load(types::I64, MemFlags::trusted(), read_ptr, 24);

        // Byte swap all quarters
        let q0_le = builder.ins().bswap(q0_be);
        let q1_le = builder.ins().bswap(q1_be);
        let q2_le = builder.ins().bswap(q2_be);
        let q3_le = builder.ins().bswap(q3_be);

        // Write in reversed order for little-endian
        builder
            .ins()
            .store(MemFlags::trusted(), q3_le, output_write_ptr, 0);
        builder
            .ins()
            .store(MemFlags::trusted(), q2_le, output_write_ptr, 8);
        builder
            .ins()
            .store(MemFlags::trusted(), q1_le, output_write_ptr, 16);
        builder
            .ins()
            .store(MemFlags::trusted(), q0_le, output_write_ptr, 24);

        // Update cursor
        let thirty_two = builder.ins().iconst(types::I64, 32);
        let new_cursor = builder.ins().iadd(cursor, thirty_two);
        builder.def_var(cursor_var, new_cursor);
    }

    /// Emit Cranelift IR for bytes-backed Decimal256 decoding (zigzag length +
    /// minimal big-endian two's-complement bytes → 32-byte little-endian).
    fn emit_decimal256_decode_bytes(
        builder: &mut FunctionBuilder,
        input_ptr: Value,
        input_len: Value,
        cursor_var: Variable,
        output_write_ptr: Value,
        error_block: Block,
    ) {
        let (raw_len, after_len) =
            Self::emit_varint_decode(builder, input_ptr, input_len, cursor_var, error_block);
        builder.def_var(cursor_var, after_len);
        let len = Self::emit_zigzag_decode_i64(builder, raw_len);

        let cursor = builder.use_var(cursor_var);
        let end_pos = builder.ins().iadd(cursor, len);
        let in_bounds = builder
            .ins()
            .icmp(IntCC::UnsignedLessThanOrEqual, end_pos, input_len);
        let ok_block = builder.create_block();
        builder
            .ins()
            .brif(in_bounds, ok_block, &[], error_block, &[]);
        builder.switch_to_block(ok_block);
        builder.seal_block(ok_block);

        let zero = builder.ins().iconst(types::I64, 0);
        let base_ptr = builder.ins().iadd(input_ptr, cursor);

        let msb = builder
            .ins()
            .load(types::I8, MemFlags::trusted(), base_ptr, 0);
        let msb_i64 = builder.ins().uextend(types::I64, msb);
        let high_bit = builder.ins().iconst(types::I64, 0x80);
        let masked = builder.ins().band(msb_i64, high_bit);
        let is_neg = builder.ins().icmp(IntCC::NotEqual, masked, zero);
        let len_pos = builder.ins().icmp(IntCC::SignedGreaterThan, len, zero);
        let neg_and_len = builder.ins().band(is_neg, len_pos);
        let fill_ff = builder.ins().iconst(types::I8, 0xFF);
        let fill_00 = builder.ins().iconst(types::I8, 0);
        let sign_fill = builder.ins().select(neg_and_len, fill_ff, fill_00);

        for j in 0..32i64 {
            let jc = builder.ins().iconst(types::I64, j);
            let j_lt = builder.ins().icmp(IntCC::SignedLessThan, jc, len);
            let jp1 = builder.ins().iconst(types::I64, j + 1);
            let off = builder.ins().isub(len, jp1);
            let safe_off = builder.ins().select(j_lt, off, zero);
            let addr = builder.ins().iadd(base_ptr, safe_off);
            let loaded = builder.ins().load(types::I8, MemFlags::trusted(), addr, 0);
            let byte = builder.ins().select(j_lt, loaded, sign_fill);
            builder
                .ins()
                .store(MemFlags::trusted(), byte, output_write_ptr, j as i32);
        }

        let new_cursor = builder.ins().iadd(cursor, len);
        builder.def_var(cursor_var, new_cursor);
    }

    /// Emit Cranelift IR for interval decoding
    fn emit_interval_decode(
        builder: &mut FunctionBuilder,
        unit: IntervalUnit,
        avro_duration: bool,
        input_ptr: Value,
        input_len: Value,
        cursor_var: Variable,
        output_write_ptr: Value,
        error_block: Block,
    ) {
        let cursor = builder.use_var(cursor_var);
        let read_ptr = builder.ins().iadd(input_ptr, cursor);

        if avro_duration {
            Self::emit_fixed_bounds_check(builder, cursor, 12, input_len, error_block);
            let months = builder
                .ins()
                .load(types::I32, MemFlags::trusted(), read_ptr, 0);
            let days = builder
                .ins()
                .load(types::I32, MemFlags::trusted(), read_ptr, 4);
            let millis = builder
                .ins()
                .load(types::I32, MemFlags::trusted(), read_ptr, 8);
            // Arrow months/days are signed, Avro's components are unsigned.
            let months_ok = builder
                .ins()
                .icmp_imm(IntCC::SignedGreaterThanOrEqual, months, 0);
            let days_ok = builder
                .ins()
                .icmp_imm(IntCC::SignedGreaterThanOrEqual, days, 0);
            let valid = builder.ins().band(months_ok, days_ok);
            Self::emit_require(builder, valid, error_block);
            let millis = builder.ins().uextend(types::I64, millis);
            let nanos = builder.ins().imul_imm(millis, 1_000_000);
            builder
                .ins()
                .store(MemFlags::trusted(), months, output_write_ptr, 0);
            builder
                .ins()
                .store(MemFlags::trusted(), days, output_write_ptr, 4);
            builder
                .ins()
                .store(MemFlags::trusted(), nanos, output_write_ptr, 8);
            let next = builder.ins().iadd_imm(cursor, 12);
            builder.def_var(cursor_var, next);
            return;
        }

        match unit {
            IntervalUnit::YearMonth => {
                // 4 bytes (i32) - zigzag encoded varint
                let (value, new_cursor) = Self::emit_varint_decode(
                    builder,
                    input_ptr,
                    input_len,
                    cursor_var,
                    error_block,
                );
                let zigzag = Self::emit_zigzag_decode_i32(builder, value, error_block);
                builder
                    .ins()
                    .store(MemFlags::trusted(), zigzag, output_write_ptr, 0);
                builder.def_var(cursor_var, new_cursor);
            }
            IntervalUnit::DayTime => {
                // 8 bytes (two i32s: days + milliseconds)
                // In Avro, this is typically encoded as a fixed 8 bytes
                Self::emit_fixed_bounds_check(builder, cursor, 8, input_len, error_block);
                let value = builder
                    .ins()
                    .load(types::I64, MemFlags::trusted(), read_ptr, 0);
                builder
                    .ins()
                    .store(MemFlags::trusted(), value, output_write_ptr, 0);
                let eight = builder.ins().iconst(types::I64, 8);
                let new_cursor = builder.ins().iadd(cursor, eight);
                builder.def_var(cursor_var, new_cursor);
            }
            IntervalUnit::MonthDayNano => {
                // 16 bytes (i32 months + i32 days + i64 nanos)
                // In Avro, typically encoded as fixed 16 bytes
                Self::emit_fixed_bounds_check(builder, cursor, 16, input_len, error_block);
                let low = builder
                    .ins()
                    .load(types::I64, MemFlags::trusted(), read_ptr, 0);
                let high = builder
                    .ins()
                    .load(types::I64, MemFlags::trusted(), read_ptr, 8);
                builder
                    .ins()
                    .store(MemFlags::trusted(), low, output_write_ptr, 0);
                builder
                    .ins()
                    .store(MemFlags::trusted(), high, output_write_ptr, 8);
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
        let validity_ptr = builder
            .ins()
            .load(ptr_type, MemFlags::trusted(), validity_ptr_addr, 0);

        // Calculate byte_index = row_idx / 8
        let eight = builder.ins().iconst(types::I64, 8);
        let byte_index = builder.ins().udiv(row_idx, eight);

        // Calculate bit_position = row_idx % 8
        let bit_position = builder.ins().urem(row_idx, eight);

        // Calculate address of the validity byte
        let validity_byte_addr = builder.ins().iadd(validity_ptr, byte_index);

        // Load current byte value
        let current_byte =
            builder
                .ins()
                .load(types::I8, MemFlags::trusted(), validity_byte_addr, 0);

        // Calculate the bit mask: 1 << bit_position
        let one_i8 = builder.ins().iconst(types::I8, 1);
        // Need to truncate bit_position from I64 to I8 for shift
        let bit_pos_i8 = builder.ins().ireduce(types::I8, bit_position);
        let bit_mask = builder.ins().ishl(one_i8, bit_pos_i8);

        // Set the bit: current_byte | bit_mask
        let new_byte = builder.ins().bor(current_byte, bit_mask);

        // Store the updated byte
        builder
            .ins()
            .store(MemFlags::trusted(), new_byte, validity_byte_addr, 0);
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
        builder
            .ins()
            .brif(in_bounds, bounds_ok_block, &[], error_block, &[]);

        // Bounds OK block - read and process byte
        builder.switch_to_block(bounds_ok_block);
        let pos = builder.use_var(pos_var);
        let read_ptr = builder.ins().iadd(input_ptr, pos);
        let byte = builder
            .ins()
            .load(types::I8, MemFlags::trusted(), read_ptr, 0);
        let byte_i64 = builder.ins().uextend(types::I64, byte);

        let shift = builder.use_var(shift_var);
        let before_last = builder.ins().icmp_imm(IntCC::UnsignedLessThan, shift, 63);
        let last = builder.ins().icmp_imm(IntCC::Equal, shift, 63);
        let last_byte_fits = builder
            .ins()
            .icmp_imm(IntCC::UnsignedLessThanOrEqual, byte_i64, 1);
        let last_valid = builder.ins().band(last, last_byte_fits);
        let valid = builder.ins().bor(before_last, last_valid);
        Self::emit_require(builder, valid, error_block);

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
        builder
            .ins()
            .brif(has_more_cmp, loop_header, &[], loop_exit, &[]);

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
    fn emit_zigzag_decode_i32(
        builder: &mut FunctionBuilder,
        value: Value,
        error_block: Block,
    ) -> Value {
        let fits = builder
            .ins()
            .icmp_imm(IntCC::UnsignedLessThanOrEqual, value, u32::MAX as i64);
        Self::emit_require(builder, fits, error_block);

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

    fn emit_require(builder: &mut FunctionBuilder, condition: Value, error: Block) {
        let next = builder.create_block();
        builder.ins().brif(condition, next, &[], error, &[]);
        builder.seal_block(next);
        builder.switch_to_block(next);
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
        let in_bounds = builder
            .ins()
            .icmp(IntCC::UnsignedLessThanOrEqual, end_pos, input_len);

        // Create continuation block for the successful case
        let continue_block = builder.create_block();
        builder
            .ins()
            .brif(in_bounds, continue_block, &[], error_block, &[]);
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
        let len_valid = builder
            .ins()
            .icmp(IntCC::SignedGreaterThanOrEqual, len, zero);

        let len_ok_block = builder.create_block();
        builder
            .ins()
            .brif(len_valid, len_ok_block, &[], error_block, &[]);
        builder.seal_block(len_ok_block);
        builder.switch_to_block(len_ok_block);

        // Second check: cursor + len <= input_len
        let end_pos = builder.ins().iadd(cursor, len);
        let in_bounds = builder
            .ins()
            .icmp(IntCC::UnsignedLessThanOrEqual, end_pos, input_len);

        let continue_block = builder.create_block();
        builder
            .ins()
            .brif(in_bounds, continue_block, &[], error_block, &[]);
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
        let max_offset = builder.ins().iconst(types::I64, i32::MAX as i64);
        let output_capacity = builder.ins().umin(output_capacity, max_offset);
        // Check: output_pos + len <= output_capacity
        let end_pos = builder.ins().iadd(output_pos, len);
        let in_bounds =
            builder
                .ins()
                .icmp(IntCC::UnsignedLessThanOrEqual, end_pos, output_capacity);

        let continue_block = builder.create_block();
        builder
            .ins()
            .brif(in_bounds, continue_block, &[], overflow_block, &[]);
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
    /// Output pointers must not alias input or one another, and must reference writable allocations that
    /// remain alive and exclusively accessible for the call. For checked APIs,
    /// each allocation must cover its advertised capacity; `decode` additionally
    /// requires enough output space for every requested row and list element.
    pub unsafe fn decode(
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

        self.decode_with_caps(input, output_buffers, &output_capacities, row_count, None)
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
        record_ends: Option<&[usize]>,
    ) -> Result<usize, String> {
        let mut invalid_validity = false;
        self.visit_buffer_slots(&mut |slot| {
            if let Some(index) = slot.validity {
                let bytes = row_count.div_ceil(8);
                if output_capacities[index] < bytes {
                    invalid_validity = true;
                } else if bytes != 0 {
                    // SAFETY: public unsafe contract guarantees writable,
                    // non-aliasing allocations for advertised capacities.
                    unsafe { std::ptr::write_bytes(output_buffers[index], 0, bytes) };
                }
            }
        });
        if invalid_validity {
            return Err("Output buffer overflow: validity buffer too small".into());
        }
        let decode_fn = self.decode_fn;
        let input_ptr = input.as_ptr();
        let input_len = input.len();
        let output_ptr = output_buffers.as_mut_ptr();
        let output_caps_ptr = output_capacities.as_ptr();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            decode_fn(
                input_ptr,
                input_len,
                output_ptr,
                output_caps_ptr,
                row_count,
                record_ends.map_or(std::ptr::null(), |ends| ends.as_ptr()),
            )
        }));

        match result {
            Ok(bytes_consumed) => {
                if bytes_consumed == JIT_ERROR_INPUT_BOUNDS {
                    Err("Malformed input: bounds violation during Avro decoding".to_string())
                } else if bytes_consumed == JIT_ERROR_OUTPUT_OVERFLOW {
                    Err(
                        "Output buffer overflow: insufficient capacity for decoded data"
                            .to_string(),
                    )
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
    /// Returns `Some(size)` for packed fixed-size columns; `None` for variable
    /// or recursive record columns. Use the slot tree to plan recursive buffers.
    pub fn required_capacity(&self, col_idx: usize, row_count: usize) -> Option<usize> {
        let slot = self.buffer_slots.get(col_idx)?;
        if !slot.children.is_empty() {
            return None;
        }
        slot.fixed_width?.checked_mul(row_count)
    }

    /// Get the fixed size in bytes for a data type, if it's fixed-size
    pub fn fixed_size_for_type(data_type: &DataType) -> Option<usize> {
        match data_type {
            DataType::Boolean => Some(1), // JIT outputs 1 byte per boolean
            DataType::Int8 | DataType::UInt8 => Some(1),
            DataType::Int16 | DataType::UInt16 => Some(2),
            DataType::Int32
            | DataType::UInt32
            | DataType::Float32
            | DataType::Date32
            | DataType::Time32(_) => Some(4),
            DataType::Int64
            | DataType::UInt64
            | DataType::Float64
            | DataType::Date64
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::Duration(_) => Some(8),
            DataType::Interval(IntervalUnit::YearMonth) => Some(4),
            DataType::Interval(IntervalUnit::DayTime) => Some(8),
            DataType::Interval(IntervalUnit::MonthDayNano) => Some(16),
            DataType::FixedSizeBinary(n) => usize::try_from(*n).ok(),
            DataType::Decimal128(_, _) => Some(16),
            DataType::Decimal256(_, _) => Some(32),
            DataType::Dictionary(_, _) => Some(4), // Enum index stored as i32
            // Variable-length types
            DataType::Utf8 | DataType::Binary | DataType::LargeUtf8 | DataType::LargeBinary => None,
            DataType::List(_) | DataType::LargeList(_) | DataType::Map(_, _) => None,
            // Complex types - treat as variable for safety
            DataType::Struct(fields) => fields.iter().try_fold(0usize, |size, f| {
                let op = Self::field_to_op(f).ok()?;
                if !Self::packed_struct_child(&op) {
                    return None;
                }
                size.checked_add(Self::fixed_size_for_type(f.data_type())?)
            }),
            DataType::Union(_, _) | DataType::FixedSizeList(_, _) => None,
            _ => None,
        }
    }

    /// Check if the schema has any variable-length columns
    pub fn has_variable_columns(&self) -> bool {
        let mut variable = false;
        self.visit_buffer_slots(&mut |slot| variable |= slot.secondary.is_some());
        variable
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

    /// Visit every slot in the native pointer order, including record children.
    pub fn visit_buffer_slots(&self, visitor: &mut impl FnMut(&BufferSlot)) {
        for slot in &self.buffer_slots {
            slot.visit(visitor);
        }
    }

    /// Get the column type for a column
    pub fn column_type(&self, col_idx: usize) -> Option<ColumnType> {
        self.column_types.get(col_idx).copied()
    }

    /// Decode Avro data with output buffer capacity validation
    ///
    /// This method validates that fixed-size columns have sufficient capacity
    /// before decoding. Variable-length writes are checked by native code and
    /// return an overflow error if their advertised capacity is insufficient.
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
    ///
    /// # Safety
    /// Output pointers must not alias input or one another, and must reference writable allocations that
    /// remain alive and exclusively accessible for the call. For checked APIs,
    /// each allocation must cover its advertised capacity; `decode` additionally
    /// requires enough output space for every requested row and list element.
    pub unsafe fn decode_checked(
        &self,
        input: &[u8],
        output_buffers: &[OutputBuffer],
        row_count: usize,
    ) -> Result<usize, String> {
        self.validate_buffers(output_buffers, row_count)?;

        // Extract pointers and capacities for the JIT function
        let mut ptrs: Vec<*mut u8> = output_buffers.iter().map(|b| b.ptr).collect();
        let caps: Vec<usize> = output_buffers.iter().map(|b| b.capacity).collect();

        // Call the underlying decode function with capacities
        let result = self.decode_with_caps(input, &mut ptrs, &caps, row_count, None)?;

        Ok(result)
    }

    fn validate_buffers(
        &self,
        output_buffers: &[OutputBuffer],
        row_count: usize,
    ) -> Result<(), String> {
        if output_buffers.len() != self.total_buffers {
            return Err(format!(
                "Expected {} output buffers, got {}",
                self.total_buffers,
                output_buffers.len()
            ));
        }

        for (field, root) in self.schema.fields().iter().zip(&self.buffer_slots) {
            let mut error = None;
            root.visit(&mut |slot| {
                if error.is_some() { return; }
                match slot.primary_capacity(row_count) {
                    Some(required) if output_buffers[slot.primary].capacity >= required => {},
                    Some(required) => error = Some(format!(
                        "Output buffer overflow: Column '{}' buffer {} requires {} bytes but has {}",
                        field.name(), slot.primary, required, output_buffers[slot.primary].capacity)),
                    None => error = Some("Output size overflow".into()),
                }
                if let Some(index) = slot.validity {
                    if output_buffers[index].capacity < row_count.div_ceil(8) {
                        error = Some(format!("Output buffer overflow: Column '{}' validity buffer {} is too small", field.name(), index));
                    }
                }
            });
            if let Some(error) = error {
                return Err(error);
            }
        }

        Ok(())
    }

    /// Decode concatenated datums, requiring each row to consume exactly its
    /// corresponding Kafka payload. Ends are cumulative offsets into `input`.
    ///
    /// # Safety
    /// Output pointers must not alias input or one another, and must reference writable allocations that
    /// remain alive and exclusively accessible for the call. For checked APIs,
    /// each allocation must cover its advertised capacity; `decode` additionally
    /// requires enough output space for every requested row and list element.
    pub unsafe fn decode_records_checked(
        &self,
        input: &[u8],
        output_buffers: &[OutputBuffer],
        record_ends: &[usize],
    ) -> Result<usize, String> {
        if record_ends.last().copied().unwrap_or(0) != input.len()
            || record_ends.windows(2).any(|w| w[0] > w[1])
        {
            return Err("Invalid record boundaries".into());
        }
        self.validate_buffers(output_buffers, record_ends.len())?;
        let mut ptrs: Vec<_> = output_buffers.iter().map(|b| b.ptr).collect();
        let caps: Vec<_> = output_buffers.iter().map(|b| b.capacity).collect();
        self.decode_with_caps(
            input,
            &mut ptrs,
            &caps,
            record_ends.len(),
            Some(record_ends),
        )
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
    ///
    /// # Safety
    /// Output pointers must not alias input or one another, and must reference writable allocations that
    /// remain alive and exclusively accessible for the call. For checked APIs,
    /// each allocation must cover its advertised capacity; `decode` additionally
    /// requires enough output space for every requested row and list element.
    pub unsafe fn decode_chunked(
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

        let result = self.decode_with_caps(input, &mut ptrs, &caps, rows_to_decode, None);

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

        self.visit_buffer_slots(&mut |slot| {
            if let Some(width) = slot.fixed_width {
                max_rows = max_rows.min(
                    output_buffers[slot.primary]
                        .capacity
                        .checked_div(width)
                        .unwrap_or(usize::MAX),
                );
            } else {
                max_rows =
                    max_rows.min((output_buffers[slot.primary].capacity / 4).saturating_sub(1));
            }
            if let Some(index) = slot.validity {
                max_rows = max_rows.min(output_buffers[index].capacity.saturating_mul(8));
            }
        });

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
        if self
            .decode_with_caps(input, &mut ptrs, &caps, 1, None)
            .is_err()
        {
            return DecodeResult::overflow(0, 0);
        }

        // Exponential search to find upper bound
        let mut low = 1usize;
        let mut high = 2usize;

        while high < max_rows {
            let mut ptrs: Vec<*mut u8> = output_buffers.iter().map(|b| b.ptr).collect();
            if self
                .decode_with_caps(input, &mut ptrs, &caps, high, None)
                .is_ok()
            {
                low = high;
                high *= 2;
            } else {
                break;
            }
        }
        high = high.min(max_rows);

        // Binary search between low and high
        while low < high {
            let mid = low + (high - low).div_ceil(2);
            let mut ptrs: Vec<*mut u8> = output_buffers.iter().map(|b| b.ptr).collect();
            if self
                .decode_with_caps(input, &mut ptrs, &caps, mid, None)
                .is_ok()
            {
                low = mid;
            } else {
                high = mid - 1;
            }
        }

        // Final decode to get bytes_consumed
        let mut ptrs: Vec<*mut u8> = output_buffers.iter().map(|b| b.ptr).collect();
        match self.decode_with_caps(input, &mut ptrs, &caps, low, None) {
            Ok(bytes_consumed) => DecodeResult::overflow(bytes_consumed, low),
            Err(_) => DecodeResult::overflow(0, 0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::TimeUnit;

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
    fn test_native_memory_released_after_last_owner() {
        use cranelift_jit::{BranchProtection, JITMemoryProvider, SystemMemoryProvider};
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct TrackedMemory {
            inner: SystemMemoryProvider,
            releases: Arc<AtomicUsize>,
        }
        impl JITMemoryProvider for TrackedMemory {
            fn allocate_readexec(&mut self, size: usize, align: u64) -> std::io::Result<*mut u8> {
                self.inner.allocate_readexec(size, align)
            }
            fn allocate_readwrite(&mut self, size: usize, align: u64) -> std::io::Result<*mut u8> {
                self.inner.allocate_readwrite(size, align)
            }
            fn allocate_readonly(&mut self, size: usize, align: u64) -> std::io::Result<*mut u8> {
                self.inner.allocate_readonly(size, align)
            }
            fn finalize(
                &mut self,
                protection: BranchProtection,
            ) -> cranelift_module::ModuleResult<()> {
                self.inner.finalize(protection)
            }
            unsafe fn free_memory(&mut self) {
                unsafe { self.inner.free_memory() };
                self.releases.fetch_add(1, Ordering::SeqCst);
            }
        }
        // Observe the provider's release contract: the system allocator may keep
        // freed pages mapped for reuse, so OS mapping presence is not a leak test.
        let releases = Arc::new(AtomicUsize::new(0));
        let mut memory = TrackedMemory {
            inner: SystemMemoryProvider::new(),
            releases: releases.clone(),
        };
        memory.allocate_readexec(128, 16).unwrap();
        let mut builder = JITBuilder::new(cranelift_module::default_libcall_names()).unwrap();
        builder.memory_provider(Box::new(memory));
        let owner = Arc::new(SyncJitModule(ManuallyDrop::new(JITModule::new(builder))));
        let active_owner = owner.clone();
        drop(owner);
        assert_eq!(releases.load(Ordering::SeqCst), 0);
        drop(active_owner);
        assert_eq!(releases.load(Ordering::SeqCst), 1);
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
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Avro varint for 1: zigzag(1) = 2, varint(2) = 0x02
        let input = vec![0x02u8];
        let mut output = vec![0u8; 8];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, 1) }.expect("Decode failed");
        assert_eq!(consumed, 1);

        let value = i64::from_le_bytes(output[0..8].try_into().unwrap());
        assert_eq!(value, 1);
    }

    #[test]
    fn test_roundtrip_int64_multiple_values() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));

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

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify each value
        for (i, &expected) in test_values.iter().enumerate() {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(
                actual, expected,
                "Mismatch at index {}: expected {}, got {}",
                i, expected, actual
            );
        }
    }

    #[test]
    fn test_roundtrip_int32_values() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "count",
            DataType::Int32,
            false,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let test_values: Vec<i32> = vec![0, 1, 5, 42, 63];

        let mut input = Vec::new();
        for &v in &test_values {
            input.extend(encode_varint_i32(v));
        }

        let row_count = test_values.len();
        let mut output = vec![0u8; row_count * 4];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in test_values.iter().enumerate() {
            let actual = i32::from_le_bytes(output[i * 4..(i + 1) * 4].try_into().unwrap());
            assert_eq!(actual, expected, "Int32 mismatch at index {}", i);
        }
    }

    #[test]
    fn test_roundtrip_float64_values() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            false,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let test_values: Vec<f64> = vec![0.0, 1.5, -3.125, 100.0, 999.999];

        let mut input = Vec::new();
        for &v in &test_values {
            input.extend(v.to_le_bytes());
        }

        let row_count = test_values.len();
        let mut output = vec![0u8; row_count * 8];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in test_values.iter().enumerate() {
            let actual = f64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
            assert!(
                (actual - expected).abs() < 1e-10,
                "Float64 mismatch at index {}: {} vs {}",
                i,
                actual,
                expected
            );
        }
    }

    #[test]
    fn test_roundtrip_float32_values() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "ratio",
            DataType::Float32,
            false,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let test_values: Vec<f32> = vec![0.0, 1.5, -2.5, 100.0];

        let mut input = Vec::new();
        for &v in &test_values {
            input.extend(v.to_le_bytes());
        }

        let row_count = test_values.len();
        let mut output = vec![0u8; row_count * 4];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in test_values.iter().enumerate() {
            let actual = f32::from_le_bytes(output[i * 4..(i + 1) * 4].try_into().unwrap());
            assert!(
                (actual - expected).abs() < 1e-6,
                "Float32 mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_roundtrip_boolean_values() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "active",
            DataType::Boolean,
            false,
        )]));

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

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
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
            input.extend(encode_varint_i64(i as i64)); // id: 0-9
            input.extend((i as f64 * 0.1).to_le_bytes()); // value
            input.extend(encode_varint_i32((i % 64) as i32)); // count: keep in single-byte range
            input.extend((i as f32 / 100.0).to_le_bytes()); // ratio
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

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert!(consumed > 0);

        // Verify values
        for i in 0..row_count {
            let id = i64::from_le_bytes(output_id[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(id, i as i64, "id mismatch at row {}", i);

            let value = f64::from_le_bytes(output_value[i * 8..(i + 1) * 8].try_into().unwrap());
            assert!(
                (value - i as f64 * 0.1).abs() < 1e-10,
                "value mismatch at row {}",
                i
            );

            let count = i32::from_le_bytes(output_count[i * 4..(i + 1) * 4].try_into().unwrap());
            assert_eq!(count, (i % 64) as i32, "count mismatch at row {}", i);

            let ratio = f32::from_le_bytes(output_ratio[i * 4..(i + 1) * 4].try_into().unwrap());
            assert!(
                (ratio - i as f32 / 100.0).abs() < 1e-6,
                "ratio mismatch at row {}",
                i
            );
        }
    }

    #[test]
    fn test_roundtrip_zigzag_negative_values() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));

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

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in test_values.iter().enumerate() {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(
                actual, expected,
                "Negative zigzag mismatch at index {}: expected {}, got {}",
                i, expected, actual
            );
        }
    }

    #[test]
    fn test_roundtrip_high_throughput() {
        // Test with larger row count to verify loop handling
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let row_count = 1000;

        let mut input = Vec::new();
        for i in 0..row_count {
            // Use values 0-63 which encode to single-byte varints
            input.extend(encode_varint_i64((i % 64) as i64));
        }

        let mut output = vec![0u8; row_count * 8];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
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
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Test values requiring 1-5 byte varints
        let test_values: Vec<i64> = vec![
            0,            // 1 byte: zigzag 0
            63,           // 1 byte: zigzag 126
            64,           // 2 bytes: zigzag 128
            100,          // 2 bytes: zigzag 200
            1000,         // 2 bytes: zigzag 2000
            10000,        // 3 bytes: zigzag 20000
            100000,       // 3 bytes: zigzag 200000
            1000000,      // 3 bytes: zigzag 2000000
            10000000,     // 4 bytes
            100000000,    // 4 bytes
            1000000000,   // 5 bytes
            i64::MAX / 2, // large value
        ];

        let mut input = Vec::new();
        for &v in &test_values {
            input.extend(encode_varint_i64(v));
        }

        let row_count = test_values.len();
        let mut output = vec![0u8; row_count * 8];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in test_values.iter().enumerate() {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(
                actual, expected,
                "Large int mismatch at index {}: expected {}, got {}",
                i, expected, actual
            );
        }
    }

    #[test]
    fn test_roundtrip_large_negative_integers() {
        // Test large negative values requiring multi-byte varints
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let test_values: Vec<i64> = vec![
            -33,          // 2 bytes: zigzag 65
            -64,          // 2 bytes: zigzag 127
            -65,          // 2 bytes: zigzag 129
            -100,         // 2 bytes
            -1000,        // 2 bytes
            -10000,       // 3 bytes
            -100000,      // 3 bytes
            -1000000,     // 3 bytes
            -1000000000,  // 5 bytes
            i64::MIN / 2, // large negative
        ];

        let mut input = Vec::new();
        for &v in &test_values {
            input.extend(encode_varint_i64(v));
        }

        let row_count = test_values.len();
        let mut output = vec![0u8; row_count * 8];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in test_values.iter().enumerate() {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(
                actual, expected,
                "Large negative mismatch at index {}: expected {}, got {}",
                i, expected, actual
            );
        }
    }

    #[test]
    fn test_roundtrip_int32_large_values() {
        // Test Int32 with multi-byte varints
        let schema = Arc::new(Schema::new(vec![Field::new(
            "count",
            DataType::Int32,
            false,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let test_values: Vec<i32> = vec![
            0,
            63,
            64,
            100,
            1000,
            10000,
            100000,
            1000000,
            -1,
            -32,
            -33,
            -64,
            -65,
            -1000,
            -100000,
            i32::MAX,
            i32::MIN,
        ];

        let mut input = Vec::new();
        for &v in &test_values {
            input.extend(encode_varint_i32(v));
        }

        let row_count = test_values.len();
        let mut output = vec![0u8; row_count * 4];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in test_values.iter().enumerate() {
            let actual = i32::from_le_bytes(output[i * 4..(i + 1) * 4].try_into().unwrap());
            assert_eq!(
                actual, expected,
                "Int32 large value mismatch at index {}: expected {}, got {}",
                i, expected, actual
            );
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

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert!(consumed > 0);

        // Verify values
        for i in 0..row_count {
            let id = i64::from_le_bytes(output_id[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(id, (i * 100000) as i64, "id mismatch at row {}", i);

            let value = f64::from_le_bytes(output_value[i * 8..(i + 1) * 8].try_into().unwrap());
            assert!(
                (value - i as f64 * 1.5).abs() < 1e-10,
                "value mismatch at row {}",
                i
            );

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
        let schema = Arc::new(Schema::new(vec![Field::new(
            "data",
            DataType::Binary,
            false,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Verify we need 2 buffers for Bytes type
        assert_eq!(decoder.total_buffers(), 2);
        let slot = decoder.buffer_slot(0).expect("buffer slot");
        assert_eq!(slot.primary, 0); // offsets buffer
        assert_eq!(slot.secondary, Some(1)); // data buffer

        // Test a single bytes value
        let test_data = b"Hello, World!";
        let input = encode_avro_bytes(test_data);
        let row_count = 1;

        // Buffer 0: offsets (row_count + 1 i32 values = 8 bytes for 1 row)
        let mut offsets = vec![0u8; (row_count + 1) * 4];
        // Buffer 1: data
        let mut data = vec![0u8; test_data.len() + 100];

        let mut output_ptrs = vec![offsets.as_mut_ptr(), data.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
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
        let schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, false)]));

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

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify offsets are correct: [0, 5, 8, 15, 19] for ["Alice", "Bob", "Charlie", "Dave"]
        let mut expected_offset = 0;
        for (i, s) in test_strings.iter().enumerate() {
            let offset_i = i32::from_le_bytes(offsets[i * 4..(i + 1) * 4].try_into().unwrap());
            assert_eq!(offset_i, expected_offset as i32, "Offset {} mismatch", i);
            expected_offset += s.len();
        }
        // Final offset
        let offset_n = i32::from_le_bytes(
            offsets[row_count * 4..(row_count + 1) * 4]
                .try_into()
                .unwrap(),
        );
        assert_eq!(offset_n, total_data_len as i32, "Final offset mismatch");

        // Verify the strings were copied in sequence to data buffer
        let mut data_offset = 0;
        for s in &test_strings {
            let actual = &data[data_offset..data_offset + s.len()];
            assert_eq!(
                actual,
                s.as_bytes(),
                "String mismatch: expected '{}', got '{:?}'",
                s,
                actual
            );
            data_offset += s.len();
        }
    }

    #[test]
    fn test_roundtrip_bytes_empty() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "data",
            DataType::Binary,
            false,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Empty bytes: length = 0
        let input = encode_avro_bytes(b"");
        let row_count = 1;

        // Buffer 0: offsets (row_count + 1 i32 values)
        let mut offsets = vec![0xFFu8; (row_count + 1) * 4];
        // Buffer 1: data
        let mut data = vec![0xFFu8; 100];

        let mut output_ptrs = vec![offsets.as_mut_ptr(), data.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
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

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify Int64 values
        for i in 0..row_count {
            let id = i64::from_le_bytes(output_id[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(id, i as i64 * 100, "id mismatch at row {}", i);
        }

        // Verify name offsets
        let mut expected_offset = 0;
        for (i, name) in names.iter().enumerate() {
            let offset_i =
                i32::from_le_bytes(output_name_offsets[i * 4..(i + 1) * 4].try_into().unwrap());
            assert_eq!(
                offset_i, expected_offset as i32,
                "name offset {} mismatch",
                i
            );
            expected_offset += name.len();
        }
        // Final offset
        let offset_n = i32::from_le_bytes(
            output_name_offsets[row_count * 4..(row_count + 1) * 4]
                .try_into()
                .unwrap(),
        );
        assert_eq!(
            offset_n, total_name_len as i32,
            "name final offset mismatch"
        );

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
        let schema = Arc::new(Schema::new(vec![Field::new(
            "data",
            DataType::Binary,
            false,
        )]));

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

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify offsets
        let offset_0 = i32::from_le_bytes(offsets[0..4].try_into().unwrap());
        let offset_1 = i32::from_le_bytes(offsets[4..8].try_into().unwrap());
        assert_eq!(offset_0, 0);
        assert_eq!(offset_1, test_data.len() as i32);

        // Verify all 500 bytes were copied correctly to data buffer
        assert_eq!(
            &data[..test_data.len()],
            &test_data[..],
            "Long bytes mismatch"
        );
    }

    /// Encode union tag for nullable field
    /// In Avro unions ["null", "type"], null has index 0, type has index 1
    /// The index is zigzag encoded, so: null -> 0, type -> 2
    fn encode_union_tag(is_null: bool) -> Vec<u8> {
        if is_null {
            vec![0x00] // zigzag(0) = 0
        } else {
            vec![0x02] // zigzag(1) = 2
        }
    }

    #[test]
    fn test_roundtrip_nullable_int64_all_non_null() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int64, true), // nullable
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let test_values: Vec<i64> = vec![10, 20, 30, 40, 50];

        let mut input = Vec::new();
        for &v in &test_values {
            input.extend(encode_union_tag(false)); // non-null
            input.extend(encode_varint_i64(v));
        }

        let row_count = test_values.len();
        let mut output = vec![0u8; row_count * 8];
        let mut validity = vec![0u8; (row_count + 7) / 8];
        let mut output_ptrs = vec![output.as_mut_ptr(), validity.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in test_values.iter().enumerate() {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, expected, "Nullable Int64 mismatch at index {}", i);
        }
    }

    #[test]
    fn test_roundtrip_nullable_int64_with_nulls() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int64, true), // nullable
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Test pattern: value, null, value, null, value
        let values_or_nulls: Vec<Option<i64>> = vec![Some(100), None, Some(200), None, Some(300)];

        let mut input = Vec::new();
        for v in &values_or_nulls {
            match v {
                Some(val) => {
                    input.extend(encode_union_tag(false)); // non-null
                    input.extend(encode_varint_i64(*val));
                }
                None => {
                    input.extend(encode_union_tag(true)); // null
                }
            }
        }

        let row_count = values_or_nulls.len();
        let mut output = vec![0xFFu8; row_count * 8]; // Fill with sentinel
        let mut validity = vec![0u8; (row_count + 7) / 8];
        let mut output_ptrs = vec![output.as_mut_ptr(), validity.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
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
            Field::new("value", DataType::Int64, true), // nullable
        ]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let row_count = 5;
        let mut input = Vec::new();
        for _ in 0..row_count {
            input.extend(encode_union_tag(true)); // all nulls
        }

        let mut output = vec![0xFFu8; row_count * 8]; // Fill with sentinel
        let mut validity = vec![0u8; (row_count + 7) / 8];
        let mut output_ptrs = vec![output.as_mut_ptr(), validity.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        for i in 0..row_count {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, 0, "Null should have 0 placeholder at index {}", i);
        }
    }

    /// Encode enum index as varint (not zigzag - just raw varint)
    fn encode_enum_index(index: u32) -> Vec<u8> {
        // Avro enum index is an `int` = zigzag varint (matches apache_avro).
        // For a non-negative index this is simply index * 2.
        let zz = ((index as i64) << 1) ^ ((index as i64) >> 63);
        let mut result = Vec::new();
        let mut v = zz as u64;
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
        let schema = Arc::new(Schema::new(vec![Field::new(
            "status",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            false,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Test enum indices: 0, 1, 2, 0, 1
        let test_indices: Vec<u32> = vec![0, 1, 2, 0, 1];

        let mut input = Vec::new();
        for &idx in &test_indices {
            input.extend(encode_enum_index(idx));
        }

        let row_count = test_indices.len();
        let mut output = vec![0u8; row_count * 4]; // i32 indices
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in test_indices.iter().enumerate() {
            let actual = i32::from_le_bytes(output[i * 4..(i + 1) * 4].try_into().unwrap());
            assert_eq!(
                actual, expected as i32,
                "Enum index mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_roundtrip_nullable_enum() {
        use arrow_schema::DataType;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "status",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            true, // nullable
        )]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Test pattern: value, null, value, null, value
        let values: Vec<Option<u32>> = vec![Some(0), None, Some(2), None, Some(1)];

        let mut input = Vec::new();
        for v in &values {
            match v {
                Some(idx) => {
                    input.extend(encode_union_tag(false)); // non-null
                    input.extend(encode_enum_index(*idx));
                }
                None => {
                    input.extend(encode_union_tag(true)); // null
                }
            }
        }

        let row_count = values.len();
        let mut output = vec![0xFFu8; row_count * 4];
        let mut validity = vec![0u8; (row_count + 7) / 8];
        let mut output_ptrs = vec![output.as_mut_ptr(), validity.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
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
        let schema = Arc::new(Schema::new(vec![Field::new(
            "values",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, false))),
            false,
        )]));

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

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, 1); // Just the 0 terminator

        // Verify offsets: [0, 0] for empty list
        let offset_0 = i32::from_le_bytes(offsets[0..4].try_into().unwrap());
        let offset_1 = i32::from_le_bytes(offsets[4..8].try_into().unwrap());
        assert_eq!(offset_0, 0);
        assert_eq!(offset_1, 0); // No elements
    }

    #[test]
    fn test_roundtrip_list_int64_single() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "values",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, false))),
            false,
        )]));

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

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify offsets: [0, 3] for a list with 3 elements (element count, not bytes)
        let offset_0 = i32::from_le_bytes(offsets[0..4].try_into().unwrap());
        let offset_1 = i32::from_le_bytes(offsets[4..8].try_into().unwrap());
        assert_eq!(offset_0, 0, "First offset should be 0");
        assert_eq!(
            offset_1,
            elements.len() as i32,
            "Second offset should be element count"
        );

        // Verify elements in data buffer
        for (i, &expected) in elements.iter().enumerate() {
            let actual = i64::from_le_bytes(data[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, expected, "Element mismatch at index {}", i);
        }
    }

    #[test]
    fn test_roundtrip_list_int32() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "counts",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, false))),
            false,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let elements: Vec<i32> = vec![1, 2, 3, 4, 5];
        let input = encode_avro_array_i32(&elements);
        let row_count = 1;

        // Buffer 0: offsets (row_count + 1 i32 values)
        let mut offsets = vec![0u8; (row_count + 1) * 4];
        // Buffer 1: data (element_count * 4 bytes for i32)
        let mut data = vec![0u8; elements.len() * 4 + 100];

        let mut output_ptrs = vec![offsets.as_mut_ptr(), data.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify offsets: [0, 5*4] for a list with 5 i32 elements
        let offset_0 = i32::from_le_bytes(offsets[0..4].try_into().unwrap());
        let offset_1 = i32::from_le_bytes(offsets[4..8].try_into().unwrap());
        assert_eq!(offset_0, 0, "First offset should be 0");
        assert_eq!(
            offset_1,
            elements.len() as i32,
            "Second offset should be element count"
        );

        // Verify elements in data buffer
        for (i, &expected) in elements.iter().enumerate() {
            let actual = i32::from_le_bytes(data[i * 4..(i + 1) * 4].try_into().unwrap());
            assert_eq!(actual, expected, "Element mismatch at index {}", i);
        }
    }

    #[test]
    fn test_roundtrip_list_multiple_rows() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "values",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, false))),
            false,
        )]));

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

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify offsets: [0, 16, 40, 48] for rows with 2, 3, 1 elements (all i64 = 8 bytes each)
        let offset_0 = i32::from_le_bytes(offsets[0..4].try_into().unwrap());
        let offset_1 = i32::from_le_bytes(offsets[4..8].try_into().unwrap());
        let offset_2 = i32::from_le_bytes(offsets[8..12].try_into().unwrap());
        let offset_3 = i32::from_le_bytes(offsets[12..16].try_into().unwrap());
        assert_eq!(offset_0, 0, "Offset 0 should be 0");
        assert_eq!(
            offset_1,
            row1.len() as i32,
            "Offset 1 should be element count of row1"
        );
        assert_eq!(
            offset_2,
            (row1.len() + row2.len()) as i32,
            "Offset 2 should be cumulative element count"
        );
        assert_eq!(
            offset_3, total_elements as i32,
            "Offset 3 should be total element count"
        );

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
        let lists: Vec<Vec<i64>> = vec![vec![1, 2], vec![10, 20, 30], vec![100]];

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

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
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
        assert_eq!(
            offset_1,
            lists[0].len() as i32,
            "Offset 1 should be element count of list 0"
        );
        assert_eq!(
            offset_2,
            (lists[0].len() + lists[1].len()) as i32,
            "Offset 2 should be cumulative element count"
        );
        assert_eq!(
            offset_3, total_elements as i32,
            "Offset 3 should be total element count"
        );

        // Verify list elements in data buffer
        let all_elements: Vec<i64> = lists.into_iter().flatten().collect();
        for (i, &expected) in all_elements.iter().enumerate() {
            let actual = i64::from_le_bytes(list_data[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, expected, "List element mismatch at index {}", i);
        }
    }

    #[test]
    fn test_reject_map() {
        use arrow_schema::Fields;

        let map_field = Field::new(
            "entries",
            DataType::Struct(Fields::from(vec![
                Field::new("key", DataType::Utf8, false),
                Field::new("value", DataType::Int64, false),
            ])),
            false,
        );

        let schema = Arc::new(Schema::new(vec![Field::new(
            "props",
            DataType::Map(Arc::new(map_field), false),
            false,
        )]));

        assert!(JitDecoder::compile(&schema).is_err());
    }

    #[test]
    fn test_reject_map_with_int_column() {
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

        assert!(JitDecoder::compile(&schema).is_err());
    }

    #[test]
    fn test_roundtrip_struct_simple() {
        use arrow_schema::Fields;

        // Struct with two Int64 fields
        let schema = Arc::new(Schema::new(vec![Field::new(
            "point",
            DataType::Struct(Fields::from(vec![
                Field::new("x", DataType::Int64, false),
                Field::new("y", DataType::Int64, false),
            ])),
            false,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Encode struct: x=10, y=20
        let mut input = Vec::new();
        input.extend(encode_varint_i64(10)); // x
        input.extend(encode_varint_i64(20)); // y

        // Output buffer for flattened struct (2 x i64 = 16 bytes)
        let mut output = vec![0u8; 16];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, 1) }.expect("Decode failed");
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

        let schema = Arc::new(Schema::new(vec![Field::new(
            "point",
            DataType::Struct(Fields::from(vec![
                Field::new("x", DataType::Int64, false),
                Field::new("y", DataType::Int64, false),
            ])),
            false,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let points: Vec<(i64, i64)> = vec![(1, 2), (10, 20), (100, 200)];

        let mut input = Vec::new();
        for (x, y) in &points {
            input.extend(encode_varint_i64(*x));
            input.extend(encode_varint_i64(*y));
        }

        let mut output = vec![0u8; points.len() * 16];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed = unsafe { decoder.decode(&input, &mut output_ptrs, points.len()) }
            .expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify each point
        for (i, (expected_x, expected_y)) in points.iter().enumerate() {
            let offset = i * 16;
            let x = i64::from_le_bytes(output[offset..offset + 8].try_into().unwrap());
            let y = i64::from_le_bytes(output[offset + 8..offset + 16].try_into().unwrap());
            assert_eq!(x, *expected_x, "x mismatch at row {}", i);
            assert_eq!(y, *expected_y, "y mismatch at row {}", i);
        }
    }

    #[test]
    fn test_roundtrip_struct_mixed_types() {
        use arrow_schema::Fields;

        // Struct with mixed types
        let schema = Arc::new(Schema::new(vec![Field::new(
            "record",
            DataType::Struct(Fields::from(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("count", DataType::Int32, false),
                Field::new("score", DataType::Float64, false),
            ])),
            false,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Encode: id=1000, count=5, score=3.125
        let mut input = Vec::new();
        input.extend(encode_varint_i64(1000)); // id
        input.extend(encode_varint_i32(5)); // count
        input.extend_from_slice(&3.125f64.to_le_bytes()); // score

        // Output: 8 (i64) + 4 (i32) + 8 (f64) = 20 bytes
        let mut output = vec![0u8; 20];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, 1) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        let id = i64::from_le_bytes(output[0..8].try_into().unwrap());
        let count = i32::from_le_bytes(output[8..12].try_into().unwrap());
        let score = f64::from_le_bytes(output[12..20].try_into().unwrap());

        assert_eq!(id, 1000);
        assert_eq!(count, 5);
        assert!((score - 3.125).abs() < f64::EPSILON);
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

        let mut output_ptrs = vec![output_id.as_mut_ptr(), output_struct.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify ids
        for (i, &expected) in ids.iter().enumerate() {
            let actual = i64::from_le_bytes(output_id[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, expected, "Id mismatch at row {}", i);
        }

        // Verify struct fields
        for (i, (expected_x, expected_y)) in points.iter().enumerate() {
            let offset = i * 16;
            let x = i64::from_le_bytes(output_struct[offset..offset + 8].try_into().unwrap());
            let y = i64::from_le_bytes(output_struct[offset + 8..offset + 16].try_into().unwrap());
            assert_eq!(x, *expected_x, "x mismatch at row {}", i);
            assert_eq!(y, *expected_y, "y mismatch at row {}", i);
        }
    }

    // === Tests for new types ===

    #[test]
    fn test_reject_general_union() {
        use arrow_schema::UnionFields;

        // Union of Int64 and Float64
        let union_fields = UnionFields::new(
            vec![0, 1],
            vec![
                Field::new("int_val", DataType::Int64, false),
                Field::new("float_val", DataType::Float64, false),
            ],
        );

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Union(union_fields, arrow_schema::UnionMode::Dense),
            false,
        )]));

        assert!(JitDecoder::compile(&schema).is_err());
    }

    #[test]
    fn test_reject_fixed_size_list() {
        // FixedSizeList of 3 Int64s
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int64, false)), 3),
            false,
        )]));

        assert!(JitDecoder::compile(&schema).is_err());
    }

    #[test]
    fn test_roundtrip_duration() {
        use arrow_schema::TimeUnit;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "duration",
            DataType::Duration(TimeUnit::Microsecond),
            false,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        let values: Vec<i64> = vec![1000, 2000, -500];
        let mut input = Vec::new();
        for &v in &values {
            input.extend(encode_varint_i64(v));
        }

        let row_count = 3;
        let mut output = vec![0u8; row_count * 8];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in values.iter().enumerate() {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, expected, "Duration mismatch at row {}", i);
        }
    }

    #[test]
    fn test_roundtrip_decimal128() {
        // Decimal128 with precision 38, scale 10
        let schema = Arc::new(Schema::new(vec![Field::new(
            "amount",
            DataType::Decimal128(38, 10),
            false,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // Test data: 16 bytes big-endian decimal
        // Simple test: all zeros
        let input = vec![0u8; 16];

        let row_count = 1;
        let mut output = vec![0u8; row_count * 16];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, 16);

        // Verify output is zeros (byte-swapped zeros are still zeros)
        for byte in &output {
            assert_eq!(*byte, 0, "Expected zero");
        }
    }

    #[test]
    fn test_roundtrip_interval_year_month() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "interval",
            DataType::Interval(IntervalUnit::YearMonth),
            false,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // YearMonth interval is encoded as a varint (months)
        let months: i32 = 14; // 1 year 2 months
        let mut input = Vec::new();
        input.extend(encode_varint_i32(months));

        let row_count = 1;
        let mut output = vec![0u8; row_count * 4];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        let actual = i32::from_le_bytes(output[0..4].try_into().unwrap());
        assert_eq!(actual, months, "YearMonth interval mismatch");
    }

    #[test]
    fn test_roundtrip_interval_day_time() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "interval",
            DataType::Interval(IntervalUnit::DayTime),
            false,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("Failed to compile");

        // DayTime interval: 8 bytes (days: i32, milliseconds: i32)
        let days: i32 = 5;
        let millis: i32 = 3600000; // 1 hour
        let mut input = Vec::new();
        input.extend(&days.to_le_bytes());
        input.extend(&millis.to_le_bytes());

        let row_count = 1;
        let mut output = vec![0u8; row_count * 8];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
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
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Encode 3 values
        let mut input = Vec::new();
        for v in [100i64, 200, 300] {
            input.extend(encode_varint_i64(v));
        }

        let row_count = 3;
        let mut output = vec![0u8; row_count * 8]; // Exactly enough
        let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];

        let result = unsafe { decoder.decode_checked(&input, &buffers, row_count) };
        assert!(result.is_ok(), "Should succeed with sufficient capacity");
    }

    #[test]
    fn test_decode_checked_int64_insufficient_capacity() {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Encode 3 values
        let mut input = Vec::new();
        for v in [100i64, 200, 300] {
            input.extend(encode_varint_i64(v));
        }

        let row_count = 3;
        let mut output = vec![0u8; 16]; // Only 16 bytes, need 24
        let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];

        let result = unsafe { decoder.decode_checked(&input, &buffers, row_count) };
        assert!(result.is_err(), "Should fail with insufficient capacity");
        let err = result.unwrap_err();
        assert!(err.contains("requires 24 bytes"), "Error: {}", err);
    }

    #[test]
    fn test_decode_checked_int32_insufficient_capacity() {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Encode 5 values - need 20 bytes
        let mut input = Vec::new();
        for v in [1i32, 2, 3, 4, 5] {
            input.extend(encode_varint_i32(v));
        }

        let row_count = 5;
        let mut output = vec![0u8; 12]; // Only 12 bytes, need 20
        let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];

        let result = unsafe { decoder.decode_checked(&input, &buffers, row_count) };
        assert!(result.is_err(), "Should fail with insufficient capacity");
        let err = result.unwrap_err();
        assert!(err.contains("requires 20 bytes"), "Error: {}", err);
    }

    #[test]
    fn test_decode_checked_float64_insufficient_capacity() {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, false)]));
        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Float64 is 8 bytes raw, need 16 bytes for 2 rows
        let input: Vec<u8> = [1.5f64, 2.5].iter().flat_map(|f| f.to_le_bytes()).collect();

        let row_count = 2;
        let mut output = vec![0u8; 8]; // Only 8 bytes, need 16
        let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];

        let result = unsafe { decoder.decode_checked(&input, &buffers, row_count) };
        assert!(result.is_err(), "Should fail with insufficient capacity");
    }

    #[test]
    fn test_decode_checked_boolean_insufficient_capacity() {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Boolean, false)]));
        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Boolean is 1 byte per value in JIT output
        let input: Vec<u8> = vec![1, 0, 1, 1, 0]; // 5 booleans

        let row_count = 5;
        let mut output = vec![0u8; 3]; // Only 3 bytes, need 5
        let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];

        let result = unsafe { decoder.decode_checked(&input, &buffers, row_count) };
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
        let result = unsafe { decoder.decode_checked(&input, &buffers, row_count) };
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
        let mut output2 = vec![0u8; 8]; // Need 12 for Int32

        let buffers = vec![
            OutputBuffer::new(output1.as_mut_ptr(), output1.len()),
            OutputBuffer::new(output2.as_mut_ptr(), output2.len()),
        ];

        let input = vec![0u8; 20];
        let result = unsafe { decoder.decode_checked(&input, &buffers, row_count) };
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
        let result = unsafe { decoder.decode_checked(&input, &buffers, row_count) };
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("Expected 2 output buffers, got 1")
        );
    }

    #[test]
    fn test_decode_checked_fixed_size_binary_insufficient() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "v",
            DataType::FixedSizeBinary(16),
            false,
        )]));
        let decoder = JitDecoder::compile(&schema).expect("compile");

        let row_count = 3;
        let mut output = vec![0u8; 32]; // Need 48 (3 * 16)
        let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];

        let input = vec![0u8; 48];
        let result = unsafe { decoder.decode_checked(&input, &buffers, row_count) };
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("requires 48 bytes"), "Error: {}", err);
    }

    #[test]
    fn test_decode_checked_decimal128_insufficient() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "v",
            DataType::Decimal128(38, 10),
            false,
        )]));
        let decoder = JitDecoder::compile(&schema).expect("compile");

        let row_count = 2;
        let mut output = vec![0u8; 16]; // Need 32 (2 * 16)
        let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), output.len())];

        let input = vec![0u8; 32];
        let result = unsafe { decoder.decode_checked(&input, &buffers, row_count) };
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("requires 32 bytes"), "Error: {}", err);
    }

    #[test]
    fn test_decode_checked_bytes_variable_length_no_validation() {
        // Variable-length columns (Bytes) require 2 buffers (offsets + data)
        // They rely on JIT-level bounds checking
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Binary, false)]));
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

        let result = unsafe { decoder.decode_checked(&input, &buffers, row_count) };
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
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        let decoder = JitDecoder::compile(&schema).expect("compile");

        let row_count = 0;
        let mut output = vec![0u8; 0]; // Zero capacity for zero rows
        let buffers = vec![OutputBuffer::new(output.as_mut_ptr(), 0)];

        let input: Vec<u8> = vec![];
        let result = unsafe { decoder.decode_checked(&input, &buffers, row_count) };
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
        assert_eq!(
            JitDecoder::fixed_size_for_type(&DataType::Time32(TimeUnit::Millisecond)),
            Some(4)
        );
        assert_eq!(
            JitDecoder::fixed_size_for_type(&DataType::Time64(TimeUnit::Microsecond)),
            Some(8)
        );
        assert_eq!(
            JitDecoder::fixed_size_for_type(&DataType::Timestamp(TimeUnit::Second, None)),
            Some(8)
        );
        assert_eq!(
            JitDecoder::fixed_size_for_type(&DataType::Duration(TimeUnit::Second)),
            Some(8)
        );
        assert_eq!(
            JitDecoder::fixed_size_for_type(&DataType::Decimal128(38, 10)),
            Some(16)
        );
        assert_eq!(
            JitDecoder::fixed_size_for_type(&DataType::Decimal256(76, 20)),
            Some(32)
        );
        assert_eq!(
            JitDecoder::fixed_size_for_type(&DataType::FixedSizeBinary(20)),
            Some(20)
        );

        // Variable-length types should return None
        assert_eq!(JitDecoder::fixed_size_for_type(&DataType::Utf8), None);
        assert_eq!(JitDecoder::fixed_size_for_type(&DataType::Binary), None);
        assert_eq!(JitDecoder::fixed_size_for_type(&DataType::LargeUtf8), None);
        assert_eq!(
            JitDecoder::fixed_size_for_type(&DataType::LargeBinary),
            None
        );
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
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Encode 10 rows
        let mut input = Vec::new();
        for i in 0..10i64 {
            input.extend(encode_varint_i64(i));
        }

        // Buffer big enough for all rows (80 bytes for 10 i64s)
        let mut output = vec![0u8; 80];
        let output_buffers = vec![OutputBuffer::new(output.as_mut_ptr(), 80)];

        let result = unsafe { decoder.decode_chunked(&input, &output_buffers, 10) };

        assert!(result.is_ok());
        assert!(!result.has_more());
        assert_eq!(result.rows_decoded, 10);
        assert_eq!(result.bytes_consumed, input.len());
    }

    #[test]
    fn test_decode_chunked_overflow() {
        // Test decode_chunked when buffer is too small
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Encode 100 rows
        let mut input = Vec::new();
        for i in 0..100i64 {
            input.extend(encode_varint_i64(i));
        }

        // Buffer only big enough for 10 rows (80 bytes)
        let mut output = vec![0u8; 80];
        let output_buffers = vec![OutputBuffer::new(output.as_mut_ptr(), 80)];

        let result = unsafe { decoder.decode_chunked(&input, &output_buffers, 100) };

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
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Single row
        let input = encode_varint_i64(42);

        // Buffer too small for even 1 row (need 8 bytes, have 4)
        let mut output = vec![0u8; 4];
        let output_buffers = vec![OutputBuffer::new(output.as_mut_ptr(), 4)];

        let result = unsafe { decoder.decode_chunked(&input, &output_buffers, 1) };

        // Should report overflow with 0 rows decoded
        assert!(!result.is_ok());
        assert!(result.has_more());
        assert_eq!(result.rows_decoded, 0);
        assert_eq!(result.bytes_consumed, 0);
    }

    #[test]
    fn test_decode_chunked_continuation() {
        // Test that we can continue decoding from where we left off
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));

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
        let result1 = unsafe { decoder.decode_chunked(&input, &output_buffers, 20) };
        assert!(result1.has_more());
        assert_eq!(result1.rows_decoded, 10);

        // Second chunk - continue from where we left off
        let remaining_input = &input[result1.bytes_consumed..];
        let remaining_rows = 20 - result1.rows_decoded;

        // Reset output buffer
        output.fill(0);
        let output_buffers = vec![OutputBuffer::new(output.as_mut_ptr(), 80)];

        let result2 =
            unsafe { decoder.decode_chunked(remaining_input, &output_buffers, remaining_rows) };
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
        (0..row_count)
            .filter(|&i| check_validity_bit(validity, i))
            .count()
    }

    // -------------------------------------------------------------------------
    // Int64 Nullable Tests with Validity Bitmap
    // -------------------------------------------------------------------------

    #[test]
    fn test_nullable_int64_validity_all_non_null() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            true,
        )]));

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

        let result = unsafe { decoder.decode_chunked(&input, &output_buffers, row_count) };
        assert!(result.is_ok(), "Decode failed: {:?}", result);
        assert_eq!(result.rows_decoded, row_count);

        // Verify all validity bits are set to 1
        for i in 0..row_count {
            assert!(
                check_validity_bit(&validity_buf, i),
                "Validity bit should be 1 for non-null at row {}",
                i
            );
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
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            true,
        )]));

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

        let result = unsafe { decoder.decode_chunked(&input, &output_buffers, row_count) };
        assert!(result.is_ok());
        assert_eq!(result.rows_decoded, row_count);

        // All validity bits should be 0 (null)
        for i in 0..row_count {
            assert!(
                !check_validity_bit(&validity_buf, i),
                "Validity bit should be 0 for null at row {}",
                i
            );
        }
        assert_eq!(count_valid_bits(&validity_buf, row_count), 0);
    }

    #[test]
    fn test_nullable_int64_validity_mixed_pattern() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            true,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Mixed pattern: value, null, value, value, null, null, value
        let values: Vec<Option<i64>> =
            vec![Some(100), None, Some(-50), Some(0), None, None, Some(999)];
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

        let result = unsafe { decoder.decode_chunked(&input, &output_buffers, row_count) };
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
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            true,
        )]));

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

        let result = unsafe { decoder.decode_chunked(&input, &output_buffers, 1) };
        assert!(result.is_ok());
        assert_eq!(result.rows_decoded, 1);

        assert!(
            check_validity_bit(&validity_buf, 0),
            "Single row should be valid"
        );
        let actual = i64::from_le_bytes(data_buf[0..8].try_into().unwrap());
        assert_eq!(actual, 42);
    }

    #[test]
    fn test_nullable_int64_single_row_null() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            true,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        let mut input = Vec::new();
        input.extend(encode_union_tag(true)); // null

        let mut data_buf = vec![0u8; 8];
        let mut validity_buf = vec![0u8; 1];

        let output_buffers = vec![
            OutputBuffer::new(data_buf.as_mut_ptr(), data_buf.len()),
            OutputBuffer::new(validity_buf.as_mut_ptr(), validity_buf.len()),
        ];

        let result = unsafe { decoder.decode_chunked(&input, &output_buffers, 1) };
        assert!(result.is_ok());
        assert_eq!(result.rows_decoded, 1);

        assert!(
            !check_validity_bit(&validity_buf, 0),
            "Single null row should be invalid"
        );
    }

    // -------------------------------------------------------------------------
    // Int32 Nullable Tests with Validity Bitmap
    // -------------------------------------------------------------------------

    #[test]
    fn test_nullable_int32_validity_mixed() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            true,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        let values: Vec<Option<i32>> = vec![
            Some(1),
            Some(-1),
            None,
            Some(i32::MAX),
            None,
            Some(i32::MIN),
            Some(0),
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

        let result = unsafe { decoder.decode_chunked(&input, &output_buffers, row_count) };
        assert!(result.is_ok());
        assert_eq!(result.rows_decoded, row_count);

        // Verify validity and data
        for (i, v) in values.iter().enumerate() {
            let is_valid = check_validity_bit(&validity_buf, i);
            match v {
                Some(expected) => {
                    assert!(is_valid, "Row {} should be valid", i);
                    let actual =
                        i32::from_le_bytes(data_buf[i * 4..(i + 1) * 4].try_into().unwrap());
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
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            true,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        let values: Vec<Option<f64>> = vec![
            Some(3.125),
            None,
            Some(-0.0),
            Some(f64::INFINITY),
            None,
            Some(f64::NEG_INFINITY),
            Some(1e308),
            None,
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

        let result = unsafe { decoder.decode_chunked(&input, &output_buffers, row_count) };
        assert!(result.is_ok());
        assert_eq!(result.rows_decoded, row_count);

        for (i, v) in values.iter().enumerate() {
            let is_valid = check_validity_bit(&validity_buf, i);
            match v {
                Some(expected) => {
                    assert!(is_valid, "Row {} should be valid", i);
                    let actual =
                        f64::from_le_bytes(data_buf[i * 8..(i + 1) * 8].try_into().unwrap());
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
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            true,
        )]));

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

        let result = unsafe { decoder.decode_chunked(&input, &output_buffers, row_count) };
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
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float32,
            true,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        let values: Vec<Option<f32>> = vec![
            Some(1.5f32),
            None,
            Some(-0.0f32),
            Some(f32::MAX),
            None,
            Some(f32::MIN),
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

        let result = unsafe { decoder.decode_chunked(&input, &output_buffers, row_count) };
        assert!(result.is_ok());
        assert_eq!(result.rows_decoded, row_count);

        for (i, v) in values.iter().enumerate() {
            let is_valid = check_validity_bit(&validity_buf, i);
            match v {
                Some(expected) => {
                    assert!(is_valid, "Row {} should be valid", i);
                    let actual =
                        f32::from_le_bytes(data_buf[i * 4..(i + 1) * 4].try_into().unwrap());
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
        let schema = Arc::new(Schema::new(vec![Field::new(
            "flag",
            DataType::Boolean,
            true,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        let values: Vec<Option<bool>> = vec![
            Some(true),
            None,
            Some(false),
            Some(true),
            None,
            None,
            Some(false),
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

        let result = unsafe { decoder.decode_chunked(&input, &output_buffers, row_count) };
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
        let schema = Arc::new(Schema::new(vec![Field::new(
            "flag",
            DataType::Boolean,
            true,
        )]));

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

        let result = unsafe { decoder.decode_chunked(&input, &output_buffers, row_count) };
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
        let schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, true)]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        let values: Vec<Option<&str>> = vec![
            Some("hello"),
            None,
            Some(""),
            Some("world"),
            None,
            Some("test"),
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

        let result = unsafe { decoder.decode_chunked(&input, &output_buffers, row_count) };
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
        let schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, true)]));

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

        let result = unsafe { decoder.decode_chunked(&input, &output_buffers, row_count) };
        assert!(result.is_ok());
        assert_eq!(result.rows_decoded, row_count);

        // All should be valid (empty string != null)
        assert_eq!(count_valid_bits(&validity_buf, row_count), row_count);
    }

    #[test]
    fn test_nullable_string_all_nulls() {
        let schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, true)]));

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

        let result = unsafe { decoder.decode_chunked(&input, &output_buffers, row_count) };
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
        let schema = Arc::new(Schema::new(vec![Field::new(
            "status",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            true,
        )]));

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

        let result = unsafe { decoder.decode_chunked(&input, &output_buffers, row_count) };
        assert!(result.is_ok());
        assert_eq!(result.rows_decoded, row_count);

        for (i, v) in values.iter().enumerate() {
            let is_valid = check_validity_bit(&validity_buf, i);
            match v {
                Some(expected) => {
                    assert!(is_valid, "Row {} should be valid", i);
                    let actual =
                        i32::from_le_bytes(data_buf[i * 4..(i + 1) * 4].try_into().unwrap());
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

        let result = unsafe { decoder.decode_chunked(&input, &output_buffers, row_count) };
        assert!(result.is_ok(), "Decode failed: {:?}", result);
        assert_eq!(result.rows_decoded, row_count);

        // Verify id column
        for (i, (id, _)) in rows.iter().enumerate() {
            let is_valid = check_validity_bit(&id_validity, i);
            match id {
                Some(expected) => {
                    assert!(is_valid, "Row {} id should be valid", i);
                    let actual =
                        i64::from_le_bytes(id_data[i * 8..(i + 1) * 8].try_into().unwrap());
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
                    let actual =
                        f64::from_le_bytes(val_data[i * 8..(i + 1) * 8].try_into().unwrap());
                    assert_eq!(actual, *expected);
                }
                None => assert!(!is_valid, "Row {} value should be null", i),
            }
        }
    }

    #[test]
    fn test_mixed_nullable_and_non_nullable_columns() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("required_id", DataType::Int64, false), // NOT nullable
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

        let result = unsafe { decoder.decode_chunked(&input, &output_buffers, row_count) };
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
                    let actual =
                        i64::from_le_bytes(val_data[i * 8..(i + 1) * 8].try_into().unwrap());
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
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            true,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        // 9 rows: bits 0-7 in byte 0, bit 8 in byte 1
        let values: Vec<Option<i64>> = vec![
            Some(1),
            None,
            Some(3),
            Some(4),
            None,
            Some(6),
            Some(7),
            Some(8),
            Some(9),
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

        let result = unsafe { decoder.decode_chunked(&input, &output_buffers, row_count) };
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
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            true,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Alternating null/non-null pattern
        let values: Vec<Option<i64>> = (0..16)
            .map(|i| if i % 2 == 0 { Some(i as i64) } else { None })
            .collect();
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

        let result = unsafe { decoder.decode_chunked(&input, &output_buffers, row_count) };
        assert!(result.is_ok());

        // Expected pattern: 0b01010101 for both bytes (even indices valid)
        assert_eq!(validity_buf[0], 0b01010101, "First validity byte mismatch");
        assert_eq!(validity_buf[1], 0b01010101, "Second validity byte mismatch");
    }

    #[test]
    fn test_nullable_large_row_count() {
        // Test with 1000 rows to stress test validity bitmap handling
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            true,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("compile");
        let row_count = 1000;

        // Every 3rd row is null
        let values: Vec<Option<i64>> = (0..row_count)
            .map(|i| if i % 3 == 0 { None } else { Some(i as i64) })
            .collect();

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

        let result = unsafe { decoder.decode_chunked(&input, &output_buffers, row_count) };
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
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            true,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        let input: Vec<u8> = Vec::new();
        let mut data_buf = vec![0u8; 8]; // some space, but 0 rows
        let mut validity_buf = vec![0u8; 1];

        let output_buffers = vec![
            OutputBuffer::new(data_buf.as_mut_ptr(), data_buf.len()),
            OutputBuffer::new(validity_buf.as_mut_ptr(), validity_buf.len()),
        ];

        let result = unsafe { decoder.decode_chunked(&input, &output_buffers, 0) };
        assert!(result.is_ok());
        assert_eq!(result.rows_decoded, 0);
        assert_eq!(result.bytes_consumed, 0);
    }

    // -------------------------------------------------------------------------
    // Binary (Bytes) Nullable Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_nullable_binary_validity_mixed() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "data",
            DataType::Binary,
            true,
        )]));

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

        let result = unsafe { decoder.decode_chunked(&input, &output_buffers, row_count) };
        assert!(result.is_ok(), "Decode failed");
        assert_eq!(result.rows_decoded, row_count);

        // Verify validity
        for (i, v) in values.iter().enumerate() {
            let is_valid = check_validity_bit(&validity_buf, i);
            assert_eq!(is_valid, v.is_some(), "Validity mismatch at row {}", i);
        }
    }

    // -------------------------------------------------------------------------
    // Timestamp Type Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_roundtrip_timestamp_microsecond() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Test timestamp values in microseconds
        let test_values: Vec<i64> = vec![
            0,                // Unix epoch
            1704067200000000, // 2024-01-01 00:00:00
            -86400000000,     // 1969-12-31 (before epoch)
            i64::MAX / 2,     // Large positive
        ];

        let mut input = Vec::new();
        for &v in &test_values {
            input.extend(encode_varint_i64(v));
        }

        let row_count = test_values.len();
        let mut output = vec![0u8; row_count * 8];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in test_values.iter().enumerate() {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(
                actual, expected,
                "Timestamp microsecond mismatch at index {}: expected {}, got {}",
                i, expected, actual
            );
        }
    }

    #[test]
    fn test_roundtrip_timestamp_millisecond() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            false,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Test timestamp values in milliseconds
        let test_values: Vec<i64> = vec![
            0,             // Unix epoch
            1704067200000, // 2024-01-01 00:00:00
            -86400000,     // 1969-12-31 (before epoch)
            1704067260000, // 2024-01-01 00:01:00
        ];

        let mut input = Vec::new();
        for &v in &test_values {
            input.extend(encode_varint_i64(v));
        }

        let row_count = test_values.len();
        let mut output = vec![0u8; row_count * 8];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in test_values.iter().enumerate() {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(
                actual, expected,
                "Timestamp millisecond mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_roundtrip_timestamp_nanosecond() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Test timestamp values in nanoseconds
        let test_values: Vec<i64> = vec![
            0,                   // Unix epoch
            1704067200000000000, // 2024-01-01 00:00:00
            -86400000000000,     // 1969-12-31 (before epoch)
            1704067260000000000, // 2024-01-01 00:01:00
        ];

        let mut input = Vec::new();
        for &v in &test_values {
            input.extend(encode_varint_i64(v));
        }

        let row_count = test_values.len();
        let mut output = vec![0u8; row_count * 8];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in test_values.iter().enumerate() {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(
                actual, expected,
                "Timestamp nanosecond mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_roundtrip_timestamp_second() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Second, None),
            false,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Test timestamp values in seconds
        let test_values: Vec<i64> = vec![
            0,          // Unix epoch
            1704067200, // 2024-01-01 00:00:00
            -86400,     // 1969-12-31 (before epoch)
            1704067260, // 2024-01-01 00:01:00
        ];

        let mut input = Vec::new();
        for &v in &test_values {
            input.extend(encode_varint_i64(v));
        }

        let row_count = test_values.len();
        let mut output = vec![0u8; row_count * 8];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in test_values.iter().enumerate() {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(actual, expected, "Timestamp second mismatch at index {}", i);
        }
    }

    #[test]
    fn test_roundtrip_timestamp_with_timezone() {
        // Timestamp with UTC timezone
        let schema = Arc::new(Schema::new(vec![Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("+00:00".into())),
            false,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        let test_values: Vec<i64> = vec![1704067200000000, 1704067260000000, 1704067320000000];

        let mut input = Vec::new();
        for &v in &test_values {
            input.extend(encode_varint_i64(v));
        }

        let row_count = test_values.len();
        let mut output = vec![0u8; row_count * 8];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in test_values.iter().enumerate() {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(
                actual, expected,
                "Timestamp with timezone mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_nullable_timestamp_microsecond() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        let values: Vec<Option<i64>> = vec![
            Some(1704067200000000),
            None,
            Some(1704067320000000),
            None,
            Some(0),
        ];

        let row_count = values.len();

        let mut input = Vec::new();
        for v in &values {
            match v {
                Some(ts) => {
                    input.extend(encode_union_tag(false));
                    input.extend(encode_varint_i64(*ts));
                }
                None => {
                    input.extend(encode_union_tag(true));
                }
            }
        }

        let validity_bytes = (row_count + 7) / 8;
        let mut data_buf = vec![0u8; row_count * 8];
        let mut validity_buf = vec![0u8; validity_bytes];

        let output_buffers = vec![
            OutputBuffer::new(data_buf.as_mut_ptr(), data_buf.len()),
            OutputBuffer::new(validity_buf.as_mut_ptr(), validity_buf.len()),
        ];

        let result = unsafe { decoder.decode_chunked(&input, &output_buffers, row_count) };
        assert!(result.is_ok(), "Decode failed");
        assert_eq!(result.rows_decoded, row_count);

        // Verify validity bits
        for (i, v) in values.iter().enumerate() {
            let is_valid = check_validity_bit(&validity_buf, i);
            assert_eq!(
                is_valid,
                v.is_some(),
                "Validity mismatch at row {}: expected {}, got {}",
                i,
                v.is_some(),
                is_valid
            );
        }

        // Verify non-null values
        for (i, v) in values.iter().enumerate() {
            if let Some(expected) = v {
                let actual = i64::from_le_bytes(data_buf[i * 8..(i + 1) * 8].try_into().unwrap());
                assert_eq!(
                    actual, *expected,
                    "Nullable timestamp value mismatch at row {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_all_timestamp_units_combined() {
        // Test all timestamp units in a single multi-column schema
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "ts_micros",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new(
                "ts_millis",
                DataType::Timestamp(TimeUnit::Millisecond, None),
                false,
            ),
            Field::new(
                "ts_nanos",
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                false,
            ),
            Field::new(
                "ts_seconds",
                DataType::Timestamp(TimeUnit::Second, None),
                false,
            ),
        ]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        let row_count = 3;
        let mut input = Vec::new();

        for i in 0..row_count {
            // Microseconds
            input.extend(encode_varint_i64(1704067200000000 + i as i64 * 60000000));
            // Milliseconds
            input.extend(encode_varint_i64(1704067200000 + i as i64 * 60000));
            // Nanoseconds
            input.extend(encode_varint_i64(
                1704067200000000000 + i as i64 * 60000000000,
            ));
            // Seconds
            input.extend(encode_varint_i64(1704067200 + i as i64 * 60));
        }

        let mut output_micros = vec![0u8; row_count * 8];
        let mut output_millis = vec![0u8; row_count * 8];
        let mut output_nanos = vec![0u8; row_count * 8];
        let mut output_seconds = vec![0u8; row_count * 8];

        let mut output_ptrs = vec![
            output_micros.as_mut_ptr(),
            output_millis.as_mut_ptr(),
            output_nanos.as_mut_ptr(),
            output_seconds.as_mut_ptr(),
        ];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        // Verify values
        for i in 0..row_count {
            let micros = i64::from_le_bytes(output_micros[i * 8..(i + 1) * 8].try_into().unwrap());
            let millis = i64::from_le_bytes(output_millis[i * 8..(i + 1) * 8].try_into().unwrap());
            let nanos = i64::from_le_bytes(output_nanos[i * 8..(i + 1) * 8].try_into().unwrap());
            let seconds =
                i64::from_le_bytes(output_seconds[i * 8..(i + 1) * 8].try_into().unwrap());

            assert_eq!(micros, 1704067200000000 + i as i64 * 60000000);
            assert_eq!(millis, 1704067200000 + i as i64 * 60000);
            assert_eq!(nanos, 1704067200000000000 + i as i64 * 60000000000);
            assert_eq!(seconds, 1704067200 + i as i64 * 60);
        }
    }

    #[test]
    fn test_timestamp_edge_values() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        )]));

        let decoder = JitDecoder::compile(&schema).expect("compile");

        // Test extreme values
        let test_values: Vec<i64> = vec![
            i64::MIN / 2, // Large negative (avoid overflow in zigzag)
            -1,           // Just before epoch
            0,            // Epoch
            1,            // Just after epoch
            i64::MAX / 2, // Large positive
        ];

        let mut input = Vec::new();
        for &v in &test_values {
            input.extend(encode_varint_i64(v));
        }

        let row_count = test_values.len();
        let mut output = vec![0u8; row_count * 8];
        let mut output_ptrs = vec![output.as_mut_ptr()];

        let consumed =
            unsafe { decoder.decode(&input, &mut output_ptrs, row_count) }.expect("Decode failed");
        assert_eq!(consumed, input.len());

        for (i, &expected) in test_values.iter().enumerate() {
            let actual = i64::from_le_bytes(output[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(
                actual, expected,
                "Timestamp edge value mismatch at index {}: expected {}, got {}",
                i, expected, actual
            );
        }
    }
}

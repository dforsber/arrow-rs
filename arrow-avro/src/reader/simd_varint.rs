//! SIMD-optimized varint decoding for Avro
//!
//! This module provides platform-specific SIMD implementations for fast
//! batch varint decoding. The JIT decoder calls these functions for
//! int-heavy schemas where varint decoding is the bottleneck.
//!
//! # Supported Platforms
//! - x86_64: SSE2/SSE4.1/AVX2 (auto-detected at runtime)
//! - aarch64: NEON
//! - Fallback: Scalar implementation for other platforms
//!
//! # Design
//! Instead of decoding one varint at a time, we scan ahead to find varint
//! boundaries using SIMD, then batch-decode multiple values.

/// Result of batch varint decoding
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BatchDecodeResult {
    /// Number of values successfully decoded
    pub count: usize,
    /// Number of bytes consumed from input
    pub bytes_consumed: usize,
}

/// Batch decode signed varints (zigzag encoded) into i64 array
///
/// # Safety
/// - `input` must point to at least `input_len` valid bytes
/// - `output` must point to at least `max_values` i64 values
/// - Caller must ensure output buffer has sufficient capacity
#[unsafe(no_mangle)]
pub unsafe extern "C" fn simd_decode_varints_i64(
    input: *const u8,
    input_len: usize,
    output: *mut i64,
    max_values: usize,
) -> BatchDecodeResult {
    if input_len == 0 || max_values == 0 {
        return BatchDecodeResult {
            count: 0,
            bytes_consumed: 0,
        };
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            return unsafe { decode_varints_x86_64_sse2(input, input_len, output, max_values) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { decode_varints_aarch64_neon(input, input_len, output, max_values) };
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return unsafe { decode_varints_scalar(input, input_len, output, max_values) };
    }

    // Fallback for x86_64 without SSE2 (unlikely)
    #[cfg(target_arch = "x86_64")]
    unsafe { decode_varints_scalar(input, input_len, output, max_values) }
}

/// Batch decode unsigned varints into i32 array (for array lengths, enum indices)
///
/// # Safety
/// Same requirements as simd_decode_varints_i64
#[unsafe(no_mangle)]
pub unsafe extern "C" fn simd_decode_varints_i32(
    input: *const u8,
    input_len: usize,
    output: *mut i32,
    max_values: usize,
) -> BatchDecodeResult {
    if input_len == 0 || max_values == 0 {
        return BatchDecodeResult {
            count: 0,
            bytes_consumed: 0,
        };
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            return unsafe { decode_varints_i32_x86_64_sse2(input, input_len, output, max_values) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { decode_varints_i32_aarch64_neon(input, input_len, output, max_values) };
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return unsafe { decode_varints_i32_scalar(input, input_len, output, max_values) };
    }

    #[cfg(target_arch = "x86_64")]
    unsafe { decode_varints_i32_scalar(input, input_len, output, max_values) }
}

// ============================================================================
// Common Helper Functions
// ============================================================================

/// Decode a single varint from ptr, return (value, bytes_consumed)
#[inline]
unsafe fn decode_one_varint(ptr: *const u8, remaining: usize) -> Option<(u64, usize)> {
    unsafe {
        let first = *ptr;
        if first < 0x80 {
            return Some((first as u64, 1));
        }

        let mut value = (first & 0x7F) as u64;
        let mut shift = 7;
        let mut pos = 1;

        while pos < remaining.min(10) {
            let byte = *ptr.add(pos);
            value |= ((byte & 0x7F) as u64) << shift;
            pos += 1;
            if byte < 0x80 {
                return Some((value, pos));
            }
            shift += 7;
        }
        None
    }
}

/// Zigzag decode
#[inline]
fn zigzag_decode(n: u64) -> i64 {
    ((n >> 1) as i64) ^ (-((n & 1) as i64))
}

// ============================================================================
// x86_64 SSE2 Implementation
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn decode_varints_x86_64_sse2(
    input: *const u8,
    input_len: usize,
    output: *mut i64,
    max_values: usize,
) -> BatchDecodeResult {
    use std::arch::x86_64::*;

    unsafe {
        let mut cursor = 0usize;
        let mut count = 0usize;

        // Process in 16-byte chunks
        while cursor + 16 <= input_len && count < max_values {
            let chunk = _mm_loadu_si128(input.add(cursor) as *const __m128i);
            // Get high bits - these indicate continuation bytes
            let high_bits = _mm_movemask_epi8(chunk) as u32;
            // Invert: 0 means end of varint
            let ends_mask = (!high_bits) & 0xFFFF;

            if ends_mask == 0 {
                // All continuation bytes - decode one long varint
                if let Some((val, consumed)) = decode_one_varint(input.add(cursor), input_len - cursor) {
                    *output.add(count) = zigzag_decode(val);
                    count += 1;
                    cursor += consumed;
                } else {
                    break;
                }
            } else {
                // Fast path for 1-byte varints
                let first_end = ends_mask.trailing_zeros() as usize;

                if first_end == 0 {
                    // First byte is a complete 1-byte varint
                    let val = *input.add(cursor) as u64;
                    *output.add(count) = zigzag_decode(val);
                    count += 1;
                    cursor += 1;

                    // Try to batch more 1-byte varints
                    let mut remaining_mask = ends_mask >> 1;
                    while remaining_mask & 1 != 0 && count < max_values && cursor < input_len {
                        let val = *input.add(cursor) as u64;
                        *output.add(count) = zigzag_decode(val);
                        count += 1;
                        cursor += 1;
                        remaining_mask >>= 1;
                    }
                } else {
                    // Multi-byte varint
                    if let Some((val, consumed)) = decode_one_varint(input.add(cursor), input_len - cursor) {
                        *output.add(count) = zigzag_decode(val);
                        count += 1;
                        cursor += consumed;
                    } else {
                        break;
                    }
                }
            }
        }

        // Handle tail
        while cursor < input_len && count < max_values {
            if let Some((val, consumed)) = decode_one_varint(input.add(cursor), input_len - cursor) {
                *output.add(count) = zigzag_decode(val);
                count += 1;
                cursor += consumed;
            } else {
                break;
            }
        }

        BatchDecodeResult { count, bytes_consumed: cursor }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn decode_varints_i32_x86_64_sse2(
    input: *const u8,
    input_len: usize,
    output: *mut i32,
    max_values: usize,
) -> BatchDecodeResult {
    use std::arch::x86_64::*;

    unsafe {
        let mut cursor = 0usize;
        let mut count = 0usize;

        // Fast path: batch 16 single-byte varints at once
        while cursor + 16 <= input_len && count + 16 <= max_values {
            let chunk = _mm_loadu_si128(input.add(cursor) as *const __m128i);
            let high_bits = _mm_movemask_epi8(chunk) as u32;

            if high_bits == 0 {
                // All 16 are single-byte varints - fast copy
                for i in 0..16 {
                    *output.add(count + i) = *input.add(cursor + i) as i32;
                }
                count += 16;
                cursor += 16;
                continue;
            }
            break;
        }

        // Scalar fallback
        while cursor < input_len && count < max_values {
            let first = *input.add(cursor);
            if first < 0x80 {
                *output.add(count) = first as i32;
                count += 1;
                cursor += 1;
            } else if let Some((val, consumed)) = decode_one_varint(input.add(cursor), input_len - cursor) {
                *output.add(count) = val as i32;
                count += 1;
                cursor += consumed;
            } else {
                break;
            }
        }

        BatchDecodeResult { count, bytes_consumed: cursor }
    }
}

// ============================================================================
// aarch64 NEON Implementation
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn decode_varints_aarch64_neon(
    input: *const u8,
    input_len: usize,
    output: *mut i64,
    max_values: usize,
) -> BatchDecodeResult {
    use std::arch::aarch64::*;

    unsafe {
        let mut cursor = 0usize;
        let mut count = 0usize;

        // Process 16 bytes at a time
        while cursor + 16 <= input_len && count < max_values {
            let chunk = vld1q_u8(input.add(cursor));
            // Check for single-byte varints (high bit = 0)
            let high_bits = vshrq_n_u8(chunk, 7);
            let max_val = vmaxvq_u8(high_bits);

            if max_val == 0 {
                // All 16 bytes are 1-byte varints
                let batch_count = 16.min(max_values - count);
                for i in 0..batch_count {
                    let val = *input.add(cursor + i) as u64;
                    *output.add(count + i) = zigzag_decode(val);
                }
                count += batch_count;
                cursor += batch_count;
                continue;
            }

            // Mixed - decode one at a time
            if let Some((val, consumed)) = decode_one_varint(input.add(cursor), input_len - cursor) {
                *output.add(count) = zigzag_decode(val);
                count += 1;
                cursor += consumed;
            } else {
                break;
            }
        }

        // Tail
        while cursor < input_len && count < max_values {
            if let Some((val, consumed)) = decode_one_varint(input.add(cursor), input_len - cursor) {
                *output.add(count) = zigzag_decode(val);
                count += 1;
                cursor += consumed;
            } else {
                break;
            }
        }

        BatchDecodeResult { count, bytes_consumed: cursor }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn decode_varints_i32_aarch64_neon(
    input: *const u8,
    input_len: usize,
    output: *mut i32,
    max_values: usize,
) -> BatchDecodeResult {
    use std::arch::aarch64::*;

    unsafe {
        let mut cursor = 0usize;
        let mut count = 0usize;

        // Fast path for blocks of 16 single-byte varints
        while cursor + 16 <= input_len && count + 16 <= max_values {
            let chunk = vld1q_u8(input.add(cursor));
            let high_bits = vshrq_n_u8(chunk, 7);
            let max_val = vmaxvq_u8(high_bits);

            if max_val == 0 {
                for i in 0..16 {
                    *output.add(count + i) = *input.add(cursor + i) as i32;
                }
                count += 16;
                cursor += 16;
                continue;
            }
            break;
        }

        // Scalar fallback
        while cursor < input_len && count < max_values {
            let first = *input.add(cursor);
            if first < 0x80 {
                *output.add(count) = first as i32;
                count += 1;
                cursor += 1;
            } else if let Some((val, consumed)) = decode_one_varint(input.add(cursor), input_len - cursor) {
                *output.add(count) = val as i32;
                count += 1;
                cursor += consumed;
            } else {
                break;
            }
        }

        BatchDecodeResult { count, bytes_consumed: cursor }
    }
}

// ============================================================================
// Scalar Fallback
// ============================================================================

#[allow(dead_code)]
unsafe fn decode_varints_scalar(
    input: *const u8,
    input_len: usize,
    output: *mut i64,
    max_values: usize,
) -> BatchDecodeResult {
    unsafe {
        let mut cursor = 0usize;
        let mut count = 0usize;

        while cursor < input_len && count < max_values {
            if let Some((val, consumed)) = decode_one_varint(input.add(cursor), input_len - cursor) {
                *output.add(count) = zigzag_decode(val);
                count += 1;
                cursor += consumed;
            } else {
                break;
            }
        }

        BatchDecodeResult { count, bytes_consumed: cursor }
    }
}

#[allow(dead_code)]
unsafe fn decode_varints_i32_scalar(
    input: *const u8,
    input_len: usize,
    output: *mut i32,
    max_values: usize,
) -> BatchDecodeResult {
    unsafe {
        let mut cursor = 0usize;
        let mut count = 0usize;

        while cursor < input_len && count < max_values {
            let first = *input.add(cursor);
            if first < 0x80 {
                *output.add(count) = first as i32;
                count += 1;
                cursor += 1;
            } else if let Some((val, consumed)) = decode_one_varint(input.add(cursor), input_len - cursor) {
                *output.add(count) = val as i32;
                count += 1;
                cursor += consumed;
            } else {
                break;
            }
        }

        BatchDecodeResult { count, bytes_consumed: cursor }
    }
}

// ============================================================================
// Function Pointer Getters for JIT Integration
// ============================================================================

/// Get the batch decode function pointer for i64 varints
#[unsafe(no_mangle)]
pub extern "C" fn get_simd_varint_i64_decoder(
) -> unsafe extern "C" fn(*const u8, usize, *mut i64, usize) -> BatchDecodeResult {
    simd_decode_varints_i64
}

/// Get the batch decode function pointer for i32 varints
#[unsafe(no_mangle)]
pub extern "C" fn get_simd_varint_i32_decoder(
) -> unsafe extern "C" fn(*const u8, usize, *mut i32, usize) -> BatchDecodeResult {
    simd_decode_varints_i32
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_varint(mut n: u64, dst: &mut Vec<u8>) {
        while n >= 0x80 {
            dst.push(0x80 | (n as u8));
            n >>= 7;
        }
        dst.push(n as u8);
    }

    fn zigzag_encode(n: i64) -> u64 {
        ((n << 1) ^ (n >> 63)) as u64
    }

    #[test]
    fn test_single_byte_varints() {
        let mut input = Vec::new();
        let expected: Vec<i64> = (0..64i64).collect();

        for &val in &expected {
            encode_varint(zigzag_encode(val), &mut input);
        }

        let mut output = vec![0i64; 100];
        let result = unsafe {
            simd_decode_varints_i64(
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                100,
            )
        };

        assert_eq!(result.count, expected.len());
        assert_eq!(result.bytes_consumed, input.len());
        assert_eq!(&output[..expected.len()], &expected[..]);
    }

    #[test]
    fn test_multi_byte_varints() {
        let mut input = Vec::new();
        let expected: Vec<i64> = vec![0, 127, 128, 255, 256, 16383, 16384, 1000000, -1, -128, -1000000];

        for &val in &expected {
            encode_varint(zigzag_encode(val), &mut input);
        }

        let mut output = vec![0i64; 100];
        let result = unsafe {
            simd_decode_varints_i64(
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                100,
            )
        };

        assert_eq!(result.count, expected.len());
        assert_eq!(&output[..expected.len()], &expected[..]);
    }

    #[test]
    fn test_i32_batch_decode() {
        let input: Vec<u8> = (0..32u8).collect();
        let mut output = vec![0i32; 50];

        let result = unsafe {
            simd_decode_varints_i32(
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                50,
            )
        };

        assert_eq!(result.count, 32);
        assert_eq!(result.bytes_consumed, 32);
        for i in 0..32 {
            assert_eq!(output[i], i as i32);
        }
    }

    #[test]
    fn test_mixed_length_varints() {
        let mut input = Vec::new();
        let expected: Vec<i64> = vec![
            1, 2, 3,           // Single byte
            200,               // 2 bytes
            4, 5, 6,           // Single byte
            50000,             // 3 bytes
            7, 8, 9, 10,       // Single byte
        ];

        for &val in &expected {
            encode_varint(zigzag_encode(val), &mut input);
        }

        let mut output = vec![0i64; 100];
        let result = unsafe {
            simd_decode_varints_i64(
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                100,
            )
        };

        assert_eq!(result.count, expected.len());
        assert_eq!(&output[..expected.len()], &expected[..]);
    }

    #[test]
    fn test_empty_input() {
        let mut output = vec![0i64; 10];
        let result = unsafe {
            simd_decode_varints_i64(
                [].as_ptr(),
                0,
                output.as_mut_ptr(),
                10,
            )
        };

        assert_eq!(result.count, 0);
        assert_eq!(result.bytes_consumed, 0);
    }

    #[test]
    fn test_large_batch() {
        let mut input = Vec::new();
        let expected: Vec<i64> = (0..10000i64).collect();

        for &val in &expected {
            encode_varint(zigzag_encode(val), &mut input);
        }

        let mut output = vec![0i64; 10000];
        let result = unsafe {
            simd_decode_varints_i64(
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                10000,
            )
        };

        assert_eq!(result.count, expected.len());
        assert_eq!(&output[..], &expected[..]);
    }
}

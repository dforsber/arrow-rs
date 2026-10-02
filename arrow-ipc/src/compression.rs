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

use crate::CompressionType;
use arrow_buffer::Buffer;
use arrow_schema::ArrowError;

const LENGTH_NO_COMPRESSED_DATA: i64 = -1;
const LENGTH_OF_PREFIX_DATA: i64 = 8;

/// Additional context that may be needed for compression.
///
/// In the case of zstd, this will contain the zstd context, which can be reused between subsequent
/// compression calls to avoid the performance overhead of initialising a new context for every
/// compression.
pub struct CompressionContext {
    #[cfg(feature = "zstd")]
    compressor: zstd::bulk::Compressor<'static>,
}

// the reason we allow derivable_impls here is because when zstd feature is not enabled, this
// becomes derivable. however with zstd feature want to be explicit about the compression level.
#[allow(clippy::derivable_impls)]
impl Default for CompressionContext {
    fn default() -> Self {
        CompressionContext {
            // safety: `new` here will only return error here if using an invalid compression level
            #[cfg(feature = "zstd")]
            compressor: zstd::bulk::Compressor::new(zstd::DEFAULT_COMPRESSION_LEVEL)
                .expect("can use default compression level"),
        }
    }
}

impl std::fmt::Debug for CompressionContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("CompressionContext");

        #[cfg(feature = "zstd")]
        ds.field("compressor", &"zstd::bulk::Compressor");

        ds.finish()
    }
}

/// Represents compressing a ipc stream using a particular compression algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionCodec {
    Lz4Frame,
    Zstd,
}

impl TryFrom<CompressionType> for CompressionCodec {
    type Error = ArrowError;

    fn try_from(compression_type: CompressionType) -> Result<Self, ArrowError> {
        match compression_type {
            CompressionType::ZSTD => Ok(CompressionCodec::Zstd),
            CompressionType::LZ4_FRAME => Ok(CompressionCodec::Lz4Frame),
            other_type => Err(ArrowError::NotYetImplemented(format!(
                "compression type {other_type:?} not supported "
            ))),
        }
    }
}

impl CompressionCodec {
    /// Compresses the data in `input` to `output` and appends the
    /// data using the specified compression mechanism.
    ///
    /// returns the number of bytes written to the stream
    ///
    /// Writes this format to output:
    /// ```text
    /// [8 bytes]:         uncompressed length
    /// [remaining bytes]: compressed data stream
    /// ```
    pub(crate) fn compress_to_vec(
        &self,
        input: &[u8],
        output: &mut Vec<u8>,
        context: &mut CompressionContext,
    ) -> Result<usize, ArrowError> {
        let uncompressed_data_len = input.len();
        let original_output_len = output.len();

        if input.is_empty() {
            // empty input, nothing to do
        } else {
            // write compressed data directly into the output buffer
            output.extend_from_slice(&uncompressed_data_len.to_le_bytes());
            self.compress(input, output, context)?;

            let compression_len = output.len() - original_output_len;
            if compression_len > uncompressed_data_len {
                // length of compressed data was larger than
                // uncompressed data, use the uncompressed data with
                // length -1 to indicate that we don't compress the
                // data
                output.truncate(original_output_len);
                output.extend_from_slice(&LENGTH_NO_COMPRESSED_DATA.to_le_bytes());
                output.extend_from_slice(input);
            }
        }
        Ok(output.len() - original_output_len)
    }

    /// Decompresses the input into a [`Buffer`]
    ///
    /// The input should look like:
    /// ```text
    /// [8 bytes]:         uncompressed length
    /// [remaining bytes]: compressed data stream
    /// ```
    #[cfg(test)]
    pub(crate) fn decompress_to_buffer(&self, input: &Buffer) -> Result<Buffer, ArrowError> {
        self.decompress_to_buffer_with_limit(input, usize::MAX)
    }

    /// Decode one IPC buffer within the caller's remaining decoded-byte budget.
    ///
    /// This bounds output bytes, not codec-internal history windows or the total
    /// memory held by other buffers. Callers enforce aggregate admission separately.
    pub(crate) fn decompress_to_buffer_with_limit(
        &self,
        input: &Buffer,
        max_output_bytes: usize,
    ) -> Result<Buffer, ArrowError> {
        let decompressed_length = read_uncompressed_size(input)?;
        let input_data = &input[(LENGTH_OF_PREFIX_DATA as usize)..];
        if decompressed_length == LENGTH_NO_COMPRESSED_DATA {
            if input_data.len() > max_output_bytes {
                return Err(ArrowError::IpcError(format!(
                    "Uncompressed IPC buffer length {} exceeds decoded-byte limit {max_output_bytes}",
                    input_data.len()
                )));
            }
            return Ok(input.slice(LENGTH_OF_PREFIX_DATA as usize));
        }
        let decompressed_length = usize::try_from(decompressed_length).map_err(|_| {
            ArrowError::IpcError(format!(
                "Invalid uncompressed length: {decompressed_length}"
            ))
        })?;
        if decompressed_length > max_output_bytes {
            return Err(ArrowError::IpcError(format!(
                "Uncompressed IPC buffer length {decompressed_length} exceeds decoded-byte limit {max_output_bytes}"
            )));
        }
        if decompressed_length == 0 {
            return Ok(Buffer::from([]));
        }
        self.decompress(input_data, decompressed_length)
            .map(Buffer::from_vec)
    }

    /// Compress the data in input buffer and write to output buffer
    /// using the specified compression
    fn compress(
        &self,
        input: &[u8],
        output: &mut Vec<u8>,
        context: &mut CompressionContext,
    ) -> Result<(), ArrowError> {
        match self {
            CompressionCodec::Lz4Frame => compress_lz4(input, output),
            CompressionCodec::Zstd => compress_zstd(input, output, context),
        }
    }

    /// Decompress the data in input buffer and write to output buffer
    /// using the specified compression
    fn decompress(&self, input: &[u8], decompressed_size: usize) -> Result<Vec<u8>, ArrowError> {
        let ret = match self {
            CompressionCodec::Lz4Frame => decompress_lz4(input, decompressed_size)?,
            CompressionCodec::Zstd => decompress_zstd(input, decompressed_size)?,
        };
        if ret.len() != decompressed_size {
            return Err(ArrowError::IpcError(format!(
                "Expected decompressed length of {decompressed_size} got {}",
                ret.len()
            )));
        }
        Ok(ret)
    }
}

#[cfg(feature = "lz4")]
fn compress_lz4(input: &[u8], output: &mut Vec<u8>) -> Result<(), ArrowError> {
    use std::io::Write;
    let mut encoder = lz4_flex::frame::FrameEncoder::new(output);
    encoder.write_all(input)?;
    encoder
        .finish()
        .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
    Ok(())
}

#[cfg(not(feature = "lz4"))]
#[allow(clippy::ptr_arg)]
fn compress_lz4(_input: &[u8], _output: &mut Vec<u8>) -> Result<(), ArrowError> {
    Err(ArrowError::InvalidArgumentError(
        "lz4 IPC compression requires the lz4 feature".to_string(),
    ))
}

#[cfg(feature = "lz4")]
fn decompress_lz4(input: &[u8], decompressed_size: usize) -> Result<Vec<u8>, ArrowError> {
    read_decompressed(lz4_flex::frame::FrameDecoder::new(input), decompressed_size)
}

#[cfg(not(feature = "lz4"))]
#[allow(clippy::ptr_arg)]
fn decompress_lz4(_input: &[u8], _decompressed_size: usize) -> Result<Vec<u8>, ArrowError> {
    Err(ArrowError::InvalidArgumentError(
        "lz4 IPC decompression requires the lz4 feature".to_string(),
    ))
}

#[cfg(feature = "zstd")]
fn compress_zstd(
    input: &[u8],
    output: &mut Vec<u8>,
    context: &mut CompressionContext,
) -> Result<(), ArrowError> {
    let result = context.compressor.compress(input)?;
    output.extend_from_slice(&result);
    Ok(())
}

#[cfg(not(feature = "zstd"))]
#[allow(clippy::ptr_arg)]
fn compress_zstd(
    _input: &[u8],
    _output: &mut Vec<u8>,
    _context: &mut CompressionContext,
) -> Result<(), ArrowError> {
    Err(ArrowError::InvalidArgumentError(
        "zstd IPC compression requires the zstd feature".to_string(),
    ))
}

#[cfg(feature = "zstd")]
fn decompress_zstd(input: &[u8], decompressed_size: usize) -> Result<Vec<u8>, ArrowError> {
    read_decompressed(zstd::Decoder::with_buffer(input)?, decompressed_size)
}

#[cfg(not(feature = "zstd"))]
#[allow(clippy::ptr_arg)]
fn decompress_zstd(_input: &[u8], _decompressed_size: usize) -> Result<Vec<u8>, ArrowError> {
    Err(ArrowError::InvalidArgumentError(
        "zstd IPC decompression requires the zstd feature".to_string(),
    ))
}

/// Grow from observed output, never directly from an untrusted declared length.
/// The one-byte probe detects an oversized stream without retaining its expansion.
#[cfg(any(feature = "lz4", feature = "zstd"))]
fn read_decompressed(
    mut reader: impl std::io::Read,
    decompressed_size: usize,
) -> Result<Vec<u8>, ArrowError> {
    let mut output = Vec::new();
    let mut scratch = [0_u8; 8192];
    loop {
        let remaining = decompressed_size - output.len();
        let read_limit = remaining.min(scratch.len()).max(1);
        let read = match reader.read(&mut scratch[..read_limit]) {
            Ok(read) => read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.into()),
        };
        if read == 0 {
            return Ok(output);
        }
        if read > remaining {
            return Err(ArrowError::IpcError(format!(
                "Decompressed IPC buffer exceeds declared length {decompressed_size}"
            )));
        }
        let required = output.len() + read;
        if required > output.capacity() {
            let capacity = output
                .capacity()
                .saturating_mul(2)
                .max(required)
                .min(decompressed_size);
            output
                .try_reserve_exact(capacity - output.len())
                .map_err(|error| {
                    ArrowError::MemoryError(format!(
                        "Cannot allocate IPC decompression output: {error}"
                    ))
                })?;
        }
        output.extend_from_slice(&scratch[..read]);
    }
}

/// Get the uncompressed length
/// Notes:
///   LENGTH_NO_COMPRESSED_DATA: indicate that the data that follows is not compressed
///    0: indicate that there is no data
///   positive number: indicate the uncompressed length for the following data
#[inline]
fn read_uncompressed_size(buffer: &[u8]) -> Result<i64, ArrowError> {
    let prefix = buffer.get(..8).ok_or_else(|| {
        ArrowError::IpcError(format!(
            "IPC compression prefix requires 8 bytes, got {}",
            buffer.len()
        ))
    })?;
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(prefix);
    Ok(i64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::{CompressionCodec, LENGTH_NO_COMPRESSED_DATA};
    use arrow_buffer::Buffer;

    #[test]
    fn r6_short_compression_prefix_is_an_error() {
        for len in 0..8 {
            let input = Buffer::from(vec![0_u8; len]);
            assert!(
                CompressionCodec::Lz4Frame
                    .decompress_to_buffer(&input)
                    .is_err()
            );
            assert!(CompressionCodec::Zstd.decompress_to_buffer(&input).is_err());
        }
    }

    #[test]
    fn r6_uncompressed_framing_preserves_zero_copy_and_empty_data() {
        for codec in [CompressionCodec::Lz4Frame, CompressionCodec::Zstd] {
            let mut bytes = LENGTH_NO_COMPRESSED_DATA.to_le_bytes().to_vec();
            bytes.extend_from_slice(b"uncompressed bytes");
            let input = Buffer::from(bytes);
            let result = codec.decompress_to_buffer(&input).unwrap();
            assert_eq!(result.as_slice(), b"uncompressed bytes");
            assert_eq!(result.as_ptr(), input.as_slice()[8..].as_ptr());
            let input = Buffer::from(LENGTH_NO_COMPRESSED_DATA.to_le_bytes().to_vec());
            assert!(codec.decompress_to_buffer(&input).unwrap().is_empty());
            let input = Buffer::from(0_i64.to_le_bytes().to_vec());
            assert!(codec.decompress_to_buffer(&input).unwrap().is_empty());
            let input = Buffer::from((-2_i64).to_le_bytes().to_vec());
            assert!(codec.decompress_to_buffer(&input).is_err());
        }
    }

    #[test]
    fn r6_explicit_output_limit_checks_uncompressed_and_declared_bytes() {
        for codec in [CompressionCodec::Lz4Frame, CompressionCodec::Zstd] {
            let mut input = LENGTH_NO_COMPRESSED_DATA.to_le_bytes().to_vec();
            input.extend_from_slice(b"1234");
            let input = Buffer::from(input);
            assert_eq!(
                codec
                    .decompress_to_buffer_with_limit(&input, 4)
                    .unwrap()
                    .as_slice(),
                b"1234"
            );
            assert!(codec.decompress_to_buffer_with_limit(&input, 3).is_err());
            let input = Buffer::from(i64::MAX.to_le_bytes().to_vec());
            let error = codec
                .decompress_to_buffer_with_limit(&input, 1024)
                .unwrap_err();
            assert!(error.to_string().contains("exceeds decoded-byte limit"));
        }
    }

    #[test]
    #[cfg(any(feature = "lz4", feature = "zstd"))]
    fn r6_stream_expansion_stops_at_declared_length() {
        use std::io::{self, Read};
        struct CountingReader {
            read_bytes: usize,
            interrupted: bool,
        }
        impl Read for CountingReader {
            fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
                if !self.interrupted {
                    self.interrupted = true;
                    return Err(io::ErrorKind::Interrupted.into());
                }
                output.fill(42);
                self.read_bytes += output.len();
                Ok(output.len())
            }
        }
        let mut reader = CountingReader {
            read_bytes: 0,
            interrupted: false,
        };
        assert!(super::read_decompressed(&mut reader, 17).is_err());
        assert_eq!(
            reader.read_bytes, 18,
            "consume only the declaration and one overflow byte"
        );
        struct FailedReader;
        impl Read for FailedReader {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::new(io::ErrorKind::InvalidData, "decoder failed"))
            }
        }
        let error = super::read_decompressed(FailedReader, 17).unwrap_err();
        assert!(error.to_string().contains("decoder failed"));
    }

    #[cfg(any(feature = "lz4", feature = "zstd"))]
    fn check_decompression_boundaries(codec: CompressionCodec) {
        // Large enough to span many incremental reads, with actual codec framing.
        let data = vec![42_u8; 131_073];
        let mut compressed = Vec::new();
        codec
            .compress(&data, &mut compressed, &mut Default::default())
            .unwrap();
        let framed = |declared: i64| {
            let mut bytes = declared.to_le_bytes().to_vec();
            bytes.extend_from_slice(&compressed);
            Buffer::from(bytes)
        };
        assert_eq!(
            codec
                .decompress_to_buffer(&framed(data.len() as i64))
                .unwrap()
                .as_slice(),
            data
        );
        assert_eq!(
            codec
                .decompress_to_buffer_with_limit(&framed(data.len() as i64), data.len())
                .unwrap()
                .as_slice(),
            data
        );
        assert!(
            codec
                .decompress_to_buffer_with_limit(&framed(data.len() as i64), data.len() - 1)
                .is_err()
        );
        assert!(codec.decompress_to_buffer(&framed(1)).is_err());
        assert!(
            codec
                .decompress_to_buffer(&framed(data.len() as i64 - 1))
                .is_err()
        );
        assert!(
            codec
                .decompress_to_buffer(&framed(data.len() as i64 + 1))
                .is_err()
        );
        // A forged length must not become a huge upfront allocation.
        assert!(codec.decompress_to_buffer(&framed(i64::MAX)).is_err());
        let mut corrupt = (data.len() as i64).to_le_bytes().to_vec();
        corrupt.extend_from_slice(&[0xff; 32]);
        assert!(codec.decompress_to_buffer(&Buffer::from(corrupt)).is_err());
        let mut truncated = (data.len() as i64).to_le_bytes().to_vec();
        truncated.extend_from_slice(&compressed[..compressed.len() / 2]);
        assert!(
            codec
                .decompress_to_buffer(&Buffer::from(truncated))
                .is_err()
        );
    }

    #[test]
    #[cfg(feature = "lz4")]
    fn r6_lz4_decompression_boundaries() {
        check_decompression_boundaries(CompressionCodec::Lz4Frame);
    }

    #[test]
    #[cfg(feature = "zstd")]
    fn r6_zstd_decompression_boundaries() {
        check_decompression_boundaries(CompressionCodec::Zstd);
    }

    #[test]
    #[cfg(feature = "lz4")]
    fn test_lz4_compression() {
        let input_bytes = b"hello lz4";
        let codec = super::CompressionCodec::Lz4Frame;
        let mut output_bytes: Vec<u8> = Vec::new();
        codec
            .compress(input_bytes, &mut output_bytes, &mut Default::default())
            .unwrap();
        let result = codec
            .decompress(output_bytes.as_slice(), input_bytes.len())
            .unwrap();
        assert_eq!(input_bytes, result.as_slice());
    }

    #[test]
    #[cfg(feature = "zstd")]
    fn test_zstd_compression() {
        let input_bytes = b"hello zstd";
        let codec = super::CompressionCodec::Zstd;
        let mut output_bytes: Vec<u8> = Vec::new();
        codec
            .compress(input_bytes, &mut output_bytes, &mut Default::default())
            .unwrap();
        let result = codec
            .decompress(output_bytes.as_slice(), input_bytes.len())
            .unwrap();
        assert_eq!(input_bytes, result.as_slice());
    }
}

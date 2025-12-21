// This file is part of zoo_vision.
//
// zoo_vision is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version.
//
// zoo_vision is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with
// zoo_vision. If not, see <https://www.gnu.org/licenses/>.

#include "zoo_vision/video_writer.hpp"

#include "zoo_vision/utils.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}
#include <nlohmann/json.hpp>

#include <iostream>

using namespace std::string_literals;

namespace zoo {
namespace av {
template <auto X> using constant_t = std::integral_constant<std::decay_t<decltype(X)>, X>;

template <class T> struct AVDeleteFunc;

template <class T> using DeleteFunc = void (*)(T **);

template <class T> struct AVDeleter {
  void operator()(T *p) { AVDeleteFunc<T>()(&p); }
};

template <class T> using UniqueAV = std::unique_ptr<T, AVDeleter<T>>;

///////////////////
// Specializations
template <> struct AVDeleteFunc<AVFormatContext> : public constant_t<avformat_free_context> {};
template <> struct AVDeleteFunc<AVCodecContext> : public constant_t<avcodec_free_context> {};
template <> struct AVDeleteFunc<AVPacket> : public constant_t<av_packet_free> {};
template <> struct AVDeleteFunc<AVFrame> : public constant_t<av_frame_free> {};
template <> struct AVDeleteFunc<AVIOContext> : public constant_t<avio_closep> {};

using UniqueAVFormatContext = std::unique_ptr<AVFormatContext, constant_t<avformat_free_context>>;
using UniqueSwsContext = std::unique_ptr<SwsContext, constant_t<sws_freeContext>>;

template <class T> using UniqueAVBuffer = std::unique_ptr<T, constant_t<av_free>>;

///////////////////
// make functions

std::string make_error_string(int averror) {
  char buf[AV_ERROR_MAX_STRING_SIZE];
  ::av_make_error_string(buf, AV_ERROR_MAX_STRING_SIZE, averror);
  return std::string(buf);
}

std::pair<UniqueAVFormatContext, int> make_avformat_output_context(const AVOutputFormat *oformat,
                                                                   const char *format_name, const char *filename) {
  AVFormatContext *ctx;
  int error_value = avformat_alloc_output_context2(&ctx, oformat, format_name, filename);
  return {UniqueAVFormatContext{ctx}, error_value};
}

UniqueAV<AVCodecContext> make_codec_context(const AVCodec *codec) {
  return UniqueAV<AVCodecContext>{::avcodec_alloc_context3(codec)};
}

template <class T> UniqueAVBuffer<T> malloc(size_t size) {
  return UniqueAVBuffer{reinterpret_cast<T *>(av_malloc(size))};
}

UniqueAV<AVFrame> make_frame(int fmt, int width, int height) {
  UniqueAV<AVFrame> frame{av_frame_alloc()};
  if (frame == nullptr) {
    return nullptr;
  }
  frame->format = fmt;
  frame->width = width;
  frame->height = height;
  int res = av_frame_get_buffer(frame.get(), 0);
  CHECK_EQ(res, 0);
  return frame;
}

UniqueAV<AVPacket> packet_alloc() { return UniqueAV<AVPacket>{::av_packet_alloc()}; }

std::pair<UniqueAV<AVIOContext>, int> io_open(const std::string &url, int flags) {
  AVIOContext *ctx;
  int error = ::avio_open(&ctx, url.c_str(), flags);
  return {UniqueAV<AVIOContext>{ctx}, error};
}

} // namespace av
namespace detail {

class VideoWriterImpl {
public:
  VideoWriterImpl(const std::string &filename, Vector2i frameSize, float32_t fps);
  ~VideoWriterImpl();

  void write(const cv::Mat3b &img);

private:
  int64_t frameIndex_ = 0;
  av::UniqueSwsContext swsContext_;
  av::UniqueAVFormatContext outputContext_;
  av::UniqueAV<AVCodecContext> codecContext_;
  AVStream *stream_;
  constexpr static size_t BUFFER_SIZE = 512 * 512 * 3;
  av::UniqueAVBuffer<uint8_t> buffer_;
  av::UniqueAV<AVFrame> frame_;
  av::UniqueAV<AVPacket> packet_;
  av::UniqueAV<AVIOContext> ioCtx_;
};

VideoWriterImpl::VideoWriterImpl(const std::string &filename, Vector2i frameSize, float32_t fps) {

  // const std::vector<int> params{{cv::VIDEOWRITER_PROP_IS_COLOR, 1 /*, cv::VIDEOWRITER_PROP_QUALITY, 0*/}};
  // const auto fourcc = "h264";
  // return writer_.open(std::string(filename), cv::VideoWriter::fourcc(fourcc[0], fourcc[1], fourcc[2], fourcc[3]),
  // fps,
  //                     cv::Size{frameSize[0], frameSize[1]}, params);
  AVDictionary *opt = nullptr;

  swsContext_.reset(sws_getContext(frameSize[0], frameSize[1], AV_PIX_FMT_RGB24, frameSize[0], frameSize[1],
                                   AV_PIX_FMT_YUV420P, 0, nullptr, nullptr, nullptr));
  CHECK_NOT_NULL(swsContext_);

  // Open output file
  constexpr auto CODEC_ID = AV_CODEC_ID_H265;

  AVOutputFormat fmt{};
  fmt.name = "mp4";
  fmt.video_codec = CODEC_ID;
  fmt.audio_codec = AV_CODEC_ID_NONE;
  fmt.subtitle_codec = AV_CODEC_ID_NONE;
  {
    auto [ctx, error_value] = av::make_avformat_output_context(&fmt, nullptr, filename.c_str());
    if (ctx == nullptr) {
      // Could not open output context
      throw std::runtime_error("Could not create video output context: "s + av::make_error_string(error_value));
    }
    outputContext_ = std::move(ctx);
  }
  outputContext_->oformat = &fmt;
  outputContext_->video_codec_id = CODEC_ID;

  // New video stream
  stream_ = avformat_new_stream(outputContext_.get(), nullptr);
  CHECK_NOT_NULL(stream_);
  stream_->index = 0;
  stream_->sample_aspect_ratio = AVRational{frameSize[0], frameSize[1]};
  stream_->time_base = AVRational{static_cast<int>(fps * 1000.f), 1000};
  stream_->codecpar->codec_id = CODEC_ID;
  stream_->codecpar->bit_rate = 4000; // getConfig()["track_bitrate"].get<int>();
  stream_->codecpar->format = AV_PIX_FMT_YUV420P;
  stream_->codecpar->width = frameSize[0];
  stream_->codecpar->height = frameSize[1];

  av_dump_format(outputContext_.get(), 0, filename.c_str(), 1);

  // Open codec
  outputContext_->video_codec = avcodec_find_encoder(CODEC_ID);
  codecContext_ = av::make_codec_context(outputContext_->video_codec);
  codecContext_->time_base = stream_->time_base;
  codecContext_->gop_size = 10;
  codecContext_->pix_fmt = AV_PIX_FMT_YUV420P;
  codecContext_->width = frameSize[0];
  codecContext_->height = frameSize[1];
  {
    const int error = avcodec_open2(codecContext_.get(), outputContext_->video_codec, nullptr);
    CHECK_EQ(error, 0);
  }

  // Buffer
  packet_ = av::packet_alloc();

  // Frame
  frame_ = av::make_frame(AV_PIX_FMT_YUV420P, frameSize[0], frameSize[1]);
  CHECK_NOT_NULL(frame_);
  {
    const int error = avcodec_parameters_from_context(stream_->codecpar, codecContext_.get());
    CHECK_EQ(error, 0);
  }

  {
    auto [ioCtx, error] = av::io_open(filename, AVIO_FLAG_WRITE);
    if (ioCtx == nullptr) {
      throw std::runtime_error("Could not open video file for writting ("s + filename + "): "s +
                               av::make_error_string(error));
    }
    ioCtx_ = std::move(ioCtx);
  }
  outputContext_->pb = ioCtx_.get();

  {
    const int error = avformat_init_output(outputContext_.get(), &opt);
    CHECK_GE(error, 0);
  }
  {
    const int error = avformat_write_header(outputContext_.get(), &opt);
    CHECK_GE(error, 0);
  }
}

void VideoWriterImpl::write(const cv::Mat3b &img) {
  CHECK_EQ(av_buffer_is_writable(frame_->buf[0]), 1);

  const int srcStrides[1] = {img.step[0]};
  uint8_t *outSlices[2] = {frame_->buf[0]->data, frame_->buf[1]->data};
  CHECK_NOT_NULL(outSlices[0]);
  CHECK_NOT_NULL(outSlices[1]);
  CHECK_GE(frame_->linesize[0], 1);
  CHECK_GE(frame_->linesize[1], 1);
  sws_scale(swsContext_.get(), &img.data, srcStrides, 0, img.rows, outSlices, frame_->linesize);

  // // Wrap buffer with cv::Mat3b
  // cv::Mat3b buffer{};
  // for (int y = 0; y < img.rows; ++y) {
  //   const uchar *inputRow = &img.data[y * img.step[0]];
  //   uint8_t *outputRow = &frame_->buf[0]->data[y * frame_->linesize[0]];

  //   for (int x = 0; x < img.cols; ++x) {
  //     for (int c = 0; c < 3; ++c) {
  //       outputRow[3 * x + c] = inputRow[3 * x + c];
  //     }
  //   }
  // }
  frame_->pts = frameIndex_++;

  // Encode
  {
    const int error = avcodec_send_frame(codecContext_.get(), frame_.get());
    CHECK_EQ(error, 0);
  }
  while (true) {
    {
      const int error = avcodec_receive_packet(codecContext_.get(), packet_.get());
      if (error == AVERROR(EAGAIN) || error == AVERROR_EOF) {
        break;
      } else if (error < 0) {
        throw std::runtime_error("Could not retrieve encoder packet: "s + av::make_error_string(error));
      }
    }

    av_packet_rescale_ts(packet_.get(), codecContext_->time_base, stream_->time_base);
    packet_->stream_index = stream_->index;

    {
      const int error = av_interleaved_write_frame(outputContext_.get(), packet_.get());
      CHECK_EQ(error, 0);
    }
  }
}

VideoWriterImpl::~VideoWriterImpl() { av_write_trailer(outputContext_.get()); }

} // namespace detail

VideoWriter::VideoWriter() = default;

VideoWriter::~VideoWriter() = default;

bool VideoWriter::open(const std::string &filename, Vector2i frameSize, float32_t fps) {
  impl_ = std::make_unique<detail::VideoWriterImpl>(filename, frameSize, fps);
  return true;
}
void VideoWriter::write(const cv::Mat3b &img) { impl_->write(img); }
} // namespace zoo
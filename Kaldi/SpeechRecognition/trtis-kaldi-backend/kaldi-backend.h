// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#define HAVE_CUDA 0  // Loading Kaldi headers with GPU

#include <cfloat>
#include <sstream>

#include "online2/online-nnet3-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"

#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-utils.h"
#include "util/kaldi-thread.h"

#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/custom/sdk/custom_instance.h"
#include <unordered_map>

using kaldi::BaseFloat;

namespace nvidia {
namespace inferenceserver {
namespace custom {
namespace kaldi_cbe {

class Context;

class ASRPipeline {
 public:
  kaldi::OnlineNnet2FeaturePipelineInfo pipeline_info_;
  kaldi::OnlineNnet2FeaturePipeline feature_pipeline_; 
  kaldi::nnet3::DecodableNnetSimpleLoopedInfo decodable_info_;
  kaldi::SingleUtteranceNnet3Decoder  decoder_;
  int32_t frame_offset;

  ASRPipeline(Context& ctx);
};
 
typedef std::unordered_map<CorrelationID, std::shared_ptr<ASRPipeline> > ASRDecoderMap; 

// Context object. All state must be kept in this object.
class Context {
 public:
  Context(const std::string& instance_name, const ModelConfig& config,
          const int gpu_device);
  virtual ~Context();

  // Initialize the context. Validate that the model configuration,
  // etc. is something that we can handle.
  int Init();

  // Perform custom execution on the payloads.
  int Execute(const uint32_t payload_cnt, CustomPayload* payloads,
              CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn);
  friend class ASRPipeline;

 private:
  // init kaldi pipeline
  int InitializeKaldiPipeline();
  int InputOutputSanityCheck();
  int ReadModelParameters();
  int GetSequenceInput(CustomGetNextInputFn_t& input_fn, void* input_context,
                       CorrelationID* corr_id, int32_t* start, int32_t* ready,
                       int32_t* dim, int32_t* end,
                       const kaldi::BaseFloat** wave_buffer,
                       std::vector<uint8_t>* input_buffer);

  int SetOutputTensor(const std::string& output, CustomGetOutputFn_t output_fn,
                      CustomPayload payload);

  bool CheckPayloadError(const CustomPayload& payload);
  void ResetPipeline();

  // The name of this instance of the backend.
  const std::string instance_name_;

  // The model configuration.
  const ModelConfig model_config_;

  // Models paths
  std::string nnet3_rxfilename_, fst_rxfilename_;
  std::string word_syms_rxfilename_;

  // batch_size
  int max_batch_size_;
  int num_channels_;
  int num_worker_threads_;

  BaseFloat sample_freq_, seconds_per_chunk_;
  int chunk_num_bytes_, chunk_num_samps_;

  kaldi::OnlineNnet2FeaturePipelineConfig feature_opts;
  kaldi::nnet3::NnetSimpleLoopedComputationOptions compute_opts;
  kaldi::LatticeFasterDecoderConfig decoder_opts;
  std::unique_ptr<fst::Fst<fst::StdArc>> decode_fst_;

  // Maintain the state of some shared objects
  kaldi::TransitionModel trans_model_;

  kaldi::nnet3::AmNnetSimple am_nnet_;
  fst::SymbolTable* word_syms_;

  ASRDecoderMap pipeline_;

  const uint64_t int32_byte_size_;
  const uint64_t int64_byte_size_;
  std::vector<int64_t> output_shape_;

  std::vector<uint8_t> byte_buffer_;
  std::vector<std::vector<uint8_t>> wave_byte_buffers_;

};

}  // kaldi
}  // custom
}  // inferenceserver
}  // nvidia

#include <torch/torch.h>
#include <torch/script.h>
#include <c10/util/Exception.h>
#include <iostream>
#include <omp.h>

template <typename scalar_t, bool sorted>
at::Tensor nms_cpu_kernel(const at::Tensor& dets,
                          const at::Tensor& scores,
                          const float threshold, float bias) {
  AT_ASSERTM(!dets.type().is_cuda(), "dets must be a CPU tensor");
  AT_ASSERTM(!scores.type().is_cuda(), "scores must be a CPU tensor");
  AT_ASSERTM(dets.type() == scores.type(), "dets should have the same type as scores");

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  }

  auto x1_t = dets.select(1, 0).contiguous();
  auto y1_t = dets.select(1, 1).contiguous();
  auto x2_t = dets.select(1, 2).contiguous();
  auto y2_t = dets.select(1, 3).contiguous();

  at::Tensor areas_t = (x2_t - x1_t + bias) * (y2_t - y1_t + bias);

  auto ndets = dets.size(0);
  // If scores and dets are already sorted in descending order, we don't need to sort it again.
  auto order_t = sorted ? at::arange(0, ndets, scores.options().dtype(at::kLong)) :
                          std::get<1>(scores.sort(0, /* descending=*/true));

  at::Tensor suppressed_t = at::zeros({ndets}, dets.options().dtype(at::kByte).device(at::kCPU));

  auto suppressed = suppressed_t.data<uint8_t>();
  auto order = order_t.data<int64_t>();
  auto x1 = x1_t.data<scalar_t>();
  auto y1 = y1_t.data<scalar_t>();
  auto x2 = x2_t.data<scalar_t>();
  auto y2 = y2_t.data<scalar_t>();
  auto areas = areas_t.data<scalar_t>();
  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1)
      continue;
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto iarea = areas[i];
    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1)
        continue;
      auto xx1 = std::max(ix1, x1[j]);
      auto yy1 = std::max(iy1, y1[j]);
      auto xx2 = std::min(ix2, x2[j]);
      auto yy2 = std::min(iy2, y2[j]);

      auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1 + bias);
      auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1 + bias);
      auto inter = w * h;
      auto ovr = inter / (iarea + areas[j] - inter);
      if (ovr >= threshold)
        suppressed[j] = 1;
   }
  }
  return at::nonzero(suppressed_t == 0).squeeze(1);
}

std::vector<at::Tensor> remove_empty(std::vector<at::Tensor>& candidate, int64_t start, int64_t end) {
  std::vector<at::Tensor> valid_candidate;
  for (int64_t i = start; i < end; i++) {
    if (candidate[i].defined()) {
      valid_candidate.push_back(candidate[i]);
    }
  }
  return valid_candidate;
}

std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor>> batch_score_nms_kernel(const at::Tensor& batch_dets,
                          const at::Tensor& batch_scores,
                          const float threshold, const int max_output=200) {
  auto nbatch = batch_scores.size(0); // number of batches
  auto ndets = batch_scores.size(1); // number of boxes
  auto nscore = batch_scores.size(2); // number of labels

  auto nbatch_x_nscore = nbatch * nscore; // (number of batches) * (number of labels)
  std::vector<at::Tensor> bboxes_out(nbatch_x_nscore);
  std::vector<at::Tensor> scores_out(nbatch_x_nscore);
  std::vector<at::Tensor> labels_out(nbatch_x_nscore);

  //at::set_num_threads(at::intraop_default_num_threads());
  int64_t grain_size = std::min(at::internal::GRAIN_SIZE / nbatch_x_nscore, (int64_t)1);
  at::parallel_for(0, nbatch_x_nscore, grain_size, [&](int64_t begin, int64_t end){
    for (int index = begin; index < end; index++) {
      // Parallel in the dimentaion of: batch * nscore
      auto bs = index / nscore;
      auto i = index % nscore;

      // skip background (i = 0)
      if (i == 0) {
        continue;
      }

      at::Tensor dets = batch_dets[bs].squeeze(0); // dets for boxes per image: (num_bbox, 4); For example: (15130, 4)
      at::Tensor scores = batch_scores[bs].squeeze(0); // scores for boxes per image: (num_bbox, 81); For example: (15130, 81)

      at::Tensor score = scores.slice(1, i, i+1).squeeze(1); // score for boxes per image per class: (num_bbox); For example: (15130)

      at::Tensor mask_index = at::nonzero(score > 0.05).squeeze(1);
      at::Tensor bboxes = at::index_select(dets, /*dim*/0, mask_index);
      score = at::index_select(score, /*dim*/0, mask_index);

      if (score.size(0) == 0) {
        continue;
      }

      at::Tensor score_sliced, score_idx_sorted;
      // select max_output highest' score and bboxes
      std::tie(score_sliced, score_idx_sorted) = at::topk(score, (max_output>score.size(0))?score.size(0):max_output, 0);
      at::Tensor bboxes_sliced = at::index_select(bboxes, /*dim*/0, score_idx_sorted);

      at::Tensor keep = nms_cpu_kernel<float, /*sorted*/true>(bboxes_sliced, score_sliced, threshold, /*bias*/0);

      bboxes_out[index] = at::index_select(bboxes_sliced, /*dim*/0, keep);
      scores_out[index] = at::index_select(score_sliced, /*dim*/0, keep);
      // TODO optimize the fill_
      labels_out[index] = at::empty({keep.sizes()}).fill_(i);
    }
  });
  std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor>> output(nbatch);
  at::parallel_for(0, nbatch, 1, [&](int64_t begin, int64_t end){
    for (int bs = begin; bs < end; bs++) {
      // Post process the tensors to get the top max_output(number) for each Batchsize
      std::vector<at::Tensor> valid_bboxes_out = remove_empty(bboxes_out, bs*nscore, (bs+1)*nscore);
      std::vector<at::Tensor> valid_scores_out = remove_empty(scores_out, bs*nscore, (bs+1)*nscore);
      std::vector<at::Tensor> valid_labels_out = remove_empty(labels_out, bs*nscore, (bs+1)*nscore);

      at::Tensor bboxes_out_ = at::cat(valid_bboxes_out, 0);
      at::Tensor labels_out_ = at::cat(valid_labels_out, 0);
      at::Tensor scores_out_ = at::cat(valid_scores_out, 0);

      std::tuple<at::Tensor, at::Tensor> sort_result = scores_out_.sort(0);
      at::Tensor max_ids = std::get<1>(sort_result);
      max_ids = max_ids.slice(/*dim*/0, /*start*/std::max(max_ids.size(0) - max_output, static_cast<int64_t>(0)), /*end*/max_ids.size(0));
      output[bs] = std::tuple<at::Tensor, at::Tensor, at::Tensor>(bboxes_out_.index_select(/*dim*/0, /*index*/max_ids),
                                                                  labels_out_.index_select(/*dim*/0, /*index*/max_ids),
                                                                  scores_out_.index_select(/*dim*/0, /*index*/max_ids));
    }
  });
  return output;
}

int main()
{
  torch::jit::script::Module container = torch::jit::load("../scores3.pt");
  torch::jit::script::Module container2 = torch::jit::load("../dets.pt");
  // Load values by name
  torch::Tensor batch_scores = container.attr("scores").toTensor();
  torch::Tensor batch_dets = container2.attr("dets").toTensor();
  const float threshold = 0.5;
  const int max_output = 200;

  batch_score_nms_kernel(batch_dets, batch_scores, threshold, max_output);

  return 0;
}
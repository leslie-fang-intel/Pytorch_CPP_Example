#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

int main()
{
  torch::jit::script::Module container = torch::jit::load("../scores3.pt");

  // Load values by name
  torch::Tensor batch_scores = container.attr("scores").toTensor();

  std::cout<<"at::nonzero(score > 0.05): "<<at::nonzero(batch_scores[15].squeeze(0).slice(1, 1, 2).squeeze(1) > 0.05).sizes()<<std::endl;

  auto nbatch = batch_scores.size(0); // number of batches 16
  auto ndets = batch_scores.size(1); // number of boxes 15130
  auto nscore = batch_scores.size(2); // number of labels 81

  auto nbatch_x_nscore = nbatch * nscore; // (number of batches) * (number of labels)

#ifdef _OPENMP
#if (_OPENMP >= 201307)
# pragma omp parallel for simd schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#else
# pragma omp parallel for schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#endif
#endif
  for (int index = 0; index < nbatch_x_nscore; index++) {
    auto bs = index / nscore;
    auto i = index % nscore;
    at::Tensor scores = batch_scores[bs].squeeze(0); // scores for boxes per image: (num_bbox, 81); For example: (15130, 81)

    at::Tensor score = scores.slice(1, i, i+1).squeeze(1); // score for boxes per image per class: (num_bbox); For example: (15130)

    if (bs == 15 && i == 1) {
      std::cout<<"score: "<<score.sizes()<<std::endl;
      std::cout<<"score > 0.05: "<<(score > 0.05).sizes()<<std::endl;
      std::cout<<"at::nonzero(score > 0.05): "<<at::nonzero(score > 0.05).sizes()<<std::endl;
    }

    at::Tensor mask_index = at::nonzero(score > 0.05).squeeze(1);

    if (bs == 15 && i == 1) {
      std::cout<<"mask_index: "<<mask_index.sizes()<<std::endl;
    }
  }
  return 0;
}
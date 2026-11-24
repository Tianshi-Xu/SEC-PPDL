#pragma once
#include <QuantExperiment/Context.h>
#include <chrono>

namespace QuantExperiment {
struct Component {
    double seconds = 0;
    uint64_t bytes = 0;
};

class Timer {
  public:
    Timer(Context &context, Component &target)
        : context_(context), target_(target), before_(context.Bytes()),
          start_(std::chrono::steady_clock::now()) {}
    ~Timer() {
        target_.seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
        target_.bytes += context_.Bytes() - before_;
    }

  private:
    Context &context_;
    Component &target_;
    uint64_t before_;
    std::chrono::steady_clock::time_point start_;
};
} // namespace QuantExperiment

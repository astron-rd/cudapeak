#include <memory>
#include <string>
#include <vector>

#include <cudawrappers/cu.hpp>
#include <cudawrappers/nvrtc.hpp>

class KernelFactory {
public:
  KernelFactory(const std::string &kernel_source);

  std::vector<std::shared_ptr<cu::Function>>
  compileKernels(cu::Device &device,
                 const std::vector<std::string> &kernel_name);

  std::shared_ptr<cu::Function> compileKernel(cu::Device &device,
                                              const std::string &kernel_name);

private:
  std::string kernel_source_;
  std::unique_ptr<nvrtc::Program> program_;
  std::unique_ptr<cu::Module> module_;
};
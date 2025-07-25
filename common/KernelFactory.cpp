#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <cudawrappers/cu.hpp>
#include <cudawrappers/nvrtc.hpp>

#include "KernelFactory.h"

KernelFactory::KernelFactory(const std::string &kernel_source)
    : kernel_source_(kernel_source) {
  program_ = std::make_unique<nvrtc::Program>(kernel_source_, "");
}

std::vector<std::shared_ptr<cu::Function>>
KernelFactory::compileKernels(cu::Device &device,
                              const std::vector<std::string> &kernel_names) {
  const std::string cuda_include_path = nvrtc::findIncludePath();
  const std::string arch = device.getArch();

  std::vector<std::string> options = {
      "-I" + cuda_include_path,
      "-ffast-math",
#if defined(__HIP__)
      "-std=c++17",
      "--offload-arch=" + arch,
#else
      "-arch=" + arch,
#endif
  };

  for (const auto &kernel_name : kernel_names) {
    program_->addNameExpression(kernel_name);
  }
  try {
    program_->compile(options);
  } catch (nvrtc::Error &error) {
    std::cerr << program_->getLog();
    throw;
  }

  module_ = std::make_unique<cu::Module>(
      static_cast<const void *>(program_->getPTX().data()));

  std::vector<std::shared_ptr<cu::Function>> functions;
  for (auto kernel_name : kernel_names) {
    functions.push_back(std::make_shared<cu::Function>(
        *module_, program_->getLoweredName(kernel_name)));
  }

  return functions;
}

std::shared_ptr<cu::Function>
KernelFactory::compileKernel(cu::Device &device,
                             const std::string &kernel_name) {
  return compileKernels(device, {kernel_name}).front();
}
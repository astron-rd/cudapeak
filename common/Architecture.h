#ifndef ARCHITECTURE_H
#define ARCHITECTURE_H

#include <string>

// Blackwell comes in three architecture classes. The datacenter parts
// (sm_100, sm_101, sm_103) and Thor (sm_110) implement the tcgen05 tensor
// core instruction family. Consumer Blackwell (sm_120, sm_121) has no
// tcgen05, but does have FP4/FP6/FP8 mma instructions.

inline bool isBlackwellDatacenter(const std::string &arch) {
  return arch.find("sm_100") != std::string::npos ||
         arch.find("sm_101") != std::string::npos ||
         arch.find("sm_103") != std::string::npos;
}

inline bool isBlackwellThor(const std::string &arch) {
  return arch.find("sm_110") != std::string::npos;
}

inline bool isBlackwellConsumer(const std::string &arch) {
  return arch.find("sm_120") != std::string::npos ||
         arch.find("sm_121") != std::string::npos;
}

#endif // ARCHITECTURE_H

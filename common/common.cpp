#include <cmath>

#include "common.h"

unsigned roundToPowOf2(unsigned number) {
  double logd = log(number) / log(2);
  logd = floor(logd);
  return (unsigned)pow(2, (int)logd);
}

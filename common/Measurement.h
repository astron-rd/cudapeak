#ifndef MEASUREMENT_H
#define MEASUREMENT_H

#include <nlohmann/json.hpp>

class Measurement {
public:
  double runtime; // milliseconds
  double power;   // watts
  int frequency;  // MHz
  double gops;    // number of giga operations
  double gbytes;  // number of gigabytes

  nlohmann::json toJson() const;
};

std::ostream &operator<<(std::ostream &stream, const Measurement &m);

#endif // MEASUREMENT_H
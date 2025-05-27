#ifndef MEASUREMENT_H
#define MEASUREMENT_H

class Measurement {
public:
  double runtime; // milliseconds
  double power;   // watts
  int frequency;  // MHz
  double gops;    // number of giga operations
  double gbytes;  // number of gigabytes
};

std::ostream &operator<<(std::ostream &stream, const Measurement &m);

#endif // MEASUREMENT_H
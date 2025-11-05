#ifndef MEASUREMENT_H
#define MEASUREMENT_H

class Measurement {
public:
  double runtime; // milliseconds
  double power;   // watts
  int frequency;  // MHz
  double gops;    // number of giga operations
  double gbytes;  // number of gigabytes

  void print_runtime(std::ostream &stream, bool json = false) const;
  void print_ops(std::ostream &stream, bool json = false) const;
  void print_power(std::ostream &stream, bool json = false) const;
  void print_efficiency(std::ostream &stream, bool json = false) const;
  void print_bandwidth(std::ostream &stream, bool json = false) const;
  void print_frequency(std::ostream &stream, bool json = false) const;

  void toJson(std::ostream &stream) const;
};

std::ostream &operator<<(std::ostream &stream, const Measurement &m);

#endif // MEASUREMENT_H
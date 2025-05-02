#include <iomanip>
#include <iostream>

#include "common.h"
#include "Measurement.h"

namespace {
void print_ops(std::ostream& stream, const Measurement& m) {
  const double seconds = m.runtime * 1e-3;
  if (m.gops != 0) {
    stream << ", " << std::setw(w2) << m.gops / seconds * 1e-3 << " TOps/s";
  }
}

void print_power(std::ostream& stream, const Measurement& m) {
  if (m.power > 1) {
    stream << ", " << std::setw(w2) << m.power << " W";
  }
}

void print_efficiency(std::ostream& stream, const Measurement& m) {
  const double seconds = m.runtime * 1e-3;
  if (m.gops != 0 && m.power > 1) {
    stream << ", " << std::setw(w2) << m.gops / seconds / m.power << " GOps/W";
  }
}

void print_bandwidth(std::ostream& stream, const Measurement& m) {
  const double seconds = m.runtime * 1e-3;
  if (m.gbytes != 0) {
    stream << ", " << std::setw(w2) << m.gbytes / seconds << " GB/s";
  }
}

void print_oi(std::ostream& stream, const Measurement& m) {
  if (m.gops != 0 && m.gbytes != 0) {
    const float operational_intensity = m.gops / m.gbytes;
    stream << ", " << std::setw(w2) << operational_intensity << " Op/byte";
  }
}

void print_frequency(std::ostream& stream, const Measurement& m) {
  if (m.frequency != 0) {
    stream << ", " << std::setw(w2) << m.frequency << " MHz";
  }
}
}  // namespace

std::ostream& operator<<(std::ostream& stream, const Measurement& m) {
  print_ops(stream, m);
  print_power(stream, m);
  print_efficiency(stream, m);
  print_bandwidth(stream, m);
  print_oi(stream, m);
  print_frequency(stream, m);
  return stream;
}
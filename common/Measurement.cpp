#include <iomanip>
#include <iostream>

#include "Measurement.h"
#include "common.h"

void Measurement::print_runtime(std::ostream &stream, bool json) const {
  if (json) {
    stream << "\"runtime\": " << runtime << ", ";
  } else {
    stream << std::setw(w2) << runtime << " ms";
  }
}

void Measurement::print_ops(std::ostream &stream, bool json) const {
  const double seconds = runtime * 1e-3;
  if (gops != 0) {
    const double tops = gops / seconds * 1e-3;
    if (json) {
      stream << "\"tops\": " << tops << ", ";
    } else {
      stream << ", " << std::setw(w2) << tops << " TOps/s";
    }
  }
}

void Measurement::print_power(std::ostream &stream, bool json) const {
  if (power > 1) {
    if (json) {
      stream << "\"power\": " << power << ", ";
    } else {
      stream << ", " << std::setw(w2) << power << " W";
    }
  }
}

void Measurement::print_efficiency(std::ostream &stream, bool json) const {
  const double seconds = runtime * 1e-3;
  if (gops != 0 && power > 1) {
    const double efficiency = gops / seconds / power;
    if (json) {
      stream << "\"efficiency\": " << efficiency << ", ";
    } else {
      stream << ", " << std::setw(w2) << efficiency << " GOps/W";
    }
  }
}

void Measurement::print_bandwidth(std::ostream &stream, bool json) const {
  const double seconds = runtime * 1e-3;
  if (gbytes != 0) {
    const double bandwidth = gbytes / seconds;
    if (json) {
      stream << "\"bandwidth\": " << bandwidth << ", ";
    } else {
      stream << ", " << std::setw(w2) << gbytes / seconds << " GB/s";
    }
  }
}

void Measurement::print_frequency(std::ostream &stream, bool json) const {
  if (frequency != 0) {
    if (json) {
      stream << "\"frequency\": " << frequency;
    } else {
      stream << ", " << std::setw(w2) << frequency << " MHz";
    }
  }
}

std::ostream &operator<<(std::ostream &stream, const Measurement &m) {
  m.print_runtime(stream);
  m.print_ops(stream);
  m.print_power(stream);
  m.print_efficiency(stream);
  m.print_bandwidth(stream);
  m.print_frequency(stream);
  return stream;
}

void Measurement::toJson(std::ostream &stream) const {
  print_runtime(stream, true);
  print_ops(stream, true);
  print_power(stream, true);
  print_efficiency(stream, true);
  print_bandwidth(stream, true);
  print_frequency(stream, true);
}

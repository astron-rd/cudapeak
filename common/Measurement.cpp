#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>

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
      stream << "\"tops\": " << tops;
      if (power > 0 || gbytes > 0 || frequency > 0) {
        stream << ", ";
      }
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

void Measurement::toJson(nlohmann::json &j) const {
  j["runtime"] = runtime;
  const double seconds = runtime * 1e-3;
  if (gops != 0) {
    const double tops = gops / seconds * 1e-3;
    j["tops"] = tops;
  }
  if (power > 1) {
    j["power"] = power;
  }
  if (gops != 0 && power > 1) {
    const double efficiency = gops / seconds / power;
    j["efficiency"] = efficiency;
  }
  if (gbytes != 0) {
    const double bandwidth = gbytes / seconds;
    j["bandwidth"] = bandwidth;
  }
  if (frequency != 0) {
    j["frequency"] = frequency;
  }
}

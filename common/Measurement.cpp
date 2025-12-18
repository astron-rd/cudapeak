#include <cmath>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>

#include "Measurement.h"
#include "common.h"

std::ostream &operator<<(std::ostream &stream, const Measurement &m) {
  auto j = m.toJson();

  // runtime
  if (j.contains("runtime")) {
    stream << std::setw(w2) << j["runtime"].get<double>() << " ms";
  } else {
    stream << std::setw(w2) << m.runtime << " ms";
  }

  if (j.contains("tops")) {
    stream << ", " << std::setw(w2) << j["tops"].get<double>() << " TOps/s";
  }
  if (j.contains("power")) {
    stream << ", " << std::setw(w2) << j["power"].get<double>() << " W";
  }
  if (j.contains("efficiency")) {
    stream << ", " << std::setw(w2) << j["efficiency"].get<double>()
           << " GOps/W";
  }
  if (j.contains("bandwidth")) {
    stream << ", " << std::setw(w2) << j["bandwidth"].get<double>() << " GB/s";
  }
  if (j.contains("frequency")) {
    stream << ", " << std::setw(w2) << j["frequency"].get<int>() << " MHz";
  }

  return stream;
}

nlohmann::json Measurement::toJson() const {
  auto round_digits = [](double v, int digits) {
    double f = std::pow(10.0, digits);
    return std::round(v * f) / f;
  };
  const int PREC = 3;

  nlohmann::json j;

  // runtime (ms)
  if (runtime != 0) {
    j["runtime"] = round_digits(runtime, PREC);
  } else {
    j["runtime"] = runtime;
  }

  const double seconds = runtime * 1e-3;

  if (gops != 0) {
    j["gops"] = round_digits(gops, PREC);
  }
  if (gops != 0 && runtime != 0) {
    const double tops = gops / seconds * 1e-3;
    j["tops"] = round_digits(tops, PREC);
  }

  if (power > 1) {
    j["power"] = round_digits(power, PREC);
  }

  if (gops != 0 && power > 1 && runtime != 0) {
    const double efficiency = gops / seconds / power;
    j["efficiency"] = round_digits(efficiency, PREC);
  }

  if (gbytes != 0) {
    j["gbytes"] = round_digits(gbytes, PREC);
  }
  if (gbytes != 0 && runtime != 0) {
    const double bandwidth = gbytes / seconds;
    j["bandwidth"] = round_digits(bandwidth, PREC);
  }

  if (frequency != 0) {
    j["frequency"] = frequency;
  }

  return j;
}

void to_json(nlohmann::json &j, const Measurement &m) { j = m.toJson(); }

void from_json(const nlohmann::json &j, Measurement &m) {
  m.runtime = j.value("runtime", 0.0);
  m.power = j.value("power", 0.0);
  m.frequency = j.value("frequency", 0);
  m.gops = j.value("gops", 0.0);
  m.gbytes = j.value("gbytes", 0.0);
}

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

std::vector<std::string> split(const std::string &s, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(s);
  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

std::vector<std::string>
transform(const std::vector<std::string> &kernel_names) {
  std::vector<std::string> result;

  for (const auto &name : kernel_names) {
    auto parts = split(name, '_');
    if (parts.size() < 5) {
      continue;
    }

    std::ostringstream oss;
    oss << "fma:" << parts[1] << " (" << parts[2] << ") -> " << std::setw(4)
        << parts[3] << ":" << parts[4];

    result.push_back(oss.str());
  }

  return result;
}

#pragma once
#include "nlohmann/json.hpp"

nlohmann::json detect_capabilities(int max_channels = 8);

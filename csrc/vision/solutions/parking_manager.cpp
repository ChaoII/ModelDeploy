#include "vision/solutions/parking_manager.h"
#include <algorithm>
namespace modeldeploy::vision::solution {
void ParkingManager::set_slots(const std::vector<std::vector<Point2f>>& slots) {
    slots_.clear();
    occupied_.assign(slots.size(), false);
    for (const auto& s : slots) slots_.emplace_back(tool::PolygonZone(s));
}
void ParkingManager::update(const std::vector<tracking::TrackResult>& tracks) {
    std::fill(occupied_.begin(), occupied_.end(), false);
    for (const auto& t : tracks) {
        const Point2f c(t.box.x + t.box.width * 0.5f, t.box.y + t.box.height * 0.5f);
        for (size_t i = 0; i < slots_.size(); ++i)
            if (slots_[i].contains(c)) occupied_[i] = true;
    }
}
} // namespace modeldeploy::vision::solution

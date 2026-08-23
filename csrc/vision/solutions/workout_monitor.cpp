#include "vision/solutions/workout_monitor.h"
#include <cmath>
namespace modeldeploy::vision::solution {
static const float kPi = 3.14159265358979323846f;
float WorkoutMonitor::angle(Point3f a, Point3f b, Point3f c) {
    const float abx = a.x - b.x, aby = a.y - b.y;
    const float cbx = c.x - b.x, cby = c.y - b.y;
    const float dot = abx * cbx + aby * cby;
    const float n1 = std::sqrt(abx * abx + aby * aby);
    const float n2 = std::sqrt(cbx * cbx + cby * cby);
    if (n1 <= 0 || n2 <= 0) return 180.0f;
    const float cosv = std::max(-1.0f, std::min(1.0f, dot / (n1 * n2)));
    return std::acos(cosv) * 180.0f / kPi;
}
void WorkoutMonitor::update(float deg) {
    if (deg < min_) down_ = true;
    else if (deg > max_ && down_) { ++reps_; down_ = false; }
}
} // namespace modeldeploy::vision::solution

#include "vision/solutions/vision_eye.h"
namespace modeldeploy::vision::solution {
Point2f VisionEye::map_to_eye(Point2f centroid, float eye_level_y) { return Point2f(centroid.x, eye_level_y); }
void VisionEye::add(Point2f centroid) { eyes_.push_back(map_to_eye(centroid, eye_level_)); }
} // namespace modeldeploy::vision::solution

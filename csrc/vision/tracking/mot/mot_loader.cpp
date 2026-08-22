#include "vision/tracking/mot/mot_loader.h"

#include <cctype>
#include <fstream>
#include <map>
#include <sstream>
#include <unordered_set>

namespace modeldeploy::vision::tracking {
    namespace {
        std::string trim(const std::string& s) {
            size_t b = 0, e = s.size();
            while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
            while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
            return s.substr(b, e - b);
        }

        std::vector<std::string> split(const std::string& line, char sep) {
            std::vector<std::string> out;
            std::string cur;
            for (char c : line) {
                if (c == sep) {
                    out.push_back(trim(cur));
                    cur.clear();
                } else {
                    cur.push_back(c);
                }
            }
            out.push_back(trim(cur));
            return out;
        }

        bool parse_int(const std::string& s, int& out) {
            if (s.empty()) return false;
            try {
                out = std::stoi(s);
                return true;
            } catch (...) {
                return false;
            }
        }

        bool parse_float(const std::string& s, float& out) {
            if (s.empty()) return false;
            try {
                out = std::stof(s);
                return true;
            } catch (...) {
                return false;
            }
        }
    }  // namespace

    MotSequence load_mot_sequence(const std::string& filepath) {
        MotSequence seq;
        std::ifstream in(filepath);
        if (!in.is_open()) return seq;

        std::map<int, MotFrame> by_frame;  // ordered by frame_id
        std::unordered_set<int> gt_ids;
        std::string line;
        while (std::getline(in, line)) {
            std::string t = trim(line);
            if (t.empty() || t[0] == '#') continue;
            auto cols = split(t, ',');
            if (cols.size() < 7) continue;

            int frame_id = 0, obj_id = 0;
            float left = 0, top = 0, w = 0, h = 0, conf = 0;
            if (!parse_int(cols[0], frame_id)) continue;
            if (!parse_int(cols[1], obj_id)) continue;
            if (!parse_float(cols[2], left)) continue;
            if (!parse_float(cols[3], top)) continue;
            if (!parse_float(cols[4], w)) continue;
            if (!parse_float(cols[5], h)) continue;
            if (!parse_float(cols[6], conf)) continue;

            MotFrame& fr = by_frame[frame_id];
            fr.frame_id = frame_id;
            Rect2f box{left, top, w, h};
            if (conf < 0.0f) {  // GT row (visibility/conf column is -1)
                fr.gt_boxes.push_back(box);
                fr.gt_ids.push_back(obj_id);
                if (obj_id >= 0) gt_ids.insert(obj_id);
            } else {  // detection row
                Detection d;
                d.box = box;
                d.score = conf;
                d.label_id = 0;
                fr.dets.push_back(d);
            }
        }

        for (auto& kv : by_frame) seq.frames.push_back(std::move(kv.second));
        seq.num_gt_ids = static_cast<int>(gt_ids.size());
        return seq;
    }

    MotSequence load_mot_sequence_txt(const std::string& filepath) {
        return load_mot_sequence(filepath);
    }
}

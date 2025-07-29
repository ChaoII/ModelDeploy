//
// Created by aichao on 2025/6/4.
//


#pragma once

#include <cstdint>
#include <string>
#include <sstream>

namespace modeldeploy::vision {
    struct LetterBoxRecord {
        float ipt_w;
        float ipt_h;
        float out_w;
        float out_h;
        float pad_w;
        float pad_h;
        float scale;

        [[nodiscard]] std::string to_string() const {
            std::ostringstream oss;
            oss << "LetterBoxRecord(ipt_h=" << ipt_h
                << ", ipt_w=" << ipt_w
                << ", out_h=" << out_h
                << ", out_w=" << out_w
                << ", pad_h=" << pad_h
                << ", pad_w=" << pad_w
                << ", scale=" << scale
                << ")";
            return oss.str();
        }
    };


    struct Point2f {
        float x;
        float y;

        Point2f() {
            x = 0.0f;
            y = 0.0f;
        }

        Point2f(const float x, const float y) {
            this->x = x;
            this->y = y;
        }

        [[nodiscard]] std::string to_string() const {
            std::stringstream ss;
            ss << "x: " << this->x << " y: " << this->y;
            return ss.str();
        }
    };


    struct Point3f {
        float x;
        float y;
        float z;

        Point3f() {
            x = 0.0f;
            y = 0.0f;
            z = 0.0f;
        }

        Point3f(const float x, const float y, const float z) {
            this->x = x;
            this->y = y;
            this->z = z;
        }

        [[nodiscard]] std::string to_string() const {
            std::stringstream ss;
            ss << "x: " << this->x << " y: " << this->y << " z: " << this->z;
            return ss.str();
        }
    };


    struct Rect2f {
        float x;
        float y;
        float width;
        float height;

        Rect2f() {
            x = 0.0f;
            y = 0.0f;
            width = 0.0f;
            height = 0.0f;
        }

        Rect2f(const float x, const float y, const float width, const float height) {
            this->x = x;
            this->y = y;
            this->width = width;
            this->height = height;
        }

        [[nodiscard]] std::string to_string() const {
            std::stringstream ss;
            ss << "x: " << this->x << " y: " << this->y << " width: " << this->width << " height: " << this->height;
            return ss.str();
        }
    };

    struct RotatedRect {
        float xc;
        float yc;
        float width;
        float height;
        float angle;

        RotatedRect() {
            xc = 0.0f;
            yc = 0.0f;
            width = 0.0f;
            height = 0.0f;
            angle = 0.0f;
        }

        RotatedRect(const float x, const float y, const float width, const float height, const float angle) {
            this->xc = x;
            this->yc = y;
            this->width = width;
            this->height = height;
            this->angle = angle;
        }

        [[nodiscard]] std::string to_string() const {
            std::stringstream ss;
            ss << "xc: " << this->xc << " yc: " << this->yc << " width: " << this->width << " height: " << this->height
                << " angle: " << this->angle;
            return ss.str();
        }
    };
}

// Minimal glog compatibility stub.
//
// The WeTextProcessing runtime sources use <glog/logging.h> for LOG/CHECK
// logging. This header provides enough of the glog surface for that code to
// compile. In this vendor copy, LOG/VLOG are no-ops and CHECK_* abort on
// failure. It intentionally avoids a real glog dependency (which would pull in
// gflags and pthread support across platforms).

#pragma once

#include <cstdlib>
#include <iostream>

namespace mdlglog {
struct NullStream {
    template <class T>
    NullStream& operator<<(const T&) {
        return *this;
    }
    NullStream& operator<<(std::ostream& (*)(std::ostream&)) {
        return *this;
    }
};
struct CheckHelper {
    bool ok_;
    explicit CheckHelper(bool ok) : ok_(ok) {}
    template <class T>
    CheckHelper& operator<<(const T&) {
        if (!ok_) std::abort();
        return *this;
    }
};
}  // namespace mdlglog

#define LOG(severity) ::mdlglog::NullStream()
#define CHECK(expr) ::mdlglog::CheckHelper(static_cast<bool>(expr))
#define CHECK_EQ(a, b) ::mdlglog::CheckHelper((a) == (b))
#define CHECK_NE(a, b) ::mdlglog::CheckHelper((a) != (b))
#define CHECK_LE(a, b) ::mdlglog::CheckHelper((a) <= (b))
#define CHECK_LT(a, b) ::mdlglog::CheckHelper((a) < (b))
#define CHECK_GE(a, b) ::mdlglog::CheckHelper((a) >= (b))
#define CHECK_GT(a, b) ::mdlglog::CheckHelper((a) > (b))
#define VLOG(level) ::mdlglog::NullStream()
#define DLOG(severity) ::mdlglog::NullStream()

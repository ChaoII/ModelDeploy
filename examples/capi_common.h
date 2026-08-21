//
// 供 capi examples 复用的最小错误处理辅助
//
#ifndef MODELDEPLOY_EXAMPLES_CAPI_COMMON_H
#define MODELDEPLOY_EXAMPLES_CAPI_COMMON_H

#include "capi/md_capi.h"
#include <cstdio>
#include <cstdlib>

inline void die(MDStatus s, const char* what) {
    if (s != MD_OK) {
        std::fprintf(stderr, "[capi] %s failed: %s\n", what, md_get_last_error());
        std::exit(1);
    }
}

#endif

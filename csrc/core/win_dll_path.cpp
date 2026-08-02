//
// Created by aichao on 2025/8/2.
// Windows：确保优先加载 SDK 同目录下的 onnxruntime.dll。
// 问题：Windows System32 自带 onnxruntime 1.17，会抢占 SDK 捆绑的 1.22，
// 导致 ORT API 版本不匹配（22 vs 17）甚至崩溃。
// 解决：静态初始化时把 SDK 所在目录加入 DLL 搜索路径（排在 System32 之前）。
//

#if defined(_WIN32)
#include <windows.h>

namespace modeldeploy {
    static void init_ort_dll_search_path() {
        // 定位 ModelDeploySDK 模块所在目录
        wchar_t path[MAX_PATH];
        const HMODULE mod = GetModuleHandleW(L"ModelDeploySDK.dll");
        if (!mod) return;
        if (GetModuleFileNameW(mod, path, MAX_PATH) == 0) return;
        // 去掉文件名，只留目录
        wchar_t* slash = wcsrchr(path, L'\\');
        if (!slash) return;
        *slash = L'\0';
        // SetDllDirectory 使该目录在 System32 之前被搜索
        SetDllDirectoryW(path);
    }
    // 静态初始化（main 之前执行，确保 onnxruntime 按正确版本加载）
    static const bool g_ort_dll_init = [] { init_ort_dll_search_path(); return true; }();
} // namespace modeldeploy
#endif // _WIN32

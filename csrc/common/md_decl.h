//
// Created by AC on 2024-12-16.
//

#ifdef _WIN32
#ifdef MODEL_DEPLOY_LIB
#define EXPORT_DECL __declspec(dllexport)
#else
#define EXPORT_DECL __declspec(dllimport)
#endif  // MODEL_DEPLOY_LIB
#else
#define EXPORT_DECL __attribute__((visibility("default")))
#endif  // _WIN32

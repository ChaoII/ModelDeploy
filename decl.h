//
// Created by AC on 2024-12-16.
//

#ifdef _WIN32
#ifdef WRZSLABLIB
#define EXPORT_DECL __declspec(dllexport)
#else
#define EXPORT_DECL __declspec(dllimport)
#endif  // WRZSLABLIB
#else
#define EXPORT_DECL __attribute__((visibility("default")))
#endif  // _WIN32

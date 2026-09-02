// Vendored OpenFst configuration.
// Dynamic linking (dlopen/dlclose registration of external FST types) is
// disabled unconditionally across all platforms: it pulls in <dlfcn.h> which
// is not available on Windows/MSVC, and the static build always registers the
// default FST types required by the WeTextProcessing ITN backend anyway.
#define FST_NO_DYNAMIC_LINKING 1

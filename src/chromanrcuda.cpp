#include <cuda_runtime.h>
#include <memory>
#include <string>

#include "VSHelper4.h"
#include "VapourSynth4.h"

namespace {
using namespace std::string_literals;
}

#define checkError(expr)                                                 \
  do {                                                                   \
    if (cudaError_t result = expr; result != cudaSuccess) [[unlikely]] { \
      const char* error_str = cudaGetErrorString(result);                \
      return set_error("'"s + #expr + "' failed: " + error_str);         \
    }                                                                    \
  } while (0)

extern void process_cuda(float* d_dst, int w, int h, float thres, float thres_y, float thres_u, float thres_v, int sizew, int sizeh, int stepw, int steph, int chroma_ssw, int chroma_ssh,
                         bool use_euclidean, cudaTextureObject_t tex) noexcept;

struct ChromanrcudaData final {
    VSNode* node;
    const VSVideoInfo* vi;

    int d_pitch;
    float* h_src;
    float* h_dst;
    float* d_dst;
    cudaArray* d_src;
    cudaTextureObject_t tex;

    float thres;
    float thres_y;
    float thres_u;
    float thres_v;
    int sizew;
    int sizeh;
    int stepw;
    int steph;
    int chroma_ssw;
    int chroma_ssh;
    bool use_euclidean;
};

static const VSFrame* VS_CC chromanrcudaGetFrame(int n, int activationReason, void* instanceData, void** frameData, VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto d{static_cast<ChromanrcudaData*>(instanceData)};

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame* src = vsapi->getFrameFilter(n, d->node, frameCtx);
        VSFrame* dst{};

        auto set_error = [&](const std::string& error_message) -> const VSFrame* {
            if (dst != nullptr) {
                vsapi->freeFrame(dst);
            }
            vsapi->freeFrame(src);
            vsapi->setFilterError(("chromanrcuda: " + error_message).c_str(), frameCtx);
            return nullptr;
        };

        dst = vsapi->newVideoFrame(&d->vi->format, d->vi->width, d->vi->height, src, core);

        const int cssw = d->chroma_ssw;
        const int cssh = d->chroma_ssh;
        const int w = d->vi->width >> cssw;
        const int h = d->vi->height >> cssh;
        auto srcpy = vsapi->getReadPtr(src, 0);
        auto srcpu = vsapi->getReadPtr(src, 1);
        auto srcpv = vsapi->getReadPtr(src, 2);
        auto dstpy = vsapi->getWritePtr(dst, 0);
        auto dstpu = vsapi->getWritePtr(dst, 1);
        auto dstpv = vsapi->getWritePtr(dst, 2);
        auto stridey = vsapi->getStride(src, 0);
        auto strideu = vsapi->getStride(src, 1);
        auto stridev = vsapi->getStride(src, 2);

        const float* srcpyf = (const float*)srcpy;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                d->h_src[y * w + x] = srcpyf[(x << cssw)];
            }
            srcpyf += ((stridey >> 2) << cssh);
        }

        ptrdiff_t plane = w * h;
        vsh::bitblt(dstpy, stridey, srcpy, stridey, d->vi->width * sizeof(float), d->vi->height);
        vsh::bitblt(d->h_src + plane, d->d_pitch, srcpu, strideu, w * sizeof(float), h);
        vsh::bitblt(d->h_src + (plane * 2), d->d_pitch, srcpv, stridev, w * sizeof(float), h);
        checkError(cudaMemcpy2DToArray(d->d_src, 0, 0, d->h_src, d->d_pitch, d->d_pitch, h * 3, cudaMemcpyHostToDevice));

        process_cuda(d->d_dst, w, h, d->thres, d->thres_y, d->thres_u, d->thres_v, d->sizew, d->sizeh, d->stepw, d->steph, d->chroma_ssw, d->chroma_ssh, d->use_euclidean, d->tex);

        checkError(cudaMemcpy(d->h_dst, d->d_dst, d->d_pitch * h * 2, cudaMemcpyDeviceToHost));
        vsh::bitblt(dstpu, strideu, d->h_dst, d->d_pitch, w * sizeof(float), h);
        vsh::bitblt(dstpv, stridev, d->h_dst + plane, d->d_pitch, w * sizeof(float), h);

        vsapi->freeFrame(src);
        return dst;
    }
    return nullptr;
}

static void VS_CC chromanrcudaFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    auto d{static_cast<ChromanrcudaData*>(instanceData)};

    cudaFree(d->d_dst);
    cudaFreeHost(d->h_src);
    cudaFreeHost(d->h_dst);
    cudaDestroyTextureObject(d->tex);
    cudaFreeArray(d->d_src);

    vsapi->freeNode(d->node);
    delete d;
}

void VS_CC chromanrcudaCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
    auto d{std::make_unique<ChromanrcudaData>()};
    int err{};

    auto set_error = [&](const std::string& error_message) -> void {
        vsapi->freeNode(d->node);
        vsapi->mapSetError(out, ("chromanrcuda: " + error_message).c_str());
    };

    float threshold, threshold_y, threshold_u, threshold_v;

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    threshold = vsapi->mapGetFloatSaturated(in, "thres", 0, &err);
    if (err)
        threshold = 4.0f;

    threshold_y = vsapi->mapGetFloatSaturated(in, "threy", 0, &err);
    if (err)
        threshold_y = 20.0f;

    threshold_u = vsapi->mapGetFloatSaturated(in, "threu", 0, &err);
    if (err)
        threshold_u = 20.0f;

    threshold_v = vsapi->mapGetFloatSaturated(in, "threv", 0, &err);
    if (err)
        threshold_v = 20.0f;

    d->sizew = vsapi->mapGetIntSaturated(in, "sizew", 0, &err);
    if (err)
        d->sizew = 3;

    d->sizeh = vsapi->mapGetIntSaturated(in, "sizeh", 0, &err);
    if (err)
        d->sizeh = 3;

    d->stepw = vsapi->mapGetIntSaturated(in, "stepw", 0, &err);
    if (err)
        d->stepw = 1;

    d->steph = vsapi->mapGetIntSaturated(in, "steph", 0, &err);
    if (err)
        d->steph = 1;

    int distance = vsapi->mapGetIntSaturated(in, "distance", 0, &err);
    d->use_euclidean = !!distance;
    if (err)
        d->use_euclidean = false;

    if (d->vi->format.colorFamily != cfYUV)
        return set_error("only works with YUV format");

    if (d->vi->format.sampleType != stFloat)
        return set_error("only works with float format");

    if (threshold < 1.0f || threshold > 200.0f || threshold_y < 1.0f || threshold_y > 200.0f || 
        threshold_u < 1.0f || threshold_u > 200.0f || threshold_v < 1.0f || threshold_v > 200.0f){
        return set_error("\"thres\", \"threy\", \"threu\" and \"threv\" must be between 1.0 and 200.0");
    }

    if (d->sizew < 1 || d->sizew > 100 || d->sizeh < 1 || d->sizeh > 100)
        return set_error("\"sizew\" and \"sizeh\" must be between 1 and 100");

    if (d->stepw < 1 || d->stepw > 50 || d->steph < 1 || d->steph > 50)
        return set_error("\"stepw\" and \"steph\" must be between 1 and 50");

    if (distance < 0 || distance > 1)
        return set_error("\"distance\" must be \"0\" (manhattan) or \"1\" (euclidean)");

    if (d->stepw > d->sizew)
        return set_error("\"stepw\" cannot be bigger than \"sizew\"");

    if (d->steph > d->sizeh)
        return set_error("\"steph\" cannot be bigger than \"sizeh\"");

    d->chroma_ssw = d->vi->format.subSamplingW;
    d->chroma_ssh = d->vi->format.subSamplingH;
    d->thres = threshold / 255.0f;
    d->thres_y = threshold_y / 255.0f;
    d->thres_u = threshold_u / 255.0f;
    d->thres_v = threshold_v / 255.0f;

    const int w = d->vi->width >> d->chroma_ssw;
    const int h = d->vi->height >> d->chroma_ssh;
    d->d_pitch = w * sizeof(float);

    checkError(cudaMallocHost(&d->h_src, d->d_pitch * h * 3));
    checkError(cudaMallocHost(&d->h_dst, d->d_pitch * h * 2));
    checkError(cudaMalloc(&d->d_dst, d->d_pitch * h * 2));

    cudaChannelFormatDesc channelDesc;
    channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    checkError(cudaMallocArray(&d->d_src, &channelDesc, w, h * 3));

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = d->d_src;
    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;
    checkError(cudaCreateTextureObject(&d->tex, &texRes, &texDescr, NULL));

    VSFilterDependency deps[] = {{d->node, rpGeneral}};
    vsapi->createVideoFilter(out, "CNR", d->vi, chromanrcudaGetFrame, chromanrcudaFree, fmParallelRequests, deps, 1, d.get(), core);
    d.release();
}

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.julek.chromanrcuda", "chromanrcuda", "chromanr cuda filter", VS_MAKE_VERSION(1, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("CNR",
                             "clip:vnode;"
                             "thres:float:opt;"
                             "threy:float:opt;"
                             "threu:float:opt;"
                             "threv:float:opt;"
                             "sizew:int:opt;"
                             "sizeh:int:opt;"
                             "stepw:int:opt;"
                             "steph:int:opt;"
                             "distance:int:opt;",
                             "clip:vnode;", chromanrcudaCreate, nullptr, plugin);
}
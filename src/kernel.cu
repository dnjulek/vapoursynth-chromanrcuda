extern void process_cuda(float* d_dst, int w, int h, float thres, float thres_y, float thres_u, float thres_v, int sizew, int sizeh, int stepw, int steph, int chroma_ssw, int chroma_ssh,
                         bool use_euclidean, cudaTextureObject_t tex) noexcept;

__global__
__launch_bounds__(128)
static void cnr_tex(float* __restrict__ dst, int w, int h, float thres, float thres_y, float thres_u, float thres_v, int sizew, int sizeh, int stepw,
                                                      int steph, int chroma_ssw, int chroma_ssh, bool use_euclidean, cudaTextureObject_t tex) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int h2 = h * 2;
    const int yystart = y - sizeh;
    const int yystop = y + sizeh;
    const int xxstart = x - sizew;
    const int xxstop = x + sizew;
    const float cy = tex2D<float>(tex, x, y);
    const float cu = tex2D<float>(tex, x, y + h);
    const float cv = tex2D<float>(tex, x, y + h2);
    float su = cu;
    float sv = cv;
    int cn = 1;
    float distance;
    for (int yy = yystart; yy <= yystop; yy += steph) {
        for (int xx = xxstart; xx <= xxstop; xx += stepw) {
            const float Y = tex2D<float>(tex, xx, yy);
            const float U = tex2D<float>(tex, xx, yy + h);
            const float V = tex2D<float>(tex, xx, yy + h2);
            const float cyY = fabsf(cy - Y);
            const float cuU = fabsf(cu - U);
            const float cvV = fabsf(cv - V);
            if (use_euclidean) {
                distance = sqrtf(cyY * cyY + cuU * cuU + cvV * cvV);
            } else {
                distance = (cyY + cuU + cvV);
            }

            if (distance < thres && cuU < thres_u && cvV < thres_v && cyY < thres_y) {
                su += U;
                sv += V;
                cn++;
            }
        }
    }

    dst[y * w + x] = su / cn;
    dst[(y + h) * w + x] = sv / cn;
}

extern void process_cuda(float* d_dst, int w, int h, float thres, float thres_y, float thres_u, float thres_v, int sizew, int sizeh, int stepw, int steph, int chroma_ssw, int chroma_ssh,
                         bool use_euclidean, cudaTextureObject_t tex) noexcept {

    const dim3 block{32, 4};
    const dim3 grid{(w + block.x - 1) / block.x, 
                    (h + block.y - 1) / block.y};

    cnr_tex<<<grid, block, 0>>>(d_dst, w, h, thres, thres_y, thres_u, thres_v, sizew, sizeh, stepw, steph, chroma_ssw, chroma_ssh, use_euclidean, tex);
}

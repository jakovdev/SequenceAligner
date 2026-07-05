#ifndef BIO_ALIGN_H
#define BIO_ALIGN_H

struct input;
struct output;

bool align_cpu(struct input, struct output);
#ifndef USE_CUDA
#define align(in, out) align_cpu(in, out)
#else
bool align_cuda(struct input, struct output);
#define align(in, out) align_cuda(in, out)
#endif

#endif /* BIO_ALIGN_H */

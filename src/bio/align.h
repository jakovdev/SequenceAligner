#ifndef BIO_ALIGN_H
#define BIO_ALIGN_H

struct input;
struct output;

bool align_cpu(struct input, struct output);
extern bool cuda_disable;
bool align_cuda(struct input, struct output);
#define align(i, o) (cuda_disable ? align_cpu(i, o) : align_cuda(i, o))

#endif /* BIO_ALIGN_H */

#ifndef BIO_ALIGN_H
#define BIO_ALIGN_H

struct input;
struct output;

bool align_cpu(struct input, struct output);
extern bool no_cuda;
bool align_cuda(struct input, struct output);
#define align(in, out) (no_cuda ? align_cpu(in, out) : align_cuda(in, out))

#endif /* BIO_ALIGN_H */

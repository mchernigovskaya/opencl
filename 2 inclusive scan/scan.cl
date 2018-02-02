#define SWAP(a, b) {__local float * tmp=a; a=b; b=tmp;}
typedef double elem_type;

__kernel void scan_hillis_steele(__global const elem_type * input,
                                 __global elem_type * output,
                                 __local elem_type * a,
                                 __local elem_type * b) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int block_size = get_local_size(0);

    a[lid] = b[lid] = input[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 1; s < block_size; s <<= 1) {
        if (lid > (s - 1)) {
            b[lid] = a[lid] + a[lid - s];
        } else {
            b[lid] = a[lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(a, b);
    }

    output[gid] = a[lid];
}

__kernel void next_step(__global const elem_type * input,
                  __global const elem_type * add,
                  __global elem_type * output) {
    int gid = get_global_id(0);
    int block_size = get_local_size(0);
    output[gid] = input[gid] + add[gid / block_size];
}
typedef double elem_type;

__kernel void convolution(__global float * A,
                          __global float * B,
                          __global float * C,
                          int n, int m)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row >= n || col >= n)
          return;

    elem_type result = 0;
    int hm = m / 2;
    for (int i = -hm; i <= hm; ++i) {
        for (int j = -hm; j <= hm; ++j) {
            int new_row = row + i;
            int new_col = col + j;
            if (new_row >= 0 && new_row < n && new_col >= 0 && new_col < n) {
                result += A[new_row * n + new_col] * B[(hm + i) * m + hm + j];
            }
        }
    }
    C[row * n + col] = result;
}

import numpy as np
import math

record = {}


def div_tile_search_space(n, max_div):
    rst = []
    for i in range(1, min(n, max_div)):
        tile = math.ceil(n / i)
        if tile not in rst:
            rst.append(tile)
    return rst


def pow2_div_tile_search_space(n, max_div=8, pow2_upper=8):
    pow2_tiles = [1 << _ for _ in range(pow2_upper) if (1 << _) <= n * 2]
    div_tiles = div_tile_search_space(n, max_div)
    range_tiles = [_ for _ in range(2, min(n, max_div))]
    rst = set(pow2_tiles + div_tiles + range_tiles)
    return rst


def calc_max_c_density(n_c_tiles, c_tile, density, c_n_groups, n=100):
    if density == 1:
        return 1
    n_c_dy_group = math.ceil(n_c_tiles * c_tile / c_n_groups)
    key = (n_c_tiles, c_tile, density, c_n_groups)
    if key in record:
        return record[key]
    else:
        rand = np.random.rand(n, n_c_dy_group)
        valid = rand < density
        valid = valid.reshape(n, n_c_dy_group, 1) * np.ones([1, 1, c_n_groups])
        valid = valid.reshape(n, -1)[:, : int(n_c_tiles) * int(c_tile)].reshape(
            n, int(n_c_tiles), int(c_tile)
        )

        max_c_density = valid.sum(2).max(1).mean() / c_tile
        # rand = np.random.rand(n, int(n_c_tiles), int(c_tile))
        # max_c_density = (rand <= density).sum(2).max(1).mean() / c_tile
        # assert max_c_density>=density
        if max_c_density < density:
            max_c_density = density
        record[key] = max_c_density
        return max_c_density


def ceil_efficiency(x, parallel):
    return x / (math.ceil(x / parallel) * parallel)


def mem_concurrent_efficiency(n, interval, concurrent):
    """
    mem request example1:
        n=4, concurrent=3 [ddd][dxx]
        two data is invalid in every two conccurent requrests for 6 data
        efficiency is 4/6
    mem request example2:
        n=4 interval=2 concurrent=5 [ddddi]i[ddddi]i
        one data is invalid in each requrest for 5 data
        efficiency=4/5
    mem request example3:
        n=4 interval=2 concurrent=6 [ddddii][ddddii]
        two data is invalid in each conccurent requrest for 6 data
        efficiency= 4/6
    """
    if interval < 0:
        interval = 0
    if n > concurrent:
        return ceil_efficiency(n, concurrent)
    if n + interval > concurrent:
        return n / concurrent
    return n / (n + interval)


if __name__ == "__main__":
    for i in range(128):
        d = pow2_div_tile_search_space(i)
        print(d)

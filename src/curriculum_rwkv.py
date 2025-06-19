import math


class Curriculum:
    def __init__(self, args):
        # args.dims and args.points each contain start, end, inc, interval attributes
        # inc denotes the change in n_dims,
        # this change is done every interval,
        # and start/end are the limits of the parameter

        ##### n_dims_schedule contains four attributes: start, end, inc, interval
        self.n_dims_truncated = args.dims_start # for rwkv
        self.n_points = args.points_start # for rwkv
        self.dim_end = args.dims_end
        self.dim_inc = args.dims_inc
        self.dim_interval = args.dims_interval
        ##### n_points_schedule contains four attributes: start, end, inc, interval
        self.points_end = args.points_end
        self.points_inc = args.points_inc
        self.points_interval = args.points_interval

        self.step_count = 0

    def update(self):
        self.step_count += 1
        self.n_dims_truncated = self.update_var(
            self.n_dims_truncated, self.dim_interval, self.dim_inc, self.dim_end
        )

        self.n_points = self.update_var(
            self.n_points, self.points_interval, self.points_inc, self.points_end
        )

    def update_var(self, var, interval, inc, end):
        if self.step_count % interval == 0:
            var += inc
        return min(var, end)


# returns the final value of var after applying curriculum.
def get_final_var(init_var, total_steps, inc, n_steps, lim):
    final_var = init_var + math.floor((total_steps) / n_steps) * inc

    return min(final_var, lim)

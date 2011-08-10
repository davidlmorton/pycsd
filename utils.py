
def check_icsd_args(voltage_traces, depths, diameter, sigma_above):
    check_forward_solution_matrix_args(depths, diameter, sigma_above)
    assert voltage_traces.shape[0] == len(depths)
    assert len(voltage_traces.shape) in [1, 2]

def check_forward_solution_matrix_args(depths, diameter, sigma_above):
    assert len(depths) > 1
    assert sigma_above in ['zero', 'same', 'inf']
    assert diameter > 0.0

function value = compute_value_qp(P, q, x)
    value = x' * P * x / 2 + q' * x;
end
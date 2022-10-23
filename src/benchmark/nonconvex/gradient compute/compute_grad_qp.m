function grad = compute_grad_qp(P, q, x)
    grad = P * x + q;
end
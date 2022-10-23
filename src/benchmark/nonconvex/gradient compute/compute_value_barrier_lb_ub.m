function value = compute_value_barrier_lb_ub(P, q, A, b, x, t,lb,ub)
if ~isempty(b)
    if  (~all(A * x <= b))
        value = Inf;
    else
        value = compute_value_qp(P, q, x);
        value =  value - sum(log(b - A * x)) / t;
        value =  value + sum( log(1./(x - lb)).^2 + log(1./(ub - x)).^2  ) / t^2;
        
    end
else
    if  (any(x <= (lb))) ||  (any(x >= (ub )))
        value = Inf;
    else
        temp = x.* (P*x);
        value = compute_value_qp(P, q, x);
        value =  value + sum( (log(ub-temp)/t).^2 +  (log(temp-lb)/t).^2   );
    end
    
    
end
end
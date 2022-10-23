function value = compute_value_barrier_lb_ub_risk_parity(P, q, A, b, x, t,lb,ub,sigma)
n = size(P,1)-1;
if ~isempty(b)
    error('?');
    if  (~all(A * x <= b))
        value = Inf;
    else
        value = compute_value_qp(P, q, x);
        value =  value - sum(log(b - A * x)) / t;
        value =  value + sum( log(1./(x - lb)).^2 + log(1./(ub - x)).^2  ) / t^2;
        
    end
else
    xslice = x(1:n);
    Px = xslice.* (sigma * xslice);

    if  (any(Px <= (lb   )  )) ||  (any( Px >= (ub   ))) 
        value = Inf;
    else
        value = compute_value_qp(P, q, x);
        value =  value + sum( (log((Px - lb))/t).^2 + (log((ub - Px))/t).^2  );
    end
    
    
end
end
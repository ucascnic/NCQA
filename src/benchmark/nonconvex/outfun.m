 function stop = outfun(x,optimValues,state)
   global iter_val
   stop = false;
     switch state
         case 'iter'
          iter_val = [ iter_val; optimValues.fval];
     end
 end
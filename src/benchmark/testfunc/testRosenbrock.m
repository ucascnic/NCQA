 
 
clear
% solver = 1; %  spq
% solver = 2; %  ipm
% solver = 3 proposed 
% solver = 4 parallel

% repeat=30;
% ns = [100:100:1500 2000 2500 3000 4000   ];
% for solver =  4
%     rosentest(ns,repeat,solver)
% end
repeat=10;
ns = [100:100:1200 ];
for solver =  2
    rosentest(ns,repeat,solver)
end
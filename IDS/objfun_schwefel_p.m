function valor=objfun_schwefel_p(x,param)


for i=1:param.n
    valor=param.p*param.p;
end
valor=-x(1).*sin(abs(x(1)).^0.5)-x(2).*sin(abs(x(2)).^0.5);



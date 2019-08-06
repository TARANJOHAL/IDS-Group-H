function value=objfun_schwefel(x)

value=-x(1).*sin(abs(x(1)).^0.5)-x(2).*sin(abs(x(2)).^0.5);
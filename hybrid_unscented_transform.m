% Copyright (C) 2023 by Michael Hesse
function [mxx, Sxx] = hybrid_unscented_transform(mx, Sx, dynfcn_phy, ...
    dynfcn_gp, Bgp)
% - dimensions
nx = length(mx);
ny = size(Bgp, 2);

% - parameters
kappa = 3 - nx;
W = [repmat(1/(2*(nx + kappa)), 1, 2*nx), kappa/(nx + kappa)];

% - sigma points
Ls = chol((nx + kappa)*Sx);
Xs = repmat(mx, 1, 2*nx+1);
for i = 1 : nx
    Xs(:, i) = Xs(:, i) + Ls(i, :)';
    Xs(:, nx+i) = Xs(:, nx+i) - Ls(i, :)';
end

% - preallocation
if isfloat(mx)
    F = zeros(nx, 2*nx+1);
    Ym = zeros(ny, 2*nx+1);
    YS = zeros(ny, 2*nx+1);
else
    import casadi.*; %#ok<*SIMPT>
    F = SX.zeros(nx, 2*nx+1);
    Ym = SX.zeros(ny, 2*nx+1);
    YS = SX.zeros(ny, 2*nx+1);
end

% - evaluate model parts for all sigma points
for i = 1 : 2*nx+1
    F(:, i) = dynfcn_phy(Xs(:, i));
    [Ym(:, i), tmp] = dynfcn_gp(Xs(:, i));
    YS(:, i) = diag(tmp);
end

% - expectation
Ef = sum(repmat(W, nx, 1).*F, 2);
Eym = sum(repmat(W, ny, 1).*Ym, 2);
EyS = diag(sum(repmat(W, ny, 1).*YS, 2));
mxx = Ef + Bgp*Eym;

% - variance
Vf = zeros(nx, nx);
Vym = zeros(ny, ny);
Cfym = zeros(nx, ny);
for i = 1 : 2*nx+1
    Vf = Vf + W(i) * (F(:, i) - Ef)*(F(:, i) - Ef)';
    Vym = Vym + W(i) * (Ym(:, i) - Eym)*(Ym(:, i) - Eym)';
    Cfym = Cfym + W(i) * (F(:, i) - Ef)*(Ym(:, i) - Eym)';
end
Sxx = Bgp*EyS*Bgp' + Vf + Bgp*Vym*Bgp' + Cfym*Bgp' + Bgp*Cfym';
end

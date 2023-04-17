% Copyright (C) 2023 by Michael Hesse
function [traj, obj, theta, time, exitflag, output] = ...
    probabilistic_multiple_shoot_mean(t, mxI, SxI, xG, propfcn, ...
    bounds, weights, theta0, solver_iter)
% - add casadi
import casadi.*;

% - dimensions
nx = length(bounds.xmax);
nu = length(bounds.umax);
nt = length(t);

% - time discretization
dt = t(2);

% - intial distribution
LxI = chol(SxI)';
LxIv = chol2vec(LxI);

% - casadi variables
mxi = SX.sym('mxi', nx, nt);
mui = SX.sym('mui', nu, nt);
theta = [mxi(:)', mui(:)']';

% - control objective
Jui = diag(mui'*weights.Wu*mui);
Ju = dt*sum(Jui);

% - state objective
mxxi = SX.sym('mxxi', nx, nt);      % successor states
Lxvi = SX.sym('Lxvi', nx*(nx+1)/2, nt);
Lxvi(:, 1) = LxIv;
mJxi = SX.zeros(1, nt);
for i = 1 : nt
    % - unwrap
    mx = mxi(:, i);
    Lxv = Lxvi(:, i);
    Lx = vec2chol(Lxv);
    Sx = Lx*Lx';
    mu = mui(:, i);
    
    % - analytical mean
    mJxi(i) = weights.Wt(i)*( Sx(:)'*weights.Wx(:) + ...
        (xG-mx)'*weights.Wx*(xG-mx) );

    % - next state distribution
    [my, Sy] = propfcn(mx, Sx, mu);
    Ly = chol(Sy)';
    Lyv = chol2vec(Ly);
    
    % - append
    mxxi(:, i) = my;
    if i < nt
        Lxvi(:, i+1) = Lyv;
    end
end
Jx = dt*sum(mJxi);
J = Jx + Ju;

% - dynamical constraints
ceq = SX.zeros(nx*nt, 1);
for i = 1 : nt-1
    ceq((i-1)*nx+1 : i*nx) = mxi(:, i+1) - mxxi(:, i);  % sure?
end
ceq((nt-1)*nx+1 : nt*nx) = mxi(:, 1) - mxI;

% - state chance constraints
ixbound = find(~isinf(bounds.xmax));
Px = 0.95;
probfcn = @(x, m, s2) (1 + erf((x-m)./sqrt(2*s2)))/2;
c = SX.zeros(1, nt);
for i = 1 : nt
    % - unwrap
    mx = mxi(:, i);
    Lxv = Lxvi(:, i);
    Lx = vec2chol(Lxv);
    Sx = Lx*Lx';
    Sxii = diag(Sx);
    
    % - probability
    Pj = probfcn(bounds.xmax(ixbound), mx(ixbound), Sxii(ixbound)) - ...
        probfcn(bounds.xmin(ixbound), mx(ixbound), Sxii(ixbound));
    P = SX.ones(1, 1);
    for j = 1 : length(Pj)
        P = P*Pj(j);
    end
    
    % - append
    c(i) = -P + Px;
end
c = c(:);

% - gradient and jacobians
tic;
dJdtheta = gradient(J, theta);
dceqdtheta = jacobian(ceq, theta);
dcdtheta = jacobian(c, theta);
time1 = toc;

% - objective and constraints
objective_casadi = Function('main', {theta}, ...
    {J, dJdtheta, Jx, Ju, mxi, Lxvi});
objfcn = returntypes('full', objective_casadi);

constraints_casadi = Function('main', {theta}, ...
    {c, ceq, dcdtheta', dceqdtheta'});
confcn = returntypes('full', constraints_casadi);

% - initial guess
if isempty(theta0)
    mxi0 = mxI + (xG-mxI)*(0:nt-1)/(nt-1);
    mui0 = zeros(nu, nt);
    theta0 = [mxi0(:)', mui0(:)']';
end

% - linear constraints
mxlb = repmat(-inf*ones(nx, 1), 1, nt);
mxub = repmat(inf*ones(nx, 1), 1, nt);
mulb = repmat(bounds.umin, 1, nt);
muub = repmat(bounds.umax, 1, nt);
thetalb = [mxlb(:)', mulb(:)']';
thetaub = [mxub(:)', muub(:)']';

% - optimization
% 'interior-point' 'sqp' 'active-set'
% 'off' 'iter' 'iter-detailed' 'final' 'final-detailed'
opts = optimoptions('fmincon', 'Algorithm', 'sqp', ...
                    'Display', 'final-detailed', ...
                    'SpecifyObjectiveGradient', true, ...
                    'SpecifyConstraintGradient', true, ...
                    'MaxIterations', solver_iter, ...
                    'MaxFunctionEvaluations', inf, ...
                    'ConstraintTolerance', 1e-6, ...
                    'OptimalityTolerance', 1e-6, 'StepTolerance', 1e-3);
tic;
[theta, ~, exitflag, output] = fmincon(objfcn, theta0, [], [], [], ...
    [], thetalb, thetaub, confcn, opts);
time2 = toc;

% - evaluate objective parts
[~, ~, Jx, Ju, ~, Lxvi] = objfcn(theta);

% - unwrap
mxi = reshape(theta(1:nx*nt), nx, nt);
mui = reshape(theta(nx*nt+1:nx*nt+nu*nt), nu, nt);
sxi = zeros(nx, nt);
Sxj = zeros(nx, nx, nt);
for i = 1 : nt
    Lxi = vec2chol(Lxvi(:, i));
    Sxi = Lxi*Lxi';
    sxi(:, i) = sqrt(diag(Sxi)');
    Sxj(:, :, i) = Sxi;
end

% - wrap
traj.mu = mui;
traj.mx = mxi;
traj.sx = sxi;
traj.Sx = Sxj;

obj.Jx = Jx;
obj.Ju = Ju;

time.diff = time1;
time.solve = time2;
end



% - subfunctions
function Lv = chol2vec(L)
import casadi.*;
[n, ~] = size(L);
idx = reshape(1:n^2, n, n);
mask = idx(tril(true(n, n)));
Lv = L(mask);
end

function L = vec2chol(Lv)
n = (sqrt(8*length(Lv) + 1) - 1)/2;
idx = reshape(1:n^2, n, n);
mask = idx(tril(true(n, n)));
if isfloat(Lv)
    L = zeros(n, n);
else    % casadi variable
    import casadi.*;    %#ok<*SIMPT>
    L = SX.zeros(n, n);
end
L(mask) = Lv;
end

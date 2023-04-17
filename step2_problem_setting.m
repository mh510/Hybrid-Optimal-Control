% Copyright (C) 2023 by Michael Hesse
% - fix random number generator
seed = 510;
rng(seed);

% - dimensions
nq = 2;
nx = 2*nq;
nu = 2;

% - time discretization
dt = 0.05;
T = 2;
t = 0 : dt : T;
nt = length(t);

% - plant
% l1, l2, m1, m2, d1, d2, g, bin
plant.p = [1, 1, 0.5, 0.5, 0.05, 0.05, 9.81, 1]';
plant.odefcn = @(x, u) ode_double_pend(x, u, plant.p);
plant.dynfcn = @(x, u) x + dt*plant.odefcn(x, u);

% - model
model.p = [plant.p(1:end-1); 0];
model.odefcn = @(x, u) ode_double_pend(x, u, model.p);
model.dynfcn = @(x, u) x + dt*model.odefcn(x, u);

% - dummy model
dummy_model.odefcn = @(x, u) [x(nq+1:nx, :); zeros(nq, size(x, 2))];
dummy_model.dynfcn = @(x, u) x + dt*dummy_model.odefcn(x, u);

% - initial state distribution and goal state
mxI = [pi, pi, 0, 0]';
SxI = 0.001*eye(nx);
xG = [0, 0, 0, 0]';

% - bounds
bounds.xmax = [4, 4, 2, 2]';
bounds.xmin = [-1, -1, -4, -4]';
bounds.umax = [10, 10]';
bounds.umin = [-10, -10]';

% - weights
weights.Wx = diag([25, 25, 1, 1]);
weights.Wu = 0.01;
weights.gamma = 3;
weights.Wt = (exp(weights.gamma*t) - 1)/(exp(weights.gamma*T) - 1);
weights.thres = 0.05;
weights.Wt(weights.Wt <= weights.thres) = 0;    % prune irrelevant weights

% - initial (constant) gaussian process
mgp0 = zeros(nq, 1);
Sgp0 = zeros(nq);
pred0fcn = @(x) deal(mgp0, Sgp0);

% - gaussian process
Bgp = [zeros(nq, nq), eye(nq)]';
ni = 50;    % number of inducing inputs

% - number of iterations
solver_iter = 300;
learn_iter = 10;

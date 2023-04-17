% Copyright (C) 2023 by Michael Hesse
% - cleanup
clear;
close all;
clc;

% - load setting
step2_problem_setting;

% - define scenario
% foldername = 'w_prior';
% mkdir(foldername);
% 
% choose_kernel = 1;      % ardse: 1, ardmatern32: 2
% choose_solver = 3;      % shoot- single: 1, multi (mean): 2, multi: 3
% use_prior = true;

% foldername = 'wo_prior';
% mkdir(foldername);
% 
% choose_kernel = 1;      % ardse: 1, ardmatern32: 2
% choose_solver = 3;      % shoot- single: 1, multi (mean): 2, multi: 3
% use_prior = false;

foldername = 'single_shoot';
mkdir(foldername);

choose_kernel = 1;      % ardse: 1, ardmatern32: 2
choose_solver = 1;      % shoot- single: 1, multi (mean): 2, multi: 3
use_prior = true;

% - initial trial
if use_prior
    fprintf('*** Solving initial optimal control problem. ***\n');
    propfcn = @(mx, Sx, u) hybrid_unscented_transform(...
        mx, Sx, @(x) model.dynfcn(x, u), pred0fcn, Bgp);
    if choose_solver == 1
        [traj, obj, theta, time, exitflag, output] = ...
            probabilistic_single_shoot(t, mxI, SxI, xG, propfcn, ...
            bounds, weights, [], solver_iter);
    elseif choose_solver == 2
        [traj, obj, theta, time, exitflag, output] = ...
            probabilistic_multiple_shoot_mean(t, mxI, SxI, xG, propfcn, ...
            bounds, weights, [], solver_iter);
    elseif choose_solver == 3
        [traj, obj, theta, time, exitflag, output] = ...
            probabilistic_multiple_shoot(t, mxI, SxI, xG, propfcn, ...
            bounds, weights, [], solver_iter);
    end
else
    fprintf('*** Random initial trial. ***\n'); %#ok<*UNRCH> 
    traj.mx = zeros(nx, nt);
    traj.sx = zeros(nx, nt);
    traj.Sx = zeros(nx, nx, nt);
    traj.mu = bounds.umin + (bounds.umax - bounds.umin) .* rand(nu, nt);
    obj.Jx = NaN;
    obj.Ju = dt*sum(diag(traj.mu'*weights.Wu*traj.mu));
    theta = [];
    time.diff = 0;
    time.solve = 0;
    model = dummy_model;
end

x = zeros(nx, nt);
x(:, 1) = mxI;
for k = 1 : nt-1
    x(:, k+1) = plant.dynfcn(x(:, k), traj.mu(:, k));
end
Jx = dt*sum(weights.Wt.*diag((xG - x)'*weights.Wx*(xG - x))');
J = Jx + obj.Ju;
% plot_traj(t, x, traj, 0);

X = x;
U = traj.mu;
Y = Bgp'*(plant.dynfcn(X, U) - model.dynfcn(X, U));

% - save workspace
save([foldername, '\workspace_', num2str(0), '.mat']);

% - learning loop
for i = 1 : learn_iter
    % - training
    fprintf('*** Training Gaussian process. ***\n');
    if use_prior
        Z = X;
    else
        Z = [X; U];
    end
    if size(X, 2) <= ni
        gp = train_gaussian_process(Z, Y, choose_kernel);
    else
        gp = train_sparse_gaussian_process(Z, Y, choose_kernel, ni);
    end

    % - optimization
    fprintf('*** Solving hybrid optimal control problem %g/%g. ***\n', ...
        i, learn_iter);
    if use_prior
        propfcn = @(mx, Sx, u) hybrid_unscented_transform(...
            mx, Sx, @(x) model.dynfcn(x, u), @(x) gp.predfcn(x), Bgp);
    else
        propfcn = @(mx, Sx, u) hybrid_unscented_transform(...
            mx, Sx, @(x) model.dynfcn(x, u), @(x) gp.predfcn([x; u]), Bgp);
    end
    if choose_solver == 1
        [traj, obj, theta, time, exitflag, output] = ...
            probabilistic_single_shoot(t, mxI, SxI, xG, propfcn, ...
            bounds, weights, theta, solver_iter);
    elseif choose_solver == 2
        [traj, obj, theta, time, exitflag, output] = ...
            probabilistic_multiple_shoot_mean(t, mxI, SxI, xG, propfcn, ...
            bounds, weights, theta, solver_iter);
    elseif choose_solver == 3
        [traj, obj, theta, time, exitflag, output] = ...
            probabilistic_multiple_shoot(t, mxI, SxI, xG, propfcn, ...
            bounds, weights, theta, solver_iter);
    end

    % - application
    x = zeros(nx, nt);
    x(:, 1) = mxI;
    for k = 1 : nt-1
        x(:, k+1) = plant.dynfcn(x(:, k), traj.mu(:, k));
    end
    Jx = dt*sum(weights.Wt.*diag((xG - x)'*weights.Wx*(xG - x))');
    J = Jx + obj.Ju;
%     plot_traj(t, x, traj, i);

    % - data
    X = [X, x]; %#ok<*AGROW> 
    U = [U, traj.mu];
    Y = Bgp'*(plant.dynfcn(X, U) - model.dynfcn(X, U));

    % - save workspace
    save([foldername, '\workspace_', num2str(i), '.mat']);
end



% - subfunctions
function plot_traj(t, x, traj, iter)
figure();
title(sprintf('Iteration: %g', iter));
for i = 1 : size(x, 1)
    subplot(3, 2, i);
    plot(t, traj.mx(i, :) + [0, 2, -2]'.*traj.sx(i, :), 'b-');
    grid on;
    hold on;
    plot(t, x(i, :), 'r--');
end
for i = 1 : size(traj.mu, 1)
    subplot(3, 2, size(x, 1)+i);
    plot(t, traj.mu(i, :), 'b-');
    grid on;
end
end

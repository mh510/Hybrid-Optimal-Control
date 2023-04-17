% Copyright (C) 2023 by Michael Hesse
% - cleanup
clear;
close all;
clc;

% - load setting
step2_problem_setting;

% - solve probabilistic control problem
choose_solver = 3;      % shoot- single: 1, multi (mean): 2, multi: 3
solver_iter = 3000;
propfcn = @(mx, Sx, u) hybrid_unscented_transform(...
    mx, Sx, @(x) plant.dynfcn(x, u), pred0fcn, Bgp);
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

% - export ground truth
ground_truth.traj = traj;
ground_truth.obj = obj;
save('ground_truth.mat', 'ground_truth');

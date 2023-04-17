% Copyright (C) 2023 by Michael Hesse
% - cleanup
clear;
close all;
clc;

% - load settings
step2_problem_setting;

% - plotting properties
linewidth = 1.5;
fontsize = 16;
markersize = 10;

% - load ground truth
load('ground_truth.mat', 'ground_truth');

% - objective
plot_data.wprior.Jx.pred = zeros(1, learn_iter+1);
plot_data.wprior.Jx.real = zeros(1, learn_iter+1);
plot_data.wprior.Ju = zeros(1, learn_iter+1);
plot_data.wprior.issafe = false(1, learn_iter+1);

plot_data.woprior.Jx.pred = zeros(1, learn_iter+1);
plot_data.woprior.Jx.real = zeros(1, learn_iter+1);
plot_data.woprior.Ju = zeros(1, learn_iter+1);
plot_data.woprior.issafe = false(1, learn_iter+1);

plot_data.single.Jx.pred = zeros(1, learn_iter+1);
plot_data.single.Jx.real = zeros(1, learn_iter+1);
plot_data.single.Ju = zeros(1, learn_iter+1);
plot_data.single.issafe = false(1, learn_iter+1);
for i = 0 : learn_iter
    % - w/ prior
    load(['w_prior\workspace_', num2str(i), '.mat']);
    plot_data.wprior.Jx.pred(i+1) = obj.Jx;
    plot_data.wprior.Jx.real(i+1) = Jx;
    plot_data.wprior.Ju(i+1) = obj.Ju;
    plot_data.wprior.issafe(i+1) = all(bounds.xmin <= x, 'all') && ...
        all(x <= bounds.xmax, 'all');

    % - w/o prior
    load(['wo_prior\workspace_', num2str(i), '.mat']);
    plot_data.woprior.Jx.pred(i+1) = obj.Jx;
    plot_data.woprior.Jx.real(i+1) = Jx;
    plot_data.woprior.Ju(i+1) = obj.Ju;
    plot_data.woprior.issafe(i+1) = all(bounds.xmin <= x, 'all') && ...
        all(x <= bounds.xmax, 'all');

    % - single
    load(['single_shoot\workspace_', num2str(i), '.mat']);
    plot_data.single.Jx.pred(i+1) = obj.Jx;
    plot_data.single.Jx.real(i+1) = Jx;
    plot_data.single.Ju(i+1) = obj.Ju;
    plot_data.single.issafe(i+1) = all(bounds.xmin <= x, 'all') && ...
        all(x <= bounds.xmax, 'all');

end
iter = 0:learn_iter;

figure(1);
subplot(2, 1, 1);
grid on;
hold on;
% plot(iter, ones(1, length(iter))*log(ground_truth.obj.Jx), 'k-');

plot(iter, log(plot_data.wprior.Jx.pred), 'bo-', 'LineWidth', linewidth);
plot(iter, log(plot_data.wprior.Jx.real), 'bo--', 'LineWidth', linewidth);

plot(iter, log(plot_data.woprior.Jx.pred), 'ro-', 'LineWidth', linewidth);
plot(iter, log(plot_data.woprior.Jx.real), 'ro--', 'LineWidth', linewidth);

plot(iter(plot_data.wprior.issafe), ...
    log(plot_data.wprior.Jx.real(plot_data.wprior.issafe)), 'gs', ...
    'LineWidth', 2, 'MarkerSize', 10, 'HandleVisibility', 'off');
plot(0, 0, 'w.');       % adjusts legend entry
plot(iter(plot_data.woprior.issafe), ...
    log(plot_data.woprior.Jx.real(plot_data.woprior.issafe)), 'gs', ...
    'LineWidth', 2, 'MarkerSize', 10);

% plot(iter, log(plot_data.single.Jx.pred), 'co-', 'LineWidth', linewidth);
% plot(iter, log(plot_data.single.Jx.real), 'co--', 'LineWidth', linewidth);

axis([0, learn_iter, -0.5, 6]);
xlabel('Learning iteration', 'Interpreter', 'latex');
ylabel('$\log \textrm{E}[J_x]$', 'Interpreter', 'latex');
legend('Pred. w/ prior', 'Real. w/ prior', ...
    'Pred. w/o prior', 'Real. w/o prior', ' ', 'Const. satisfied', ...
    'location', 'best', 'Interpreter', 'latex', 'NumColumns', 2, ...
    'Orientation', 'horizontal');
legend boxoff;
set(gca, 'fontsize', fontsize);

subplot(2, 1, 2);
grid on;
hold on;
% plot(iter, ones(1, length(iter))*log(ground_truth.obj.Ju), 'k-');
plot(iter, log(plot_data.wprior.Ju), 'bo-', 'LineWidth', linewidth);
plot(iter, log(plot_data.woprior.Ju), 'ro-', 'LineWidth', linewidth);

% plot(iter, log(plot_data.single.Ju), 'co-', 'LineWidth', linewidth);

axis([0, learn_iter, 0, 1.1]);
xlabel('Learning iteration', 'Interpreter', 'latex');
ylabel('$\log J_u$', 'Interpreter', 'latex');
set(gca, 'fontsize', fontsize);

% - trajectories
plot_every_nth_ellipse = 1;
linewidth_ellipse = 0.5;
for i = 0 : learn_iter
    % - new figure
    figure(2+i);
    for j = 1 : nx
        subplot(2, 2, j);
        hold on;
        grid on;
    end

    % - w/ prior
    load(['w_prior\workspace_', num2str(i), '.mat']);
    subplot(2, 2, 1);
    plot(ground_truth.traj.mx(1, :), ...
        ground_truth.traj.mx(2, :), 'g-', 'LineWidth', linewidth);
    for k = 1 : plot_every_nth_ellipse : nt
        plot_ellipse(traj.mx(1:2, k), traj.Sx(1:2, 1:2, k), 2, ...
            'b-', linewidth_ellipse);
    end
    pl(1) = plot(traj.mx(1, 1), traj.mx(2, 1), 'b-', ...
        'LineWidth', linewidth);
    pl(2) = plot(x(1, :), x(2, :), 'r-', 'LineWidth', linewidth);
    plot([bounds.xmin(1), bounds.xmin(1)], ...
        [bounds.xmin(2), bounds.xmax(2)], 'k--', 'LineWidth', linewidth);
    plot([bounds.xmax(1), bounds.xmax(1)], ...
        [bounds.xmin(2), bounds.xmax(2)], 'k--', 'LineWidth', linewidth);
    plot([bounds.xmin(1), bounds.xmax(1)], ...
        [bounds.xmin(2), bounds.xmin(2)], 'k--', 'LineWidth', linewidth);
    plot([bounds.xmin(1), bounds.xmax(1)], ...
        [bounds.xmax(2), bounds.xmax(2)], 'k--', 'LineWidth', linewidth);

    axis([-1.5, 4.5, -1.5, 4.5]);
    title('w/ prior', 'Interpreter', 'latex');
    xlabel('$\varphi_1$ [rad]', 'Interpreter', 'latex');
    ylabel('$\varphi_2$ [rad]', 'Interpreter', 'latex');
    legend(pl, 'Prediction', 'Reality', 'Location', ...
        'best', 'Interpreter', 'latex');
    set(gca, 'fontsize', fontsize);
    legend boxoff;

    subplot(2, 2, 3);
    plot(ground_truth.traj.mx(3, :), ...
        ground_truth.traj.mx(4, :), 'g-', 'LineWidth', linewidth);
    for k = 1 : plot_every_nth_ellipse : nt
        plot_ellipse(traj.mx(3:4, k), traj.Sx(3:4, 3:4, k), 2, ...
            'b-', linewidth_ellipse);
    end
    plot(x(3, :), x(4, :), 'r-', 'LineWidth', linewidth);
    plot([bounds.xmin(3), bounds.xmin(3)], ...
        [bounds.xmin(4), bounds.xmax(4)], 'k--', 'LineWidth', linewidth);
    plot([bounds.xmax(3), bounds.xmax(3)], ...
        [bounds.xmin(4), bounds.xmax(4)], 'k--', 'LineWidth', linewidth);
    plot([bounds.xmin(3), bounds.xmax(3)], ...
        [bounds.xmin(4), bounds.xmin(4)], 'k--', 'LineWidth', linewidth);
    plot([bounds.xmin(3), bounds.xmax(3)], ...
        [bounds.xmax(4), bounds.xmax(4)], 'k--', 'LineWidth', linewidth);

    axis([-4.5, 3, -4.5, 2.5]);
    title('w/ prior', 'Interpreter', 'latex');
    xlabel('$\dot{\varphi}_1$ [rad/s]', 'Interpreter', 'latex');
    ylabel('$\dot{\varphi}_2$ [rad/s]', 'Interpreter', 'latex');
    set(gca, 'fontsize', fontsize);

    % - w/o prior
    load(['wo_prior\workspace_', num2str(i), '.mat']);
    subplot(2, 2, 2);
    pl2(1) = plot(ground_truth.traj.mx(1, :), ...
        ground_truth.traj.mx(2, :), 'g-', 'LineWidth', linewidth);
    for k = 1 : plot_every_nth_ellipse : nt
        plot_ellipse(traj.mx(1:2, k), traj.Sx(1:2, 1:2, k), 2, ...
            'b-', linewidth_ellipse);
    end
    plot(x(1, :), x(2, :), 'r-', 'LineWidth', linewidth);
    plot([bounds.xmin(1), bounds.xmin(1)], ...
        [bounds.xmin(2), bounds.xmax(2)], 'k--', 'LineWidth', linewidth);
    plot([bounds.xmax(1), bounds.xmax(1)], ...
        [bounds.xmin(2), bounds.xmax(2)], 'k--', 'LineWidth', linewidth);
    plot([bounds.xmin(1), bounds.xmax(1)], ...
        [bounds.xmin(2), bounds.xmin(2)], 'k--', 'LineWidth', linewidth);
    plot([bounds.xmin(1), bounds.xmax(1)], ...
        [bounds.xmax(2), bounds.xmax(2)], 'k--', 'LineWidth', linewidth);

    axis([-1.5, 4.5, -1.5, 4.5]);
    title('w/o prior', 'Interpreter', 'latex');
    xlabel('$\varphi_1$ [rad]', 'Interpreter', 'latex');
    ylabel('$\varphi_2$ [rad]', 'Interpreter', 'latex');
    legend(pl2, ['Ground', newline, 'Truth'], 'Location', ...
        'best', 'Interpreter', 'latex');
    set(gca, 'fontsize', fontsize);
    legend boxoff;

    subplot(2, 2, 4);
    plot(ground_truth.traj.mx(3, :), ground_truth.traj.mx(4, :), 'g-', ...
        'LineWidth', linewidth);
    for k = 1 : plot_every_nth_ellipse : nt
        plot_ellipse(traj.mx(3:4, k), traj.Sx(3:4, 3:4, k), 2, ...
            'b-', linewidth_ellipse);
    end
    plot(x(3, :), x(4, :), 'r-', 'LineWidth', linewidth);
    plot([bounds.xmin(3), bounds.xmin(3)], ...
        [bounds.xmin(4), bounds.xmax(4)], 'k--', 'LineWidth', linewidth);
    plot([bounds.xmax(3), bounds.xmax(3)], ...
        [bounds.xmin(4), bounds.xmax(4)], 'k--', 'LineWidth', linewidth);
    plot([bounds.xmin(3), bounds.xmax(3)], ...
        [bounds.xmin(4), bounds.xmin(4)], 'k--', 'LineWidth', linewidth);
    plot([bounds.xmin(3), bounds.xmax(3)], ...
        [bounds.xmax(4), bounds.xmax(4)], 'k--', 'LineWidth', linewidth);

    axis([-4.5, 3, -4.5, 2.5]);
    title('w/o prior', 'Interpreter', 'latex');
    xlabel('$\dot{\varphi}_1$ [rad/s]', 'Interpreter', 'latex');
    ylabel('$\dot{\varphi}_2$ [rad/s]', 'Interpreter', 'latex');
    set(gca, 'fontsize', fontsize);
end

% - control
figure(255);
for i = 1 : 4
    subplot(2, 2, i);
    hold on;
    grid on;
    if i == 1 || i == 2 % u1
        plot(t, ones(1, nt)*bounds.umin(1), 'k--', 'LineWidth', linewidth);
        plot(t, ones(1, nt)*bounds.umax(1), 'k--', 'LineWidth', linewidth);
        plot(t, ground_truth.traj.mu(1, :), 'g-', 'LineWidth', linewidth+2);
    elseif i == 3 || i == 4 % u2
        plot(t, ones(1, nt)*bounds.umin(2), 'k--', 'LineWidth', linewidth);
        plot(t, ones(1, nt)*bounds.umax(2), 'k--', 'LineWidth', linewidth);
        plot(t, ground_truth.traj.mu(2, :), 'g-', 'LineWidth', linewidth+2);
    end
    if i == 1
        title('w/ prior', 'Interpreter', 'latex');
        xlabel('$t$ [s]', 'Interpreter', 'latex');
        ylabel('$u_1$ [Nm]', 'Interpreter', 'latex');
        axis([t(1), t(end), 1.1*bounds.umin(1), 1.1*bounds.umax(1)]);
        set(gca, 'fontsize', fontsize);
    elseif i == 2
        title('w/o prior', 'Interpreter', 'latex');
        xlabel('$t$ [s]', 'Interpreter', 'latex');
        ylabel('$u_1$ [Nm]', 'Interpreter', 'latex');
        axis([t(1), t(end), 1.1*bounds.umin(1), 1.1*bounds.umax(1)]);
        set(gca, 'fontsize', fontsize);
    elseif i == 3
        title('w/ prior', 'Interpreter', 'latex');
        xlabel('$t$ [s]', 'Interpreter', 'latex');
        ylabel('$u_2$ [Nm]', 'Interpreter', 'latex');
        axis([t(1), t(end), 1.1*bounds.umin(2), 1.1*bounds.umax(2)]);
        set(gca, 'fontsize', fontsize);
    elseif i == 4
        title('w/o prior', 'Interpreter', 'latex');
        xlabel('$t$ [s]', 'Interpreter', 'latex');
        ylabel('$u_2$ [Nm]', 'Interpreter', 'latex');
        axis([t(1), t(end), 1.1*bounds.umin(2), 1.1*bounds.umax(2)]);
        set(gca, 'fontsize', fontsize);
    end
end

for i = 0 : learn_iter
    % - w/ prior
    load(['w_prior\workspace_', num2str(i), '.mat']);
    subplot(2, 2, 1);
    plot(t, traj.mu(1, :), '-', 'Color', [1, 0, 0]*i/learn_iter, ...
        'LineWidth', linewidth);

    subplot(2, 2, 3);
    plot(t, traj.mu(2, :), '-', 'Color', [1, 0, 0]*i/learn_iter, ...
        'LineWidth', linewidth);

    % - w/o prior
    load(['wo_prior\workspace_', num2str(i), '.mat']);
    if i == 0
        subplot(2, 2, 2);
        plot(t, traj.mu(1, :), '-', 'Color', [0.7, 0.7, 0.7], ...
            'LineWidth', linewidth);
    
        subplot(2, 2, 4);
        plot(t, traj.mu(2, :), '-', 'Color', [0.7, 0.7, 0.7], ...
            'LineWidth', linewidth);
    else
        subplot(2, 2, 2);
        plot(t, traj.mu(1, :), '-', 'Color', [1, 0, 0]*i/learn_iter, ...
            'LineWidth', linewidth);
    
        subplot(2, 2, 4);
        plot(t, traj.mu(2, :), '-', 'Color', [1, 0, 0]*i/learn_iter, ...
            'LineWidth', linewidth);
    end

end

% - time
timings_diff = zeros(2, learn_iter+1);
timings_solve = zeros(2, learn_iter+1);
for i = 0 : learn_iter
    % - multi shoot
    load(['w_prior\workspace_', num2str(i), '.mat']);
    timings_diff(1, i+1) = time.diff;
    timings_solve(1, i+1) = time.solve;

    % - single shoot
    load(['single_shoot\workspace_', num2str(i), '.mat']);
    timings_diff(2, i+1) = time.diff;
    timings_solve(2, i+1) = time.solve;
end

figure(256);
subplot(2, 1, 1);
b = bar(iter, timings_diff);
grid on;

% set(gca,'YScale','log');

b(1).FaceColor = 'b';
b(2).FaceColor = 'r';
title('Automatic Differentiation', 'Interpreter', 'latex');
xlabel('Learning iteration', 'Interpreter', 'latex');
ylabel('Time [s]', 'Interpreter', 'latex');
set(gca, 'fontsize', fontsize);
box off;

subplot(2, 1, 2);
b = bar(iter, timings_solve);
grid on;

% set(gca,'YScale','log');

b(1).FaceColor = 'b';
b(2).FaceColor = 'r';
legend('Multiple', 'Single', 'Interpreter', 'latex');
title('SQP Optimization', 'Interpreter', 'latex');
xlabel('Learning iteration', 'Interpreter', 'latex');
ylabel('Time [s]', 'Interpreter', 'latex');
legend boxoff;
set(gca, 'fontsize', fontsize);
box off;





% - subfunction
function plot_ellipse(means, C, sdwidth, style, linewidth)
if nargin < 3; sdwidth = 2; end
if nargin < 4; style = 'b-'; end
if nargin < 5; linewidth = 0.5; end
npts = 50;
tt = linspace(0, 2*pi, npts)';
x = cos(tt);
y = sin(tt);
ap = [x(:), y(:)]';
[v, d] = eig(C);
d = sdwidth * sqrt(d);      % convert variance to sdwidth*sd
bp = (v*d*ap) + repmat(means, 1, size(ap, 2));
plot(bp(1, :), bp(2, :), style, 'LineWidth', linewidth);
end

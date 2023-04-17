% Copyright (C) 2023 by Michael Hesse
% - cleanup
clear;
close all;
clc;

% - symbolic variables
syms t Mu1 Mu2 real
syms l1 l2 m1 m2 d1 d2 g real
syms phi1(t) phi2(t)
[subtable, resubtable] = get_subtables;

% - center of mass coordinates, angles and velocities
C(:, 1) = [l1*sin(phi1); l1*cos(phi1)];
C(:, 2) = [l1*sin(phi1)+l2*sin(phi2); l1*cos(phi1)+l2*cos(phi2)];
Cp = diff(C, t);
P(1, :) = [phi1, phi2];     % (1, :) matters
Pp = diff(P, t);

% - kinetic, potential energy and lagrange function
Jp = diag([m1*l1^2, m2*l2^2]);      % moment of inertia
M = diag([m1, m2]);
T = 0.5*sum(diag(Cp*M*Cp.')) + 0.5*sum(diag(Pp*Jp*Pp.'));   % .' matters
U = g*sum(C(2, :)*M);
L = T - U;

% - generalized forces
Q = [Mu1 - d1*Pp(1) + d2*(Pp(2)-Pp(1)), Mu2 - d2*(Pp(2)-Pp(1))].';

% - lagrange formalism
eqs = lagrange_formalism({'phi1'; 'phi2'}, L, Q, t, subtable, resubtable);

% - time-invariant notation
syms phi1 phi1p phi1pp phi2 phi2p phi2pp real
eqs = subs(eqs, subtable(1, :), subtable(2, :));
q = [phi1; phi2];
qp = [phi1p; phi2p];
qpp = [phi1pp; phi2pp];

% - mass matrix and right-hand-side
[Mass, RHS] = equationsToMatrix(eqs, qpp);
Mass = simplify(Mass);
iMass = inv(Mass);

% - add binary (on/off) varialbe for centrifugal forces
syms bin real
RHS(1) = subs(RHS(1), -l1*l2*m2*sin(phi1-phi2)*phi2p^2, ...
    -bin*l1*l2*m2*sin(phi1-phi2)*phi2p^2);
RHS(2) = subs(RHS(2), l1*l2*m2*sin(phi1 - phi2)*phi1p^2, ...
    bin*l1*l2*m2*sin(phi1 - phi2)*phi1p^2);

% - acceleration
qpp = Mass\RHS;

% - split right-hand side
terms = children(RHS)';

% - state-space
x = [q; qp];
u = [Mu1; Mu2];
p = [l1; l2; m1; m2; d1; d2; g; bin];
f = simplify([qp; qpp]);

% - ode function
matlabFunction(f, 'File', 'ode_double_pend.m', 'Optimize', true, ...
    'Vars', {x, u, p});






% - subfunctions
function [subtable, resubtable] = get_subtables
% - substituation table (note: order matters, highest derivative first)
subtable = str2sym({'diff(phi1(t),t,t)', 'phi1pp';
                    'diff(phi1(t),t)',   'phi1p';
                    'phi1(t)',           'phi1';
                    'diff(phi2(t),t,t)', 'phi2pp';
                    'diff(phi2(t),t)',   'phi2p';
                    'phi2(t)',           'phi2'}');
resubtable = fliplr(subtable);
end

function eqs = lagrange_formalism(qs, L, Q, t, subtable, resubtable)
Ls = subs(L, subtable(1, :), subtable(2, :));
eqs = sym(zeros(size(qs)));
for i = 1 : length(qs)
    dLdq = diff(Ls, qs{i});
    dLdq = subs(dLdq, resubtable(2, :), resubtable(1, :));
    
    dLdqp = diff(Ls, [qs{i} 'p']);
    dLdqp = subs(dLdqp, resubtable(2, :), resubtable(1, :));
    ddLdqpdt = diff(dLdqp, t);
    
    eqs(i, 1) = simplify(ddLdqpdt - dLdq - Q(i)) == 0;
end
end

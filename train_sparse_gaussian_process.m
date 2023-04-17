% Copyright (C) 2023 by Michael Hesse
function gpmodel = train_sparse_gaussian_process(X, dY, choose_kernel, ni)
% - get dimensions
[nx, nd] = size(X); %#ok<*ASGLU> 
[ny, ~] = size(dY);

% - preallocate space
gpmodel.postfcn = cell(1, ny);
gpmodel.hyp = zeros(nx+2, ny);
gpmodel.induce = zeros(nx, ni, ny);

% - train one gaussian process for each output dimension
for i = 1 : ny
    % - fit hyperparameters
    switch choose_kernel
        case 1
            gp = fitrgp(X', dY(i, :)', 'BasisFunction', 'none', ...
                'KernelFunction', 'ardsquaredexponential', ...
                'FitMethod', 'exact', ...
                'PredictMethod', 'fic', ...
                'ActiveSetSize', ni, ...
                'ActiveSetMethod', 'entropy');
        case 2
            gp = fitrgp(X', dY(i, :)', 'BasisFunction', 'none', ...
                'KernelFunction', 'ardmatern32', ...
                'FitMethod', 'exact', ...
                'PredictMethod', 'fic', ...
                'ActiveSetSize', ni, ...
                'ActiveSetMethod', 'entropy');
    end
    % - hyperparameters
    gpmodel.hyp(:, i) = ...
        [gp.KernelInformation.KernelParameters(1:end); gp.Sigma];
    
    % - unwrap inducing inputs
    Xu = gp.ActiveSetVectors;
    gpmodel.induce(:, :, i) = Xu';

    % - some important quantities
    alpha = gp.Alpha;
    L = gp.Impl.LFactor;
    iL = inv(L);
    LAA = gp.Impl.LFactor2;
    iLAA = inv(LAA);

    % - setup covariance function
%     covfcn = gp.Impl.Kernel.makeKernelAsFunctionOfXNXM(gp.Impl.ThetaHat);
    covfcn = @(XN, XM) kernel_function(XN, XM, gp.Impl.ThetaHat, ...
        choose_kernel);
    
    % - define posterior function
    sn = gp.Sigma;
    sf = gp.KernelInformation.KernelParameters(end);
    gpmodel.postfcn{i} = @(x) predict_single_sparse_gaussian_process(x, ...
        Xu, covfcn, alpha, sn, sf, iLAA, iL);
end

% - setup full gaussian process
gpmodel.predfcn = @(x) predict_full_gaussian_process(x, gpmodel);
gpmodel.predmeanfcn = @(x) predict_full_gaussian_process_mean_only(x, ...
    gpmodel);
end



% - subfunctions
function [m, s2] = predict_single_sparse_gaussian_process(x, ...
    Xu, covfcn, alpha, sn, sf, iLAA, iL) %#ok<*INUSD> 
kxXu = covfcn(x', Xu);
m = kxXu*alpha;
% s2 = sn^2 + sf.^2 - sum((iLAA*kxXu').^2, 1)' + sum((iL*kxXu').^2, 1)';
s2 = sf.^2 - sum((iLAA*kxXu').^2, 1)' + sum((iL*kxXu').^2, 1)';
end

function [m, S] = predict_full_gaussian_process(x, gpmodel)
% - get dimension
[~, ny] = size(gpmodel.hyp);

% - preallocate space
if isfloat(x)
    m = zeros(ny, 1);
    S = zeros(ny, ny);
else
    import casadi.*; %#ok<*SIMPT>
    m = SX.zeros(ny, 1);
    S = SX.zeros(ny, ny);
end

% - predict
for i = 1 : ny
    % - push through each gaussian process
    [m(i), S(i, i)] = gpmodel.postfcn{i}(x);
end
end

function m = predict_full_gaussian_process_mean_only(x, gpmodel)
% - dimension
[~, ny] = size(gpmodel.hyp);
[~, nq] = size(x);

% - preallocation
if isfloat(x)
    m = zeros(ny, nq);
else
    import casadi.*; %#ok<*SIMPT>
    m = SX.zeros(ny, nq);
end

% - push through each gaussian process
for iq = 1 : nq
    for iy = 1 : ny
        m(iy, iq) = gpmodel.postfcn{iy}(x(:, iq));
    end
end
end

function KNM = kernel_function(XN, XM, logtheta, choose_kernel)
% - unwrap hyper-parameters
d = length(logtheta) - 1;
sigmaL = exp(logtheta(1:d));
sigmaF = exp(logtheta(d+1));

% - normalized euclidean distances
makepos = true;
KNM = calc_dist(XN(:,1)/sigmaL(1), XM(:,1)/sigmaL(1), makepos);
for r = 2 : d
    KNM = KNM + calc_dist(XN(:,r)/sigmaL(r), XM(:,r)/sigmaL(r), makepos);
end

if choose_kernel == 1                   % ard squared exponential
    KNM = (sigmaF^2)*exp(-0.5*KNM);
elseif choose_kernel == 2               % ard matern 32
    D = sqrt(3)*sqrt(KNM);
    KNM = (sigmaF^2)*((1 + D).*exp(-D));
end

% - subfunction
function D2 = calc_dist(XN, XM, makepos)
D2 = sum(XN.^2, 2) - 2*XN*XM' + sum(XM.^2, 2)';
if makepos
%     D2 = max(0, D2, 'includenan');
    D2 = max(0, D2);
end
end

end

% function K = secov(x1, x2, logtheta)
% % - dimension
% [~, nx] = size(x1);
% 
% % - unwarp
% l2 = exp(2*logtheta(1:nx));
% sf2 = exp(2*logtheta(end));
% 
% % - covariance matrix
% iL = diag(1./l2);
% K = sf2*exp(-(maha_repmat(x1, x2, iL))/2);
% end
% 
% function K = matern32cov(x1, x2, logtheta)
% % - dimension
% [~, nx] = size(x1);
% 
% % - unwarp
% l2 = exp(2*logtheta(1:nx));
% sf2 = exp(2*logtheta(end));
% 
% % - distance
% iL = diag(1./l2);
% d = maha_repmat(x1, x2, iL);
% 
% % - covariance
% sqrt3d = real(sqrt(3*d));
% K = sf2*(1 + sqrt3d).*exp(-sqrt3d);
% end
% 
% function d = maha_bsxfun(a, b, W)
% aW = a*W;
% d = bsxfun(@plus, sum(aW.*a, 2), sum(b*W.*b, 2)') - 2*aW*b';
% end
% 
% function d = maha_repmat(a, b, W)
% n1 = size(a, 1);
% n2= size(b, 1);
% aW = a*W;
% bW = b*W;
% d = repmat(sum(aW.*a, 2), 1, n2) + repmat(sum(bW.*b, 2)', n1, 1) - ...
%     2*aW*b';
% end

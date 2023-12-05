% Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
%
% SPDX-License-Identifier: MIT

%% First global system building
clear all
close all
clc

gamma = 1.5;
uL = 0.01; bdr = [0., uL];
L = 1.0;
Ne = 200;
dx = L / Ne;
k1 = 10.;
offset1 = 0.2;

Nsub = 4;
Nx = Ne / Nsub + 1;

xg = [];
for k=1:Nsub
    xg = [xg; linspace(0., L / Nsub, Nx) + L / Nsub * (k-1)];
end
xg = xg';

bg = sinRHS(xg, dx, k1, offset1);

figure(1)
for k = 1:Nsub
    plot(xg(:,k),bg(:,k),'-');
    hold on
end
hold off

Ku = spdiags([-ones(Nx,1) 2*ones(Nx,1) -ones(Nx,1)], [-1, 0, 1], Nx, Nx);
Ku(1,1) = 1;
Ku(end,end) = 1;
Ku = Ku / dx;

[Kglobal, bglobal] = buildGlobalWithBdry(Ku, bg, bdr, dx, gamma);

uglobal = Kglobal \ bglobal(:);
uglobal = reshape(uglobal,[Nx,Nsub]);
figure(2)
plot(xg, uglobal,'o-','linewidth',1);
set(gca,'fontsize',20,'ticklabelinterpreter','latex');
xlabel('$x$','interpreter','latex');
ylabel('$u(x)$','interpreter','latex');
title('Interior Penalty FOM','interpreter','latex');

%% Sample solutions
Nsample = 100;

% ks = k1 * (1.0 + 0.6 * randn(Nsample,1));
ks = k1 * (0.5 + 1.0 * rand(Nsample,1));
offsets = 2.0 * pi / k1 * rand(Nsample,1);

bs = zeros(Nx,Nsub,Nsample);
us = zeros(Nx,Nsub,Nsample);
for k = 1:Nsample
    bs(:,:,k) = sinRHS(xg, dx, ks(k), offsets(k));
    [Kglobal, bglobal] = buildGlobalWithBdry(Ku, bs(:,:,k), bdr, dx, gamma);
    uglobal = Kglobal \ bglobal(:);
    us(:,:,k) = reshape(uglobal,[Nx,Nsub]);
end

%% LSPG with separate subdomain basis

% POD over each subdomain
Nbasis = 6;

Ublock = {};
for k = 1:Nsub
    [U,S,V] = svd(squeeze(us(:,k,:)));
    Ublock{k} = U(:,1:Nbasis);
end

P = blkdiag(Ublock{:});

% Produce a random FOM
% k2 = k1 * (1.0 + 0.6 * randn());
% offset2 = 2.0 * pi / k1 * rand();
k2 = 20.0;
offset2 = 0.5;

b2 = sinRHS(xg, dx, k2, offset2);
[Kglobal, bglobal] = buildGlobalWithBdry(Ku, b2, bdr, dx, gamma);
fomU = reshape(Kglobal \ bglobal(:), [Nx, Nsub]);

% LSPG projection
KR = Kglobal * P;
w = KR \ bglobal(:);
romU = reshape(P * w, [Nx, Nsub]);

figure(3)
plot(xg(:), fomU(:), 'r-', 'linewidth',3);
hold on
plot(xg(:), romU(:), 'bo-', 'linewidth', 1);
hold off
title(strcat('$k = ',num2str(k2),'$'),'interpreter','latex');
set(gca,'fontsize',20,'ticklabelinterpreter','latex');
h=legend('FOM','ROM');
set(h,'interpreter','latex');

%% LSPG with unified subdomain basis

% POD over each subdomain
Nbasis = 6;

us_unified = reshape(us,[Nx,Nsub*Nsample]);
us_mean = repmat(mean(us_unified,2),1,Nsub*Nsample);
ud = us_unified - us_mean;
figure(5)
% plot(xg(:,1), us_unified, '-');
% plot(xg, us_unified(:,1:4));
plot(xg(:,1), ud, '-');

[U,S,V] = svd(ud);
Ublock = {};
for k = 1:Nsub
    Ublock{k} = U(:,1:Nbasis);
end
P = blkdiag(Ublock{:});

% Produce a random FOM
% k2 = k1 * (1.0 + 0.6 * randn());
% offset2 = 2.0 * pi / k1 * rand();

b2 = sinRHS(xg, dx, k2, offset2);
[Kglobal, bglobal] = buildGlobalWithBdry(Ku, b2, bdr, dx, gamma);
fomU = reshape(Kglobal \ bglobal(:), [Nx, Nsub]);

% LSPG projection
KR = Kglobal * P;
w = KR \ bglobal(:);
romU = reshape(P * w, [Nx, Nsub]);

figure(4)
plot(xg(:), fomU(:), 'r-', 'linewidth',3);
hold on
plot(xg(:), romU(:), 'bo-', 'linewidth', 1);
hold off
title(strcat('$k = ',num2str(k2),'$'),'interpreter','latex');
set(gca,'fontsize',20,'ticklabelinterpreter','latex');
h=legend('FOM','ROM');
set(h,'interpreter','latex');


%% auxiliary functions

function rhs = sinRHS(xgi, dxi, ki, offseti)

costerm = cos(ki * xgi + offseti);
sinterm = sin(ki * xgi + offseti);

tmp1 = - 1. / ki * costerm(2:end,:)         ...
       + 1. / ki / ki / dxi * (sinterm(2:end,:) - sinterm(1:end-1,:));

tmp2 = + 1. / ki * costerm(1:end-1,:)         ...
       - 1. / ki / ki / dxi * (sinterm(2:end,:) - sinterm(1:end-1,:));
   
rhs = zeros(size(xgi));
rhs(1:end-1,:) = rhs(1:end-1,:) + tmp2;
rhs(2:end,:) = rhs(2:end,:) + tmp1;

end

function [KK, BB] = buildGlobalWithBdry(Kunit, bunits, bdri, dxi, gammai)

numBlocks = size(bunits, 2);
Kblocks = {};
for k = 1:numBlocks
    Kblocks{k} = Kunit;
end

xb1 = zeros(size(bunits, 1), 1); xb2 = xb1;

xb1(1) = bdri(1);
bunits(1,1) = Kblocks{1}(1,1) * bdri(1);
Kblocks{1}(1,2:end) = 0;
bunits(2,1) = bunits(2,1) - Kblocks{1}(2,:) * xb1; 
Kblocks{1}(2:end,1) = 0;

xb2(end) = bdri(2);
bunits(end,end) = Kblocks{end}(end,end) * bdri(2);
Kblocks{end}(end,1:end-1) = 0;
bunits(end-1,end) = bunits(end-1,end) - Kblocks{end}(end-1,:) * xb2; 
Kblocks{end}(1:end-1,end) = 0;

BB = bunits;
KK = blkdiag(Kblocks{:});

Ai = -0.5 * [0 -1 1 0;
             -1 2 -2 1;
             1 -2 2 -1;
             0 1 -1 0;];
Ai(2:3,2:3) = Ai(2:3,2:3) + gammai * [1 -1; -1 1];
Ai = Ai / dxi; Ai = sparse(Ai);

Nx = size(bunits,1);
for k = 1:numBlocks-1
    KK(k*Nx-1:k*Nx+2, k*Nx-1:k*Nx+2) = KK(k*Nx-1:k*Nx+2, k*Nx-1:k*Nx+2) + Ai;
end

end
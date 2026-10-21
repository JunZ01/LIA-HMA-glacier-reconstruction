function run_round131_new_proxy_huber_innovation_v1(replay_path, prior_path, seasonal_path, input_path, output_path)
% Apply a fixed 3-sigma Huber clip to new-proxy innovations only.

S = load(replay_path); P = load(prior_path, 'precipitation_prior_336x848_mm_day');
Q = load(seasonal_path); I = load(input_path);
sample_indices = round(S.sample_indices(:));
Xb = P.precipitation_prior_336x848_mm_day(:, sample_indices);
years = round(I.years(:))'; assert(isequal(years, 1000:1920));
nlat = length(Q.lat);
[~, xi] = min(abs(Q.lat(:) - 30.42)); [~, xj] = min(abs(Q.lon(:) - 95.07));
[~, ki] = min(abs(Q.lat(:) - 32.75)); [~, kj] = min(abs(Q.lon(:) - 76.62));
xcell = xi + (xj - 1) * nlat; kcell = ki + (kj - 1) * nlat;
xt = Q.jja_temperature(xcell, :) - mean(Q.jja_temperature(xcell, :));
xp = Q.jja_precipitation(xcell, :) - mean(Q.jja_precipitation(xcell, :));
kt = Q.jja_temperature(kcell, :) - mean(Q.jja_temperature(kcell, :));
kp = Q.ondjfma_precipitation(kcell, :) - mean(Q.ondjfma_precipitation(kcell, :));
products = {'bea','cru'}; hnew = zeros(2,500,2); rnew = zeros(2,2);
for p = 1:2
    tag = products{p}; xc = I.(['xinpu_coeff_' tag]); kc = I.(['keylong_coeff_' tag]);
    xfull = xc(1) + xc(2)*xt + xc(3)*xp; kfull = kc(1) + kc(2)*kt + kc(3)*kp;
    hnew(1,:,p) = xfull(sample_indices); hnew(2,:,p) = kfull(sample_indices);
    rnew(1,p) = I.(['xinpu_rvar_' tag]); rnew(2,p) = I.(['keylong_rvar_' tag]);
end

nrun = length(years); posterior_mean = zeros(336,nrun,2); posterior_sigma = zeros(336,nrun,2);
effective_y = nan(2,nrun,2); raw_innovation_z = nan(2,nrun,2); clipped = false(2,nrun,2);
clip_threshold = 3;
for p = 1:2
    for j = 1:2
        mu = mean(hnew(j,:,p)); total_sd = sqrt(var(hnew(j,:,p),0,2) + rnew(j,p));
        active = isfinite(I.y_new(j,:)); delta = I.y_new(j,active) - mu; z = delta / total_sd;
        raw_innovation_z(j,active,p) = z;
        clipped_delta = max(-clip_threshold*total_sd, min(clip_threshold*total_sd, delta));
        effective_y(j,active,p) = mu + clipped_delta;
        clipped(j,active,p) = abs(z) > clip_threshold;
    end
end

for t = 1:nrun
    ni = find(~isnan(S.y0(:,t))); y0 = S.y0(ni,t); h0 = S.HXb0(ni,:);
    residual_subset = S.R0v_filled(ni,:);
    if exist('nancov','file'); r0 = diag(diag(nancov(residual_subset'))); else; r0 = diag(var(residual_subset,0,2)); end
    for p = 1:2
        rows = find(isfinite(I.y_new(:,t)))';
        if isempty(rows); y = y0; h = h0; r = r0;
        else
            y = [y0; effective_y(rows,t,p)];
            h = [h0; reshape(hnew(rows,:,p),length(rows),500)];
            r = blkdiag(r0,diag(rnew(rows,p)));
        end
        [xa,xa_mean] = M_update(Xb,h,y,r,1);
        posterior_mean(:,t,p) = xa_mean; posterior_sigma(:,t,p) = std(xa,0,2);
    end
    if mod(t,100)==0 || t==nrun; fprintf('Round131 progress: %d/%d\n',t,nrun); end
end
save(output_path,'years','posterior_mean','posterior_sigma','effective_y','raw_innovation_z','clipped','clip_threshold','-v7.3');
disp('ROUND131_NEW_PROXY_HUBER_COMPLETE');
end

function ablate_positive_b2_group_operator_v1(replay_path, prior_path, component_path, group_path, baseline_path, output_path)
% Four frozen-background branches for the positive-b2 proxy group.
S = load(replay_path); P = load(prior_path, 'precipitation_prior_336x848_mm_day', 'lat', 'lon'); C = load(component_path); G = load(group_path); B = load(baseline_path, 'run_years', 'total_increment');
Xall = P.precipitation_prior_336x848_mm_day; samples = round(S.sample_indices(:)); Xb = Xall(:, samples); Xmean = mean(Xb, 2); Xp = Xb - Xmean; nens = size(Xb, 2);
selected = round(C.selected_proxy_indices(:)); group = round(G.positive_b2_proxy_indices(:)); assert(length(selected) == 65 && length(group) == 60, 'group-count-contract-failed');
[in_group, group_local] = ismember(selected, group); assert(sum(in_group) == 60 && all(group_local(in_group) > 0), 'group-membership-contract-failed');
Hbase = S.HXb0; group_mean = mean(Hbase(selected(in_group), :), 2);
Hpzero = Hbase; Hfullzero = Hbase; Htzero = Hbase;
Hpzero(selected(in_group), :) = group_mean + C.temperature_anomaly(in_group, :);
Hfullzero(selected(in_group), :) = repmat(group_mean, 1, nens);
Htzero(selected(in_group), :) = group_mean + C.precipitation_anomaly(in_group, :);
mean_invariance = [max(abs(mean(Hbase,2)-mean(Hpzero,2))), max(abs(mean(Hbase,2)-mean(Hfullzero,2))), max(abs(mean(Hbase,2)-mean(Htzero,2)))]; assert(max(mean_invariance) <= 1e-12, 'mean-invariance-failed');
latw = repmat(cosd(P.lat(:)), 1, length(P.lon)); weights = latw(:) ./ sum(latw(:)); prior_hma_mean = weights' * Xmean;
years = 1901:1920; assert(isequal(round(B.run_years(:))', years), 'year-contract-failed'); names = {'baseline','positive_b2_P_zero','positive_b2_full_zero','positive_b2_T_zero'}; branches = {Hbase, Hpzero, Hfullzero, Htzero}; total = zeros(length(years), length(names)); closure = zeros(length(years), length(names)); innovation = nan(size(S.y0,1), length(years), length(names));
for bb = 1:length(names)
    H = branches{bb};
    for yy = 1:length(years)
        t = years(yy) - 999; ni = find(~isnan(S.y0(:,t))); y = S.y0(ni,t); HX = H(ni,:); HXmean = mean(HX,2); HXp = HX-HXmean; R = diag(diag(nancov(S.R0v_filled(ni,:)'))); K = ((Xp*HXp')./(nens-1)) / (((HXp*HXp')./(nens-1))+R); inc = y-HXmean; total(yy,bb) = weights'*(Xmean+K*inc)-prior_hma_mean; closure(yy,bb) = weights'*(Xmean+K*inc)-prior_hma_mean-sum((weights'*K)'.*inc); innovation(ni,yy,bb) = inc;
    end
end
baseline_closure = max(abs(total(:,1)-B.total_increment(:))); innovation_invariance = max(abs(innovation-innovation(:,:,1)), [], 'all', 'omitnan'); assert(max(abs(closure),[],'all') <= 1e-12 && baseline_closure <= 1e-12 && innovation_invariance <= 1e-12, 'closure-contract-failed');
save(output_path, 'years','names','group','prior_hma_mean','total','closure','mean_invariance','baseline_closure','innovation_invariance','-v7.3');
fprintf('BASELINE_CLOSURE=%.17g\n', baseline_closure); fprintf('INNOVATION_INVARIANCE=%.17g\n', innovation_invariance); disp('POSITIVE_B2_GROUP_ABLATION_COMPLETE');
end

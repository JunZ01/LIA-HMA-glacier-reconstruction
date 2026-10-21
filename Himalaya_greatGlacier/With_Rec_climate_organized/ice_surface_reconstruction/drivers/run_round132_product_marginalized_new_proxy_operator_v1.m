function run_round132_product_marginalized_new_proxy_operator_v1(replay_path, prior_path, seasonal_path, input_path, output_path)
% Assimilate a product-marginalized Xinpu+Keylong operator.
S=load(replay_path); P=load(prior_path,'precipitation_prior_336x848_mm_day'); Q=load(seasonal_path); I=load(input_path);
idx=round(S.sample_indices(:)); Xb=P.precipitation_prior_336x848_mm_day(:,idx); years=round(I.years(:))'; assert(isequal(years,1000:1920));
nlat=length(Q.lat); [~,xi]=min(abs(Q.lat(:)-30.42)); [~,xj]=min(abs(Q.lon(:)-95.07)); [~,ki]=min(abs(Q.lat(:)-32.75)); [~,kj]=min(abs(Q.lon(:)-76.62));
xc=xi+(xj-1)*nlat; kc=ki+(kj-1)*nlat;
xt=Q.jja_temperature(xc,:)-mean(Q.jja_temperature(xc,:)); xp=Q.jja_precipitation(xc,:)-mean(Q.jja_precipitation(xc,:));
kt=Q.jja_temperature(kc,:)-mean(Q.jja_temperature(kc,:)); kp=Q.ondjfma_precipitation(kc,:)-mean(Q.ondjfma_precipitation(kc,:));
xb=I.xinpu_coeff_bea; xr=I.xinpu_coeff_cru; kb=I.keylong_coeff_bea; kr=I.keylong_coeff_cru;
hx_b=xb(1)+xb(2)*xt+xb(3)*xp; hx_r=xr(1)+xr(2)*xt+xr(3)*xp;
hk_b=kb(1)+kb(2)*kt+kb(3)*kp; hk_r=kr(1)+kr(2)*kt+kr(3)*kp;
hproducts=zeros(2,848,2); hproducts(1,:,1)=hx_b; hproducts(1,:,2)=hx_r; hproducts(2,:,1)=hk_b; hproducts(2,:,2)=hk_r;
hmean=mean(hproducts,3); hnew=hmean(:,idx);
rmean=[mean([I.xinpu_rvar_bea,I.xinpu_rvar_cru]);mean([I.keylong_rvar_bea,I.keylong_rvar_cru])];
operator_product_variance=mean((hproducts-hmean).^2,2); operator_product_variance=squeeze(mean(operator_product_variance,3));
rnew=rmean+operator_product_variance;
nrun=length(years); posterior_mean=zeros(336,nrun); posterior_sigma=zeros(336,nrun); innovation_z=nan(2,nrun);
for j=1:2; active=isfinite(I.y_new(j,:)); innovation_z(j,active)=(I.y_new(j,active)-mean(hnew(j,:)))/sqrt(var(hnew(j,:),0,2)+rnew(j)); end
for t=1:nrun
 ni=find(~isnan(S.y0(:,t))); y0=S.y0(ni,t); h0=S.HXb0(ni,:); residual=S.R0v_filled(ni,:);
 if exist('nancov','file'); r0=diag(diag(nancov(residual'))); else; r0=diag(var(residual,0,2)); end
 rows=find(isfinite(I.y_new(:,t)))';
 if isempty(rows); y=y0; h=h0; r=r0; else; y=[y0;I.y_new(rows,t)]; h=[h0;reshape(hnew(rows,:),length(rows),500)]; r=blkdiag(r0,diag(rnew(rows))); end
 [xa,xm]=M_update(Xb,h,y,r,1); posterior_mean(:,t)=xm; posterior_sigma(:,t)=std(xa,0,2);
 if mod(t,100)==0||t==nrun; fprintf('Round132 progress: %d/%d\n',t,nrun); end
end
save(output_path,'years','posterior_mean','posterior_sigma','innovation_z','hnew','rnew','rmean','operator_product_variance','-v7.3'); disp('ROUND132_PRODUCT_MARGINALIZED_COMPLETE');
end

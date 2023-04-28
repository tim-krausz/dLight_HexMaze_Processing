function [dF_F,ref_fitted,slope] = isosbestic_correction(signal,ref)


% iter=1;
% 
% [x,fval,exitflag,output,lambda,grad,hessian] = fmincon(@(x)fit_isosbestic(x,signal,ref),rand(1,2),[],[],[],[],[-10 -10],[10 10]);

mdl = fitlm(ref,signal);
x = [mdl.Coefficients.Estimate];

ref_fitted = x(2)*ref + x(1);
dF_F = 100*(signal - ref_fitted)./ ref_fitted;

slope= x;

end
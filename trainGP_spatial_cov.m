function hyp = trainGP_spatial_cov(XTrain,YTrain,STrain,numinstances_train,numfeatures)
XTrain = [XTrain{:}]; 
XTrain = cell2mat(XTrain); 
XTrain = double(reshape(XTrain,numinstances_train,numfeatures));

STrain = [STrain{:}]; 
STrain = cell2mat(STrain); 
STrain = double(reshape(STrain,numinstances_train,2));
XTrain = [STrain XTrain];

YTrain = [YTrain{:}]; 
YTrain = cell2mat(YTrain); 
YTrain = double(reshape(YTrain,numinstances_train,1));

% SOLO -- Uncomment next three lines
%covfunc = {@covMask,{3:(numfeatures+2),@covSEiso}};
%ell = 1/4; sf = 1;
%hyp.cov = log([ell; sf]);

% SUM -- Uncomment next three lines
%covfunc = {@covSum,{{@covMask,{3:(numfeatures+2),@covSEiso}},{@covMask,{1:2,@covSEiso}}}};
%ell = 1/4; sf = 1; ell_s = 1/4; sf_s = 1; 
%hyp.cov = log([ell_s; sf_s; ell; sf]);

% PROD -- Uncomment next three lines
covfunc = {@covProd,{{@covMask,{3:(numfeatures+2),@covSEisoU}},{@covMask,{1:2,@covSEiso}}}};
ell_s = 1/4; ell = 1/4; sf = 1; 
hyp.cov = log([ell_s; ell; sf]);

likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
hyp = minimize(hyp, @gp, -100, @infExact, [], covfunc, likfunc, XTrain, YTrain);
hyp = [exp(hyp.cov);exp(hyp.lik)];

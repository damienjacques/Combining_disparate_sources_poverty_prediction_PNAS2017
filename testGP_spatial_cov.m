function YStar = testGP_spatial_cov(XTrain,YTrain,STrain,XTest,STest,numinstances_train,numinstances_test,numfeatures,means,covhyp)
XTrain = [XTrain{:}]; 
XTrain = cell2mat(XTrain); 
XTrain = double(reshape(XTrain,numinstances_train,numfeatures));

STrain = [STrain{:}]; 
STrain = cell2mat(STrain); 
STrain = double(reshape(STrain,numinstances_train,2));
XTrain = [STrain XTrain];

XTest = [XTest{:}]; 
XTest = cell2mat(XTest); 
XTest = double(reshape(XTest,numinstances_test,numfeatures));

STest = [STest{:}]; 
STest = cell2mat(STest); 
STest = double(reshape(STest,numinstances_test,2));
XTest = [STest XTest];

YTrain = [YTrain{:}]; 
YTrain = cell2mat(YTrain); 
YTrain = double(reshape(YTrain,numinstances_train,1));

covhyp = [covhyp{:}];
covhyp = cell2mat(covhyp);

means = [means{:}];
means = [0 0 means];%add two zeros to ignore the impact of spatial features
meanfunc = {@meanSum, {@meanLinear, @meanConst}};
hyp.mean = transpose(means);
 
%covfunc = {@covMask,{3:(numfeatures+2),@covSEiso}};
%covfunc = {@covSum,{{@covMask,{3:(numfeatures+2),@covSEiso}},{@covMask,{1:2,@covSEiso}}}};
covfunc = {@covProd,{{@covMask,{3:(numfeatures+2),@covSEisoU}},{@covMask,{1:2,@covSEiso}}}};
likfunc = @likGauss; 
hyp.cov = log(covhyp(1:end-1));
hyp.lik = log(covhyp(end));

[YStar, var] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, XTrain, YTrain, XTest);
YStar = [YStar;var];
%YStar = gp(hyp, @infExact, meanfunc, covfunc, likfunc, XTrain, YTrain, XTest);

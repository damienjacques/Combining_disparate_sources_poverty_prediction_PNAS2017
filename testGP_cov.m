function YStar = testGP_cov(XTrain,YTrain,XTest,numinstances_train,numinstances_test,numfeatures,means,covhyp)
XTrain = [XTrain{:}]; 
XTrain = cell2mat(XTrain); 
XTrain = double(reshape(XTrain,numinstances_train,numfeatures));

XTest = [XTest{:}]; 
XTest = cell2mat(XTest); 
XTest = double(reshape(XTest,numinstances_test,numfeatures));

YTrain = [YTrain{:}]; 
YTrain = cell2mat(YTrain); 
YTrain = double(reshape(YTrain,numinstances_train,1));

covhyp = [covhyp{:}];
covhyp = cell2mat(covhyp);

means = [means{:}];

meanfunc = {@meanSum, {@meanLinear, @meanConst}};
hyp.mean = transpose(means);
    
covfunc = @covSEiso;
likfunc = @likGauss; 
hyp.cov = log(covhyp(1:end-1));
hyp.lik = log(covhyp(end));
[YStar, var] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, XTrain, YTrain, XTest);
YStar = [YStar;var];
%YStar = gp(hyp, @infExact, meanfunc, covfunc, likfunc, XTrain, YTrain, XTest);

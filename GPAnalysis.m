XTrain = csvread('../../output/csvfiles/XTrainDataGP.csv');
YTrain = csvread('../../output/csvfiles/YTrainDataGP.csv');
XTest = csvread('../../output/csvfiles/XTestDataGP.csv');
YTest = csvread('../../output/csvfiles/YTestDataGP.csv');

X = [XTrain;XTest];
Y = [YTrain;YTest];

%XTrain = X(1:400,1:42);
%XTest = X(401:end,1:42);
%YTrain = Y(1:400);
%YTest = Y(401:end);
XTrain = XTrain(:,1:42);
XTest = XTest(:,1:42);
% build GP


meanfunc = {@meanSum, {@meanLinear, @meanConst}}; hyp.mean = [zeros(42,1); 0];
covfunc = @covSEiso;
ell = 1/4; sf = 1; hyp.cov = log([ell; sf]);
likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);

hyp = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, likfunc, XTrain, YTrain);
[YStar, vYStar] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, XTrain, YTrain, XTest);
errorbar(YStar,sqrt(vYStar));
hold on;plot(YTest,'g');hold off;
corrcoef(YTest,YStar)

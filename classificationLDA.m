function [rate,stdcv,cfMat] = classificationLDA(charac,class,NumFold,repeat)
s = RandStream('mt19937ar','seed',0);
RandStream.setGlobalStream(s);
cfMat = {};

mdl = ClassificationDiscriminant.fit(charac,class);

if NumFold
    
    for i = 1:repeat
        rng(i*100);
        cvpart = cvpartition(class,'kfold',NumFold);
        
        cvmdl = crossval(mdl,'cvpartition',cvpart);
        
        kloss = kfoldLoss(cvmdl,'mode','individual');
        rates(i) = 1-mean(kloss);
        stdcvs(i) = std((1-kloss)*100);
        [label,score] = kfoldPredict(cvmdl);
        [cfMat{i},grpOrder] = confusionmat(class,label);
    end
    [rate] = mean(rates);
    stdcv = std(rates*100);
    cfMat = {};
    rate = rate*100;
elseif NumFold == 0
    cvmdl = crossval(mdl,'leaveout','on');
    
    kloss = kfoldLoss(cvmdl,'mode','individual');
    
    rate = (1-mean(kloss))*100;
    
    stdcv = (std(kloss));
    
    [label,score] = kfoldPredict(cvmdl);
    [cfMat,grpOrder] = confusionmat(class,label);
end

clc,clear all,close all
dosya=dir('*.*g');
net{1}=darknet53; net{2}=densenet201();
for od=1:2
    layer{od}=net{od}.Layers(end-2).Name;
    layer{2+od}=net{od}.Layers(end-3).Name;
    inputSizeA = net{od}.Layers(1).InputSize;

for k=1:length(dosya)
    res=imread(fullfile(dosya(k).folder, dosya(k).name));
    [ww,hh,ll]=size(res);
    if ll==1
        res(:,:,1)=res(:,:,1); res(:,:,2)=res(:,:,1); res(:,:,3)=res(:,:,1);
    end
    res=imresize(res,[256,256]);
    augimdsTrainA = augmentedImageDatastore(inputSizeA(1:2),res);
    fm = activations(net{od},augimdsTrainA,layer{od},'OutputAs','rows');
    fm1 = activations(net{od},augimdsTrainA,layer{2+od},'OutputAs','rows');
    [ww,hh,lay]=size(res);
    
    [~, fmCols] = size(fm);    
    [~, fm1Cols] = size(fm1);  
    totalCols = fmCols + fm1Cols; 
    X(k,1:totalCols)=[fm fm1];
    y(k)=str2num(dosya(k).name(1:2));
    clear res
end
y = y(:);
X = normalize(X,'zscore');
[idx,weights] = relieff(X,y,10);
posIndeks = find(weights > 0);
[~, sirali] = sort(weights(posIndeks), 'descend');
selectedIndex = posIndeks(sirali);
X_posselected = X(:, selectedIndex);
son{od} = X_posselected; 
clear  X mdl xx idx X_posselected
end
combinedFeatures(:,:)=[son{1} son{2}]; 
N = 50;                
max_Iter = 500;         
CR = 0.8;               
MR = 0.3; 
cv = cvpartition(y,'KFold',10);
Split.training = training(cv,1);
Split.test = test(cv,1);
[sFeatNew, SfNew, NfNew, curveNew] = GA_FeatureSelection(combinedFeatures, y, N, max_Iter, CR, MR, Split);
son{3} = sFeatNew;  
%%%
template = templateSVM(...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 3, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true);
mdl = fitcecoc(...
    son{3}, ...
    y, ...
    'Learners', template, ...
    'Coding', 'onevsone', ...
    'ClassNames', single(1:2));
kk=crossval(mdl,'KFold',10);
loss_end = kfoldLoss(kk);
disp(['Loss değeri: ', num2str(loss_end)]);

function [Xsel, featIdx, featCount, bestCurve] = ...
    GA_FeatureSelection(X, y, popSize, maxGen, CR, MR, split)
    fitnessFcn = @GA_Fitness;
    numFeat = size(X,2);
    population = rand(popSize, numFeat) > 0.5;
    fitnessVal = inf(1, popSize);
    bestFitness = inf;

    for i = 1:popSize
        fitnessVal(i) = fitnessFcn(X, y, population(i,:), split);
        if fitnessVal(i) < bestFitness
            bestFitness = fitnessVal(i);
            bestSol = population(i,:);
        end
    end

    bestCurve = zeros(1, maxGen);
    targetAcc = 0.95;

 
    for gen = 1:maxGen

       
        selProb = (1 - fitnessVal) / sum(1 - fitnessVal);

        children = [];

     
        while size(children,1) < popSize
            if rand < CR
                parent1 = population(rouletteWheel(selProb), :);
                parent2 = population(rouletteWheel(selProb), :);

                cp = randi([1, numFeat-1]);
                childA = [parent1(1:cp), parent2(cp+1:end)];
                childB = [parent2(1:cp), parent1(cp+1:end)];

                children = [children; childA; childB];
            end
        end

        children = children(1:popSize, :);

    
        mutMask = rand(size(children)) < MR;
        children(mutMask) = ~children(mutMask);

     
        childFitness = inf(1, popSize);
        for i = 1:popSize
            childFitness(i) = fitnessFcn(X, y, children(i,:), split);
            if childFitness(i) < bestFitness
                bestFitness = childFitness(i);
                bestSol = children(i,:);
            end
        end

       
        allPop = [population; children];
        allFit = [fitnessVal, childFitness];

        [allFit, order] = sort(allFit, 'ascend');
        population = allPop(order(1:popSize), :);
        fitnessVal = allFit(1:popSize);

        bestCurve(gen) = bestFitness;

       
        if (1 - bestFitness) >= targetAcc
            break;
        end
    end

    
    featIdx = find(bestSol == 1);
    featCount = numel(featIdx);
    Xsel = X(:, featIdx);

end


function idx = rouletteWheel(prob)
    cumProb = cumsum(prob);
    r = rand;
    idx = find(cumProb >= r, 1);
end

function fitness = GA_Fitness(X, labels, featureMask, split)

    if nnz(featureMask) == 0
        featureMask(randi(numel(featureMask))) = 1;
    end

    Xsel = X(:, featureMask == 1);
    labels = labels(:);

    Xtrain = Xsel(split.training, :);
    Ytrain = labels(split.training);
    Xtest  = Xsel(split.test, :);
    Ytest  = labels(split.test);

    knn = fitcknn(Xtrain, Ytrain, 'NumNeighbors', 5);
    Ypred = predict(knn, Xtest);
    accuracy = mean(Ypred == Ytest);
    fitness = 1 - accuracy;

end
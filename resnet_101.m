imds3=imageDatastore('MRI_image_dataset_path', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
[imdsTrain3, imdsValidation3, imdsTest3]=splitEachLabel(imds3,0.6,0.1,0.3,'randomized');

net3=resnet101;  % replce resnet101 by resnet50 and resnet18 % 

lgraph2=layerGraph(net3); 

net3.Layers(1);
 
inputSize=net3.Layers(1).InputSize;

lgraph2 = removeLayers(lgraph2, {'fc1000','prob','ClassificationLayer_predictions'});

 numClasses = numel(categories(imds3.Labels));
 newLayers3 = [
     fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',20,'BiasLearnRateFactor',10)
     softmaxLayer('Name','softmax')
     classificationLayer('Name','classoutput')];
 lgraph2 = addLayers(lgraph2,newLayers3);
 lgraph2 = connectLayers(lgraph2,'pool5','fc');


 layers2 = lgraph2.Layers;
 connections = lgraph2.Connections;
 layers2(1:10) = freezeWeights(layers2(1:10));

 pixelRange = [-30 30];
 imageAugmenter3 = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
     augimdsTrain3 = augmentedImageDatastore(inputSize(1:2),imdsTrain3, ...
    'DataAugmentation',imageAugmenter3);
 
 augimdsValidation3 = augmentedImageDatastore (inputSize (1: 3), imdsValidation3);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',12, ...
    'MaxEpochs',200, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData' , augimdsValidation3, ... 
    'Verbose',true ,...
    'ValidationPatience',10,...
    'Plots','training-progress');

net3 = trainNetwork(augimdsTrain3,lgraph2,options);
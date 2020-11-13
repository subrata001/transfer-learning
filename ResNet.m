%% Load the folder containing images using imadeDatastore 
imds3=imageDatastore('G:\Alzheimer\Alzheimer_s Dataset\test', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%% Split the dataset into Training, Validation and Testing 
[imdsTrain, imdsValidation, imdsTest]=splitEachLabel(imds3,0.6,0.1,0.3,'randomized');

%% Load the Pre-Trained Model
net3=resnet101;  % replace resnet101 by resnet50 and resnet18, follow the rest of the code for all% 

lgraph2=layerGraph(net3); 

net3.Layers(1);

%% Input the training images to first layer of the pre-trained model
inputSize=net3.Layers(1).InputSize;

%% Replace the fully-connected layers from 1000 object categories into no.of classes of your dataset
lgraph2 = removeLayers(lgraph2, {'fc1000','prob','ClassificationLayer_predictions'});

 % Adding 3 new layers %
 numClasses = numel(categories(imds3.Labels));
 newLayers3 = [
     fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',20,'BiasLearnRateFactor',10)
     softmaxLayer('Name','softmax')
     classificationLayer('Name','classoutput')];
 % Connecting the new layers to the network %
 lgraph2 = addLayers(lgraph2,newLayers3);
 lgraph2 = connectLayers(lgraph2,'pool5','fc');
 layers2 = lgraph2.Layers;
 connections = lgraph2.Connections;
 layers2(1:10) = freezeWeights(layers2(1:10));
 
%% Preparing training and validation images to train
 pixelRange = [-30 30];
 imageAugmenter3 = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
     augimdsTrain3 = augmentedImageDatastore(inputSize(1:2),imdsTrain3, ...
    'DataAugmentation',imageAugmenter3);
 
 augimdsValidation3 = augmentedImageDatastore (inputSize (1: 3), imdsValidation3);

%% Setting training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',12, ...
    'MaxEpochs',200, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData' , augimdsValidation3, ... 
    'Verbose',true ,...
    'ValidationPatience',10,...
    'Plots','training-progress');

%% Training the network
net3 = trainNetwork(augimdsTrain3,lgraph2,options);
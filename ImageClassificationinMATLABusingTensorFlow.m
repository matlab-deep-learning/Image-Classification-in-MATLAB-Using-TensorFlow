%% MATLAB calling TensorFlow model for Image Classification
% This example highlights how MATLAB can directly call TensorFlow.
% It requires MATLAB, Python & TensorFlow to be installed on the same
% machine.

%% Setup Python 
% This command will launch Python using 'OutOfProcess' to avoid any 
% possible library conflicts

% checkPythonSetup 

%% Read in image

imgOriginal = imread("./Images/banana.png");
imshow(imgOriginal)

% Each pretrained model in tensorflow.keras.applications takes input images
% of different sizes. Therefore the image being classified needs to be
% resized. 
imageHWSize = 480;
img = imresize(imgOriginal, [imageHWSize, imageHWSize]);

% TensorFlow orients image data in a different format to MATLAB. This 
% requires conversion (HWCN TO NHWC)
imgforTF = permute(img, [4 1 2 3]); 

batch_size = int32(1); % Tensorflow require inputs to be converted to int32.
%% Classify Image via TensorFlow Coexecution

% importing EfficientNetV2L pretrained model from keras.applications
model = py.tensorflow.keras.applications.efficientnet_v2.EfficientNetV2L(); 

% converting input from MATLAB array into Python array.
X = py.numpy.asarray(imgforTF);

% call preprocessing function that is required for the image input in Keras
X = py.tensorflow.keras.applications.efficientnet_v2.preprocess_input(X); 

% classify image 
Y = model.predict(X, batch_size); 

% label of classification output
label = py.tensorflow.keras.applications.efficientnet_v2.decode_predictions(Y); 

label = label{1}{1}{2};  % The label is stored in a nested cell. In the file layer of the cell there is a tuple (id, class, probability) - The predicted class label is the 2nd element of the tuple
labelStr = string(label); % Convert the Python str to a MATLAB string

%Label this image
figure
imshow(imgOriginal);
title(labelStr,Interpreter="none");

%% Minimizing calls to TensorFlow 
% Calling tensorflow multiple times can introduce time overhead
% Moving the library loading and preprocessing into a TensorFlow script
% will reduce this overhead

model = py.tfInference.load_model();
Y = py.tfInference.preprocess_predict(model,imgforTF, batch_size);
label = py.tensorflow.keras.applications.efficientnet_v2.decode_predictions(Y);

label = label{1}{1}{2};  % The label is stored in a nested cell. In the file layer of the cell there is a tuple (id, class, probability) - The predicted class label is the 2nd element of the tuple
labelStr = string(label); % Convert the Python str to a MATLAB string

%Label this image
figure
imshow(imgOriginal);
title(labelStr,Interpreter="none");

%% Batch processing of images
% This example highlights how to batch process the classification of images
% from TensorFlow.

% extracting MathWorks Merch data set as seen in:
% https://www.mathworks.com/help/deeplearning/ug/data-sets-for-deep-learning.html
filename = 'MerchData.zip';
dataFolder = 'MerchData';
if ~exist(dataFolder,'dir')
    unzip(filename,dataFolder);
end

batchSize = 16; % One among: [1, 2, 4, 8, 16, 32];
imds = imageDatastore('./MerchData/', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames',...
    'ReadSize', batchSize);
batchSize = int32(batchSize); % TensorFlow expect int32 input

% We want imds to return batches of data in the form NHWC.
% This transform takes a cell array of 3D images in the format HWC
% and returns a 4D batch in the format NHWC
imds = transform(imds,...
    @(x) permute(imresize(cat(4, x{:}), [imageHWSize, imageHWSize]), [4 1 2 3]));

i = 1;
while hasdata(imds)
    X = read(imds);
    Ybatch{i} = py.tfInference.preprocess_predict(model,X,batchSize);   
    i = i+1;
end

% Label the first image from the last batch
% py.tensorflow.keras.applications.mobilenet_v2.decode_predictions takes Y
% (a vector of probabilities) and returns a sorted list of tuples. Each
% tuple has the form (id, class, probability)

topLabels = py.tensorflow.keras.applications.efficientnet_v2.decode_predictions(Ybatch{end}); 
label = topLabels{1}{1}{2}; % label first image. topLabels is a 2D list. E.g. [[ ... ]], so 'label' is still a list 
labelStr = string(label); % Convert the Python str to a MATLAB string

aImg = permute(X(1,:,:,:), [2 3 4 1]); % extracting 1st image and converting from (NHWC TO HWCN)

figure;
imshow(aImg);
title(labelStr,Interpreter="none");

clc;
clear all;
close all;
load mnist_uint8;

train_x = double(reshape(train_x',28,28,60000))/255;
test_x = double(reshape(test_x',28,28,10000))/255;
train_y = double(train_y');
test_y = double(test_y');

%% train a 6c-2s-12c-2s cnn
rand('state',0);
cnn.layers = {
    struct('type','i') %input layer
    struct('type','c','outputmaps',6,'kernelsize',5);% conv layer
    struct('type','s','scale',2);% down sampling
    strcut('type','c','outputmaps',12,'kernelsize',5);% conv layer
    struct('type','s','scale',2);% down sampling
};

opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs = 1;

cnn = cnntrain(cnn, train_x, train_y, opts);

[er, bad] = cnntest(cnn, test_x, test_y);

% plot mean squared error
figure(1);
plot(cnn.rl);
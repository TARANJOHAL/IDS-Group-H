

% opening the training test data
fileID = fopen('E:\thesis19\gurbani\IDS\IDS_train.csv');
fgetl(fileID); 


C=textscan(fileID,'%s %f %f %f %f','delimiter',',');
fclose(fileID);


Open = cell2mat(C(1,2));
Open = Open.';


High = cell2mat(C(1,3));
High = High.';


Low = cell2mat(C(1,4));
Low = Low.';


Close = cell2mat(C(1,5));
Close = Close.';


SMA_10 = tsmovavg(Open,'s',10);
SMA_50 = tsmovavg(Open,'s',50);


EMA_10 = tsmovavg(Open,'e',10);
EMA_50 = tsmovavg(Open,'e',50);

% Input vector of the input variables
Input = {Open; High; Low; SMA_10; EMA_10; SMA_50; EMA_50};
Input = cell2mat(Input);
input=mean(Input);


% Construction of feed-forward neural network
net = newff([minmax(Open); minmax(High); minmax(Low); minmax(SMA_10); minmax(EMA_10); minmax(SMA_50); minmax(EMA_50)], [abs(floor(7)), 1], {'purelin', 'purelin', 'transIm'},'traingdx');

% Maximum number of iterations
net.trainparam.epochs = 8000;

% Desired Tolerance value
net.trainparam.goal = 1e-5;

% learning rate initialisation
net.trainparam.lr = 0.001;

% using full data to train the neural network
net.divideFcn ='dividetrain';
net = train(net, Input, Close);
t = net(Input);

% eveluating the performance of the neural network - using mse as 
% the measuring standard

perf = perform(net, Close, t);
view(net);


load matlab
%% Testing the constructed neural network

% Opening sample test data
fileID = fopen('E:\thesis19\gurbani\IDS\IDS_test_final.csv');
fgetl(fileID);
C_t = textscan(fileID,'%s %f %f %f %f','delimiter',',');
fclose(fileID);
figure()
plot(Accuracy_ANN)
xlabel('classes')
ylabel('accuracy')
figure()
plot(Fmeasure_ANN)
xlabel('classes')
ylabel('Fmeasure')
figure()
plot(Precision_ANN)
xlabel('classes')
ylabel('precision')
plot(Recall_ANN)
xlabel('classes')
ylabel('recall')



clc;
clear;
% opening the training test data
fileID = fopen('IDS_train.csv');
fgetl(fileID); 

% delimitting the various sub-fields
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
CostFunction=@(x) Sphere(x);        % Cost Function

nVar=10;            % Number of Decision Variables

VarSize=[1 nVar];   % Size of Decision Variables Matrix

VarMin=-10;         % Lower Bound of Variables
VarMax= 10;         % Upper Bound of Variables


%% PSO Parameters

MaxIt=1000;      % Maximum Number of Iterations

nPop=100;        % Population Size (Swarm Size)

% PSO Parameters
w=1;            % Inertia Weight
wdamp=0.99;     % Inertia Weight Damping Ratio
c1=1.5;         % Personal Learning Coefficient
c2=2.0;         % Global Learning Coefficient

% If you would like to use Constriction Coefficients for PSO,
% uncomment the following block and comment the above set of parameters.

% % Constriction Coefficients
% phi1=2.05;
% phi2=2.05;
% phi=phi1+phi2;
% chi=2/(phi-2+sqrt(phi^2-4*phi));
% w=chi;          % Inertia Weight
% wdamp=1;        % Inertia Weight Damping Ratio
% c1=chi*phi1;    % Personal Learning Coefficient
% c2=chi*phi2;    % Global Learning Coefficient

% Velocity Limits
VelMax=0.1*(VarMax-VarMin);
VelMin=-VelMax;

%% Initialization

empty_particle.Position=[];
empty_particle.Cost=[];
empty_particle.Velocity=[];
empty_particle.Best.Position=[];
empty_particle.Best.Cost=[];

particle=repmat(empty_particle,nPop,1);

GlobalBest.Cost=inf;

for i=1:nPop
    
    % Initialize Position
    particle(i).Position=unifrnd(VarMin,VarMax,VarSize);
    
    % Initialize Velocity
    particle(i).Velocity=zeros(VarSize);
    
    % Evaluation
    particle(i).Cost=CostFunction(particle(i).Position);
    
    % Update Personal Best
    particle(i).Best.Position=particle(i).Position;
    particle(i).Best.Cost=particle(i).Cost;
    
    % Update Global Best
    if particle(i).Best.Cost<GlobalBest.Cost
        
        GlobalBest=particle(i).Best;
        
    end
    
end

BestCost=zeros(MaxIt,1);

%% PSO Main Loop

for it=1:MaxIt
    
    for i=1:nPop
        
        % Update Velocity
        particle(i).Velocity = w*particle(i).Velocity ...
            +c1*rand(VarSize).*(particle(i).Best.Position-particle(i).Position) ...
            +c2*rand(VarSize).*(GlobalBest.Position-particle(i).Position);
        
        % Apply Velocity Limits
        particle(i).Velocity = max(particle(i).Velocity,VelMin);
        particle(i).Velocity = min(particle(i).Velocity,VelMax);
        
        % Update Position
        particle(i).Position = particle(i).Position + particle(i).Velocity;
        
        % Velocity Mirror Effect
        IsOutside=(particle(i).Position<VarMin | particle(i).Position>VarMax);
        particle(i).Velocity(IsOutside)=-particle(i).Velocity(IsOutside);
        
        % Apply Position Limits
        particle(i).Position = max(particle(i).Position,VarMin);
        particle(i).Position = min(particle(i).Position,VarMax);
        
        % Evaluation
        particle(i).Cost = CostFunction(particle(i).Position);
        
        % Update Personal Best
        if particle(i).Cost<particle(i).Best.Cost
            
            particle(i).Best.Position=particle(i).Position;
            particle(i).Best.Cost=particle(i).Cost;
            
            % Update Global Best
            if particle(i).Best.Cost<GlobalBest.Cost
                
                GlobalBest=particle(i).Best;
                
            end
            
        end
        
    end
    
    BestCost(it)=GlobalBest.Cost;
    
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
    w=w*wdamp;
    
end

BestSol = GlobalBest;
    

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
% Plot generation of the market values
x = 1:size(Close,2);
plot(x, Close, x, Open, x, High, x, Low);

%% Testing the constructed neural network

% Opening sample test data
fileID = fopen('IDS_test_final.csv');

fgetl(fileID);
C_t = textscan(fileID,'%s %f %f %f %f','delimiter',',');
fclose(fileID);
figure()
plot(AccuracyANNPSO)
xlabel('classes')
ylabel('accuracy')
figure()
plot(FmeasureANNPSO)
xlabel('classes')
ylabel('Fmeasure')
figure()
plot(PrecisionANNPSO)
xlabel('classes')
ylabel('precision')
plot(RecallANNPSO)
xlabel('classes')
ylabel('recall')



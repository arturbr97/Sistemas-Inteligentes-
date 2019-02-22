%% Inicialização
clear ; close all; clc

%% Definição da arquitetura da rede
input_layer_size  = 34;  
hidden_layer_size = 12;   
num_labels = 6;  % as labels foram definidas de 1 a 10, sendo o 10 atribuídas a classe 0 
INIT_EPSILON = 0.2;

% Carregando os dados de treinamento
data = load('dermatology.data');
X = data(:,1:34);
y = data(:,35);
m = size(X, 1);

%Cálcular theta0 e theta1 randomicamente
epsilon_init = 0.12;
  
for i=1:5

initial_Theta1 = rand(12,35)*(2*INIT_EPSILON) - INIT_EPSILON;
initial_Theta2 = rand(6,13)*(2*INIT_EPSILON) - INIT_EPSILON;


% transformar as matrizes de pesos em um vetor 
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%%%%%%%%%%%% Treinando a rede neural  %%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%%  Variar o número máximo de iterações para verificar como o treinamento é 
%%  influenciado
options = optimset('MaxIter', 40);

% Tentar diferentes valores de lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Redimensionar Theta1 e Theta2 para as dimensões originais
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));



%%  Depois de treinar a rede neural, você deve utilizar a função predict para 
%%  predizer as labels do conjunto de treinamento
pred = predict(Theta1, Theta2, X);
a=mean(double(pred == y)) * 100;
vet(i) = a;
fprintf('\nAcurácia de Treinamento: %f\n', a);

%% Se eu rodar 10x, por exemplo, a taxa de acurácia será a mesma?
%% Cálcule a média, o máximo e o mínino da taxa de acurácia para 100 repetições 
%% do treinamento?    
endfor

fprintf('\n Minino de acuracia eh %f', min(vet));
fprintf('\n Maximo de acuracia eh %f' ,max(vet));
fprintf('\n Media de acuracia eh %f \n', mean(vet));

fprintf('Aperte enter para encerrar.\n');
pause;


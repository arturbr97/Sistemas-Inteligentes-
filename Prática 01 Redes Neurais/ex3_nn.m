%% Inicialização
clear ; close all; clc

%% Definição da arquitetura da rede
input_layer_size  = 400;  
hidden_layer_size = 25;   
num_labels = 10;  % as labels foram definidas de 1 a 10, sendo o 10 atribuídas a classe 0   
                       

% Carregando os dados de treinamento
load('ex3data1.mat');
m = size(X, 1);

% Carregar as matrizes de pesos Theta1 e Theta2
load('ex3weights.mat');

%% ================= Tarefa: Implementar o feedforward Predict =================
%  Você deverá codificar o algoritmo propagação para frente (feedforward) dentro
%  dentro da função predict.m. Essa função será responsável por predizer as 
%  labels do conjunto de treinamento. Abaixo, vocês podem observar no código que 
%  será cálculada a acurácia de treinamento, após implementar o algoritmo de 
%  propagação para frente vocês devem encontrar uma acurácia em torno de 97.5% 

pred = predict(Theta1, Theta2, X);

fprintf('\n Acurácia de treinamento: %f\n', mean(double(pred == y)) * 100);

fprintf('Pressione enter para continuar \n');
pause;

%  Seleciona um exemplo randômicamente
rp = randperm(m);

for i = 1:m
 
    fprintf('\nApresenta na tela a imagem selecionada\n');
    displayData(X(rp(i), :));

    %realiza a predição com o exemplo selecionado
    pred = predict(Theta1, Theta2, X(rp(i),:));
    fprintf('\nValor predito pela rede neural: %d (Dígito %d)\n', pred, mod(pred, 10));
    
    s = input('Pressione enter para continuar e q para sair: ','s');
    if s == 'q'
      break
    end
end


function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y)
%   Cálcula o custo e a função gradiente da rede neural


% redimensiona as matrizes de peso Theta1 e Theta2
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

% Inicializar as variáveis
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

Y = eye(num_labels)(y, :);

% FeedForward e cáculo da função custo

a1 = [ones(m, 1), X];

z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2, 1), 1), a2];

z3 = a2 * Theta2';
a3 = sigmoid(z3);

% Cálculo da função Custo
cost = sum((-Y .* log(a3)) - ((1 - Y) .* log(1 - a3)), 2);
J = (1 / m) * sum(cost);


% Algoritmo Backpropagation

Delta1 = 0;
Delta2 = 0;

for t = 1:m
	
	a1 = [1; X(t, :)']; 
	z2 = Theta1 * a1;
	a2 = [1; sigmoid(z2)]; 

	z3 = Theta2 * a2;
	a3 = sigmoid(z3);

	% Cálculo do erro associado a última camada
	d3 = a3 - Y(t, :)';
	
	% Cálculo do erro associado a camada escondida
	d2 = (Theta2(:, 2:end)' * d3) .* sigmoidGradient(z2);

	% Acumulando os erros
	Delta2 += (d3 * a2');
	Delta1 += (d2 * a1');
endfor

% Calculando o gradiente
Theta1_grad = (1 / m) * Delta1;
Theta2_grad = (1 / m) * Delta2;

grad = [Theta1_grad(:) ; Theta2_grad(:)];














% -------------------------------------------------------------

% =========================================================================

% Colocando as matrizes de peso em formato de vetor
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%Layer 1
%Add bias to original dataset
a1=[ones(m,1) X];

%Layer 2
%Now apply a1*Theta1
z2 = a1*Theta1.';
%Apply sigmoid
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];

%Layer 3
z3 = a2*Theta2.';
a3 = sigmoid(z3);

%Now the output layer is filled, we must compute the cost

%First step, put y into a handy form
nbClass = size(Theta2,1);
order = [1:nbClass];
order = repmat(order,m,1);
% 1 2 3
% 1 2 3
% 1 2 3
% 1 2 3

y=repmat(y,1,nbClass);
% 1 1 1
% 3 3 3
% 1 1 1
% 2 2 2

y=(order==y);
% 1 0 0
% 0 0 1
% 1 0 0
% 0 1 0

%Apply cost formula
costMatrix = -y.*log(a3) - (ones(size(y))-y).*log(ones(size(a3))-a3);
J=(1/m)*sum(sum(costMatrix)) ;


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

%Compute the delta of the last layer
delta3 = a3-y;

%Compute the delta of layer 2
sigmoidGrad = [ones(m,1) sigmoidGradient(z2)];
delta2= delta3 * Theta2 .* sigmoidGrad;
%There should be no delta on the bias unit
delta2 = delta2(:,2:size(delta2,2));

%Compute gradient
Theta1_grad = (1/m) * delta2.'*a1;
Theta2_grad = (1/m) * delta3.'*a2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Discardin the bias unit contribution in regularization
regulTheta1 = Theta1;
regulTheta1(:,1)=0;
regulTheta2 = Theta2;
regulTheta2(:,1)=0;

Theta1_grad = Theta1_grad + (lambda/m)*regulTheta1;
Theta2_grad = Theta2_grad + (lambda/m)*regulTheta2;

regularization = sum(sum(regulTheta1.^2)) + ...
		sum(sum(regulTheta2.^2)) ;
J = J + (lambda/(2*m))*regularization ;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

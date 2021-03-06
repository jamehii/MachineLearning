function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

theta2 = theta(2:end);

hypothesis = sigmoid(X * theta);
logPositive = -y .* log(hypothesis);
logNegative = -(1 - y).* log(1 - hypothesis);
regularization = (lambda * sum(theta2 .^ 2) / (2*m));

J = sum(logPositive .+ logNegative) / m + regularization;

grad(1) = sum( (hypothesis - y) .* X(: , 1) ) / m;

for i = 2:size(theta)
  grad(i) = sum( (hypothesis - y) .* X(: , i) ) / m + (lambda/m * theta(i));  
endfor


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end

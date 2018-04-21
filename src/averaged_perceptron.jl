
function trainer(model::AveragedPerceptron{T},
	              X::AbstractArray{T},
        			  Y::Vector{T}) where T<:AbstractFloat

   shuffle_epoch = model.shuffle_epoch
   random_state  = model.random_state
   max_epochs    = model.max_epochs

   if random_state!=-1
      srand(random_state)
   end

   n,m         = size(X)
   X           = hcat(X,ones(n,1)) # adding bias
   history     = []
   nerrors,nlast_errors = Inf,0
   epochs      = 0
   Θ,α         = rand(m+1),model.α
   step        = float(n*max_epochs)
   #while  nerrors>0 && epochs < max_epochs
   while  epochs < max_epochs
   # stops when error is equal to zero or grater than last_error or reached max iterations
       # shuffle dataset
       if shuffle_epoch
          sind = shuffle(1:n)
          x = X[sind,:]
          y = Y[sind]
       end
       nerrors = 0
       # weight updates for all samples
       for i=1:n
          xi = x[i,:]
          ξ   = sinal(Θ'*xi) - y[i]
          if ξ!=0
             nerrors+=1
             Θ = Θ - (step/(n*max_epochs))* α * ξ * xi
          end
          step = step - 1
		 end
       nlast_errors   = nerrors
       epochs+=1
       push!(history,nerrors)
   end
   if nerrors > 0
      warn("Perceptron: Not converged. Max epochs $(max_epochs) reached. Error history: $(history) \n Try to increase max_epochs or may be you have a non linear problem.")
   end
   model.Θ = Θ
   model.history = history
end

function predictor(model::AveragedPerceptron{T},
	                X::AbstractArray{T}) where T<:AbstractFloat

   Θ = model.Θ
   α = model.α

   n   = size(X,1)
   y   = zeros(Real,n)
   X   = hcat(X,ones(n,1)) # adding bias
   for i=1:n
      y[i] = sinal(Θ'*X[i,:])
   end
   y

end

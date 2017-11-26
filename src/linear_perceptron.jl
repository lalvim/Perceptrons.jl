# use in linear perceptron
@inline   h(Θ,x) = sinal(Θ'*x)


function trainer{T<:AbstractFloat}(model::LinearPerceptron{T},
	                              X::AbstractArray{T},
        								   Y::Vector{T})

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
   Θ           = rand(m+1)   #  already with bias
   α           = model.α   #  learning rate
   while  nerrors>0 && epochs < max_epochs
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
          ξ   = h(Θ,xi) - y[i]
          if ξ!=0
             nerrors+=1
			    Θ = Θ - α * ξ * xi
          end
		 end
       nlast_errors   = nerrors
       epochs+=1
       push!(history,nerrors)
   end
   if nerrors > 0
      warn("Perceptron: Not converged. Max epochs $(max_epochs) reached. Error history: $(history) \n Try to increase max_epochs or may be you have a non linear problem.")
   end
   model.Θ = Θ
   model.α = α
   model.history = history
end

function predictor{T<:AbstractFloat}(model::LinearPerceptron{T},
	                                    X::AbstractArray{T})

   Θ = model.Θ
   α = model.α

   n,m = size(X)
   y   = zeros(Real,n)
   X   = hcat(X,ones(n,1)) # adding bias
   for i=1:n
       y[i] = h(Θ,X[i,:])
   end
   y

end

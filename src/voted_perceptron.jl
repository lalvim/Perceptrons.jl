


@inline function vote(Θ,x,c,k)

   s = 0
   for j=1:k
       s += c[j]*sign(Θ[j]'*x) # voting (+1 or -1 * c[j] weight)
   end
   s
end

function trainer(model::VotedPerceptron{T},
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
   k,Θ,c,α     = 1,Dict(1=>rand(m+1)),Dict(1=>0),model.α
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
          ξ   = sinal(Θ[k]'*xi) - y[i]
          if ξ==0
             c[k] += 1
          else
             nerrors+=1
             c[k+1] = 1
			    Θ[k+1] = Θ[k] - α * ξ * xi
             k     += 1
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
   model.c = c
   model.k = k
   model.history = history
end

function predictor(model::VotedPerceptron{T},
	                X::AbstractArray{T}) where T<:AbstractFloat

   Θ = model.Θ
   α = model.α
   k = model.k
   c = model.c

   n   = size(X,1)
   y   = zeros(Real,n)
   X   = hcat(X,ones(n,1)) # adding bias
   for i=1:n
      y[i] = sinal(vote(Θ,X[i,:],c,k))
   end
   y

end

# A gaussian kernel function
@inline function Φ{T<:AbstractFloat}(x::Vector{T},
                                     y::Vector{T},
                                     r::T=1.0)
    n  = 1.0 / sqrt(2π*r)
    s  = 1.0 / (2r^2)
    return n*exp(-s*sum((x.-y).^2))
end

# A kernel matrix
function ΦΦ{T<:AbstractFloat}(X::AbstractArray{T},
                              r::T=1.0)
    n = size(X,1)
    K = zeros(n,n)
    for i=1:n
        for j=1:i
            K[i, j] = Φ(X[i, :], X[j, :],r)
            K[j, i] = K[i, j]
        end
        K[i, i] = Φ(X[i, :], X[i, :],r)
    end
    K
end

# A kernel matrix for test data
function ΦΦ{T<:AbstractFloat}(X::AbstractArray{T},
                              Z::AbstractArray{T},
                              r::T=1.0)
    (nx,mx)    = size(X)
    (nz,mz)    = size(Z)
    K          = zeros(T,nz, nx)
    for i=1:nz
        for j=1:nx
            K[i, j] = Φ(Z[i, :], X[j, :],r)
        end
    end
    K
end

@inline function ∑(λ,y,n,K,j)
    #    sum = .0
    #    for i=1:n
    #        sum += λ[i]*y[i]*K[i,j]
    #    end
    #    return sum
    return sum(λ .* y .* K[:,j])
end

@inline function sign(val)
    return  (val >=0 ? 1.0: 0.0 )
end

function trainer{T<:AbstractFloat}(model::LinearPerceptron{T},
	                              X::AbstractArray{T},
        								   Y::Vector{T})

   shuffle_epoch = model.shuffle_epoch
   random_state  = model.random_state
   max_epochs    = model.max_epochs

   if random_state!=-1
      srand(random_state)
   end

   K           = ΦΦ(X,model.width)

   n,m         = size(X)
   X           = hcat(X,ones(n,1)) # adding bias
   history     = []
   nerrors,nlast_errors = Inf,0
   epochs      = 0
   Θ           = model.Θ     #  already with bias
   α           = model.α    #  learning rate
   λ           = model.λ    # langrange multipliers
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
          xi   = x[i,:]
          yi   = y[i]
          yp   = sign(∑(λ,y,n,K,i))
          if yi!=yp
             nerrors+=1
			 λ[i] += 1
          end
		 end
       nlast_errors   = nerrors
       epochs+=1
       push!(history,nerrors)
   end
   if nerrors > 0
      warn("Kernel Perceptron: Not converged. Max epochs $(max_epochs) reached. Error history: $(history) \n Try to increase max_epochs or may be you have a non linear problem.")
   end
   model.Θ = Θ
   model.λ = λ
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
